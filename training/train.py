import gc
import os
import shutil
import sys
import time
import warnings
import string
import random
from functools import partial

import torch
from torch.utils.data import DataLoader

import dist
from utils import arg_util, misc
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume

def build_everything(args: arg_util.Args):
    # resume
    auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    # create tensorboard logger
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(misc.TensorboardLogger(log_dir=args.tb_log_dir_path, filename_suffix=f'__{misc.time_str("%m%d_%H%M")}'), verbose=True)
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)
    dist.barrier()

    # log args
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')

    # build data
    if not args.local_debug:
        print(f'[build PT data] ...\n')
        dataset_train, dataset_val = build_dataset(
            args.data_path, args.caption_path, args.data_load_reso, hflip=args.hflip, mid_reso=args.mid_reso,
        )
        types = str((type(dataset_train).__name__, type(dataset_val).__name__))

        ld_val = DataLoader(
            dataset_val, num_workers=0, pin_memory=True,
            batch_size=round(args.batch_size * 1.5), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
            shuffle=False, drop_last=False,
        )
        del dataset_val

        ld_train = DataLoader(
            dataset=dataset_train, num_workers=args.workers, pin_memory=True,
            generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
            batch_sampler=DistInfiniteBatchSampler(
                dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
                shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_ep=start_ep, start_it=start_it,
            ),
        )
        del dataset_train

        [print(line) for line in auto_resume_info]
        print(f'[dataloader multi processing] ...', end='', flush=True)
        stt = time.time()
        iters_train = len(ld_train)
        ld_train = iter(ld_train)
        # noinspection PyArgumentList
        print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)', flush=True, clean=True)
        print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, types(tr, va)={types}')
    
    else:
        num_classes = 1000
        ld_val = ld_train = None
        iters_train = 10

    # build models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from dart.models import DARTForT2I, DARTAutoEncoderWithDisc, DARTAutoEncoder, build_vae_dart
    from dart.training.trainer import DARTTrainer
    from dart.utils.amp_sc import AmpOptimizer
    from dart.utils.lr_control import filter_params
    from dart.models.transformer.configuration import DARTForT2IConfig
    from dart.models.autoencoder.dart_configuration import DARTAutoEncoderWithDiscConfig

    dart_autoencoder_config = DARTAutoEncoderWithDiscConfig()
    dart_config = DARTForT2IConfig()  
    vae_local, dart_wo_ddp = build_vae_dart(
        device=dist.get_device(), dart_autoencoder_config=dart_autoencoder_config,
        dart_config=dart_config
    )

    vae_ckpt = "/fs/scratch/PAS2473/ICML2025/result/vqvae/512/008-DART_tokenizer/checkpoints/0003600.pt"
    if dist.is_local_master():
        if not os.path.exists(vae_ckpt):
            os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}')
    dist.barrier()
    vae_checkpoint = torch.load(vae_ckpt, map_location='cpu')
    vae_local.load_state_dict(vae_checkpoint['model'], strict=True)

    dart_ckpt = "/fs/scratch/PAS2473/ICML2025/hart/hart/hart-0.7b-1024px/llm/pytorch_model.bin"
    dart_checkpoint =torch.load(dart_ckpt, map_location='cpu')

    checkpoint_state_dict = dart_checkpoint
    model_state_dict = dart_wo_ddp.state_dict()

    filtered_state_dict = {
        k: v for k, v in checkpoint_state_dict.items()
        if k in model_state_dict and model_state_dict[k].shape == v.shape
    }
    load_info = dart_wo_ddp.load_state_dict(filtered_state_dict, strict=False)

    print("Successfully loaded matching parameters!")
    if load_info.missing_keys:
        print("\nMissing keys (not found in the checkpoint):")
        for key in load_info.missing_keys:
            print(f"  - {key}")
    if load_info.unexpected_keys:
        print("\nUnexpected keys (found in the checkpoint but not in the model):")
        for key in load_info.unexpected_keys:
            print(f"  - {key}")

    vae_local: DARTAutoEncoder = args.compile_model(vae_local, args.vfast)
    dart_wo_ddp: DARTForT2I = args.compile_model(dart_wo_ddp, args.tfast)
    dart: DDP = (DDP if dist.initialized() else NullDDP)(dart_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)

    print(f'[INIT] DART model = {dart_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder), ('VAE.quant', vae_local.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('DART', dart_wo_ddp),)]) + '\n\n')

    # build optimizer
    names, paras, para_groups = filter_params(dart_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })
    opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=args.afuse),
    }[args.opt.lower().strip()]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')

    dart_optim = AmpOptimizer(
        mixed_precision=args.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
        grad_clip=args.tclip, n_gradient_accumulation=args.ac
    )
    del names, paras, para_groups

    # build trainer
    trainer = DARTTrainer(
        device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, dart_wo_ddp=dart_wo_ddp, dart=dart,
        text_model_path=args.text_model_path, max_token_length=args.max_token_length,
        use_llm_system_prompt=args.use_llm_system_prompt, dart_opt=dart_optim, 
        label_smooth=args.ls,
    )
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True)
    del vae_local, dart_wo_ddp, dart, dart_optim

    if args.local_debug:
        rng = torch.Generator('cpu')
        rng.manual_seed(0)
        B = 4
        inp = torch.rand(B, 3, args.data_load_reso, args.data_load_reso)
        labels = [''.join(random.choices(string.ascii_letters + string.digits, k=10)) for _ in range(B)]
        me = misc.MetricLogger(delimiter='   ')
        trainer.train_step(
            it=0, g_it=0, stepping=True, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=labels, prog_si=args.pg0, prog_wp_it=20
        )
        print({k: meter.global_avg for k, meter in me.meters.items()})

        args.dump_log(); tb_lg.flush(); tb_lg.close()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(sys.stderr, misc.SyncPrint):
            sys.stdout.close(), sys.stderr.close()
        exit(0)
    
    dist.barrier()
    return (
        tb_lg, trainer, start_ep, start_it,
        iters_train, ld_train, ld_val
    )

def main_training():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.local_debug:
        torch.autograd.set_detect_anomaly(True)
    
    (
        tb_lg, trainer,
        start_ep, start_it,
        iters_train, ld_train, ld_val
    ) = build_everything(args)

    # train
    start_time = time.time()
    best_L_mean, best_L_tail = 999., 999.
    best_val_loss_mean, best_val_loss_tail = 999, 999
    L_mean, L_tail = -1, -1
    for ep in range(start_ep, args.ep):
        if hasattr(ld_train, 'sampler') and hasattr(ld_train.sampler, 'set_epoch'):
            ld_train.sampler.set_epoch(ep)
            if ep < 3:
                # noinspection PyArgumentList
                print(f'[{type(ld_train).__name__}] [ld_train.sampler.set_epoch({ep})]', flush=True, force=True)
        tb_lg.set_step(ep * iters_train)

        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep, ep == start_ep, start_it if ep == start_ep else 0, args, tb_lg, ld_train, iters_train, trainer
        )

        L_mean, L_tail, grad_norm = stats['Lm'], stats['Lt'], stats['tnm']
        best_L_mean = min(best_L_mean, L_mean)
        if L_tail != -1: best_L_tail = min(best_L_tail, L_tail)
        args.L_mean, args.L_tail, args.grad_norm = L_mean, L_tail, grad_norm
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time

        AR_ep_loss = dict(L_mean=L_mean, L_tail=L_tail)
        is_val_and_also_saving = (ep + 1) % 10 == 0 or (ep + 1) == args.ep
        if is_val_and_also_saving:
            val_loss_mean, val_loss_tail, tot, cost = trainer.eval_ep(ld_val)
            best_updated = best_val_loss_tail > val_loss_tail
            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, val_loss_mean), min(best_val_loss_tail, val_loss_tail)
            AR_ep_loss.update(vL_mean=val_loss_mean, vL_tail=val_loss_tail)
            args.vL_mean, args.vL_tail = val_loss_mean, val_loss_tail
            print(f' [*] [ep{ep}]  (val {tot})  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Val cost: {cost:.2f}s')

            if dist.is_local_master():
                local_out_ckpt = os.path.join(args.local_out_dir_path, 'ar-ckpt-last.pth')
                local_out_ckpt_best = os.path.join(args.local_out_dir_path, 'ar-ckpt-best.pth')
                print(f'[saving ckpt] ...', end='', flush=True)
                torch.save({
                    'epoch':    ep+1,
                    'iter':     0,
                    'trainer':  trainer.state_dict(),
                    'args':     args.state_dict(),
                }, local_out_ckpt)
                if best_updated:
                    shutil.copy(local_out_ckpt, local_out_ckpt_best)
                print(f'     [saving ckpt](*) finished!  @ {local_out_ckpt}', flush=True, clean=True)
            dist.barrier()

        print(    f'     [ep{ep}]  (training )  Lm: {best_L_mean:.3f} ({L_mean:.3f}), Lt: {best_L_tail:.3f} ({L_tail:.3f}),  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        tb_lg.update(head='AR_ep_loss', step=ep+1, **AR_ep_loss)
        tb_lg.update(head='AR_z_burnout', step=ep+1, rest_hours=round(sec / 60 / 60, 2))
        args.dump_log(); tb_lg.flush()
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total cost: {total_time},   Lm: {best_L_mean:.3f} ({L_mean}),   Lt: {best_L_tail:.3f} ({L_tail})')
    print('\n\n')

    del stats
    del iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    print(f'final args:\n\n{str(args)}')
    args.dump_log(); tb_lg.flush(); tb_lg.close()
    dist.barrier()               


def train_one_ep(ep: int, is_first_ep: bool, start_it: int, args: arg_util.Args,
    tb_lg: misc.TensorboardLogger, ld_or_itri, iters_train: int, trainer             
    ):
    from trainer import DARTTrainer
    from dart.utils.lr_control import lr_wd_annealing
    trainer: DARTTrainer

    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    me.add_meter('tlr', misc.SmoothedValue(window_size=1, fmt='{value:.2g}'))
    me.add_meter('tnm', misc.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    [me.add_meter(x, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
    header = f'[Ep]: [{ep:4d}/{args.ep}]'

    if is_first_ep:
        warnings.filterwarnings('ignore')
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, max_it = ep * iters_train, args.ep * iters_train

    for it, (inp, label) in me.log_every(start_it, iters_train, 30 if iters_train > 8000 else 5, header):
        g_it = ep * iters_train + it
        if it < start_it: continue
        if is_first_ep and it == start_it: warnings.resetwarnings()

        inp = inp.to(args.device, non_blocking=True)
        label = label.to(args.device, non_blocking=True)

        args.cur_it = f'{it + 1}/{iters_train}'

        wp_it = args.wp * iters_train
        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.var_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
        args.cur_lr, args.cur_wd = max_tlr, max_twd

        if args.pg: # default: args.pg == 0.0, means no progressive training, won't get into this
            if g_it <= wp_it: prog_si = args.pg0
            elif g_it >= max_it*args.pg: prog_si = len(args.patch_nums) - 1
            else:
                delta = len(args.patch_nums) - 1 - args.pg0
                progress = min(max((g_it - wp_it) / (max_it*args.pg - wp_it), 0), 1) # from 0 to 1
                prog_si = args.pg0 + round(progress * delta)    # from args.pg0 to len(args.patch_nums)-1
        else:
            prog_si = -1

        stepping = (g_it + 1) % args.ac == 0
        step_cnt += int(stepping)

        grad_norm, scale_log2 = trainer.train_step(
            it=it, g_it=g_it, stepping=stepping, metric_lg=me, tb_lg=tb_lg,
            inp_B3HW=inp, label_B=label, prog_si=prog_si, prog_wp_it=args.pgwp * iters_train,
        )

        me.update(tlr=max_tlr)
        tb_lg.set_step(step=g_it)
        tb_lg.update(head='AR_opt_lr/lr_min', sche_tlr=min_tlr)
        tb_lg.update(head='AR_opt_lr/lr_max', sche_tlr=max_tlr)
        tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
        tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
        tb_lg.update(head='AR_opt_grad/fp16', scale_log2=scale_log2)
        
        if args.tclip > 0:
            tb_lg.update(head='AR_opt_grad/grad', grad_norm=grad_norm)
            tb_lg.update(head='AR_opt_grad/grad', grad_clip=args.tclip)
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    

