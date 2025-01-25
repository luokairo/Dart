from ast import Tuple
import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from dart.utils.amp_sc import AmpOptimizer
from dart.utils.misc import MetricLogger, TensorboardLogger
from dart.models.autoencoder.quantize.dart_quantize import (
    DartHybridQuantizer
)
from sympy import training

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

import dist
from dart.models.transformer.dart_transformer import DARTForT2I
from dart.models.autoencoder import (
    DARTAutoEncoder,
    DartHybridQuantizer,
    DARTAutoEncoderWithDisc,
)

from dart.utils import encode_prompts, llm_system_prompt, default_prompts

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

class DARTTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: DARTAutoEncoder, dart_wo_ddp: DARTForT2I, dart: DDP,
        text_model_path: str, max_token_length: int, use_llm_system_prompt: bool,
        dart_opt: AmpOptimizer, label_smooth: float,
    ):
        super(DARTTrainer, self).__init__()

        self.dart, self.vae_local, self.quantize_local = dart, vae_local, vae_local.quantize
        self.quantize_local: DartHybridQuantizer
        self.dart_wo_ddp: DARTForT2I = dart_wo_ddp # after torch.compile
        self.dart_opt = dart_opt

        # text embedding (Jingyi):
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        self.text_model = AutoModel(text_model_path).to(device)
        # not finetune text_model
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.max_token_length = max_token_length
        self.use_llm_system_prompt = use_llm_system_prompt
        
        # router (Jingyi):
        self.router = self.dart_wo_ddp.router

        del self.dart_wo_ddp.rng
        self.dart_wo_ddp.rng = torch.Generator(device=device)

        self.label_smooth = label_smooth
        self.train_loss = nn.MSELoss(reduction='none')
        self.val_loss = nn.MSELoss(reduction='none')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weights = torch.ones(1, self.L, device=device) / self.L

        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

        self.prog_it = 0
        self.last_proj_si = -1
        self.first_prog = True

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        self.dart_wo_ddp.eval()
        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)

            f_BChw = self.vae_local.quant_conv(self.vae_local.encoder(inp_B3HW))
            last_layer_gt = f_BChw.view(f_BChw.shape[0], f_BChw.shape[1], -1).transpose(1, 2)
            x_BLCv_wo_first_l_con: Ten = self.quantize_local.f_to_var_input(f_BChw)

            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW) # [(B, pn*pn) for pn in patch_nums]
            gt_Bl_dis = self.vae_local.idxBl_to_gt(gt_idx_Bl)
            gt_Bl_con = self.quantize_local.f_to_x_BLCv(f_BChw)
            
            last_layer_gt_discrete = gt_idx_Bl[-1]
            x_BLCv_wo_first_l_dis: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl) # 通过离散索引获得的不同大小的特征图
            

            (
                context_tokens,
                context_mask,
                context_position_ids,
                context_tenser
            ) = encode_prompts(
                label_B,
                self.text_model,
                self.text_tokenizer,
                self.max_token_length,
                llm_system_prompt,
                self.use_llm_system_prompt,
            )
            x_BLCv_wo_first_l = self.router(x_BLCv_wo_first_l_con,
                                            x_BLCv_wo_first_l_dis,
                                            context_tenser)
            
            assert gt_Bl_con.shape[0] == gt_Bl_dis.shape[0], "Batch size mismatch between gt_Bl_con and gt_Bl_dis"
            assert gt_Bl_con.shape[2] == gt_Bl_dis.shape[2], "Feature size mismatch between gt_Bl_con and gt_Bl_dis"
            gt_Bl = self.router(gt_Bl_con, gt_Bl_dis, context_tenser)
            self.dart_wo_ddp.forward
            x_BLC_logits, last_stage_embed, mask_to_prev_stags = self.dart_wo_ddp(context_tenser, 
                                                                x_BLCv_wo_first_l, 
                                                                context_position_ids, 
                                                                context_mask,
                                                                last_layer_gt,
                                                                last_layer_gt_discrete)
            L_mean += self.val_loss(x_BLC_logits, gt_Bl).mean(dim=(1, 2)).sum()
            L_tail += self.val_loss(last_stage_embed, gt_Bl[:, -self.last_l:, :]).mean(dim=(1, 2)).sum()
            tot += B
        self.dart_wo_ddp.train(training)

        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, _ = stats.tolist()
        return L_mean, L_tail, tot, time.time()-stt

    def train_step(
        self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger, tb_lg: TensorboardLogger,
        inp_B3HW: FTen, label_B: Union[ITen, FTen], prog_si: int, prog_wp_it: float,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # if progressive training
        self.dart_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog
        
        # forward
        B, V = inp_B3HW.shape[0], self.vae_local.vocab_size
        self.dart.require_backward_grad_sync = stepping

        f_BChw = self.vae_local.quant_conv(self.vae_local.encoder(inp_B3HW)) # (B, Cvae, ph, pw)
        last_layer_gt = f_BChw.view(B, last_layer_gt.shape[-1], -1).transpose(1, 2) # (B, pn*pn, Cvae)
        x_BLCv_wo_first_l_con: Ten = self.quantize_local.f_to_var_input(f_BChw) # (B, L, Cvae)

        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_Bl_dis = self.vae_local.idxBl_to_gt(gt_idx_Bl)
        gt_Bl_con = self.quantize_local.f_to_x_BLCv

        last_layer_gt_discrete = gt_idx_Bl[-1]
        x_BLCv_wo_first_l_dis: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl) # 通过离散索引获得的不同大小的特征图
        
        (
            context_tokens,
            context_mask,
            context_position_ids,
            context_tenser
        ) = encode_prompts(
            label_B,
            self.text_model,
            self.text_tokenizer,
            self.max_token_length,
            llm_system_prompt,
            self.use_llm_system_prompt,
        )
        x_BLCv_wo_first_l = self.router(x_BLCv_wo_first_l_con,
                                        x_BLCv_wo_first_l_dis,
                                        context_tenser)
        
        assert gt_Bl_con.shape[0] == gt_Bl_dis.shape[0], "Batch size mismatch between gt_Bl_con and gt_Bl_dis"
        assert gt_Bl_con.shape[2] == gt_Bl_dis.shape[2], "Feature size mismatch between gt_Bl_con and gt_Bl_dis"
        gt_Bl = self.router(gt_Bl_con, gt_Bl_dis, context_tenser) # (B, L, Cvae)

        with self.dart_opt.amp_ctx:
            self.dart_wo_ddp.forward
            (
                x_BLC_logits,
                last_stage_embed,
                mask_to_prev_stages
            ) = self.dart_wo_ddp(
                context_tenser,
                x_BLCv_wo_first_l,
                context_position_ids,
                context_mask,
                last_layer_gt,
                last_layer_gt_discrete
            )
            
            loss = self.train_loss(x_BLC_logits, gt_Bl).mean(dim=2)
            mask = torch.zeros_like(loss)
            mask[mask_to_prev_stages] = 1
            loss = loss * mask
            if prog_si >= 0: # in progressive training
                bg, ed = self.begin_ends[prog_si]
                assert x_BLC_logits.shape[1] == gt_Bl.shape[1] == ed
                lw = self.loss_weights[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else: # not in progressive training
                lw = self.loss_weights
            loss = loss.mul(lw).sum(dim=-1).mean()
        
        # backward
        grad_norm, scale_log2 = self.dart_opt.backward_clip_step(loss=loss, stepping=stepping)

        # log
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(x_BLC_logits, gt_Bl).mean(dim=(1, 2)).sum().item()
            if prog_si >= 0: # in progressive training
                Ltail = -1
            else: # not in progressive training
                Ltail = self.val_loss(last_stage_embed, gt_Bl[:, -self.last_l:, :]).mean(dim=(1, 2)).sum().item()
            grad_norm = grad_norm.item()
            metric_lg.update(Lm=Lmean, Lt=Ltail, tnm=grad_norm)
        
        # log to tensorborad
        if g_it == 0 or (g_it + 1) % 500 == 0:
            print("log")
        
        self.dart_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2

    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('dart_wo_ddp', 'vae_local', 'dart_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state

    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('dart_wo_ddp', 'vae_local', 'dart_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[DartTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[DartTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[Dart.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
            






        
