# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
#   LlamaGen: https://github.com/FoundationVision/LlamaGen/
from ast import arg
from math import fabs
import os
from cycler import V
from numpy import dtype
from sympy import im
import torch
import torch.amp
from zmq import device

from dart.utils import data

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import os
import time
import argparse
from glob import glob
from copy import deepcopy

from dart.models.autoencoder import (
    DartHybridQuantizer,
    DARTAutoEncoder,
    DARTAutoEncoderWithDisc
)
from dart.training.vq_loss import VQLoss
from dart.utils.distributed import init_distributed_mode
from dart.utils.ema import update_ema, requires_grad
from dart.utils.logger import create_logger
from dart.dataset.augmentation import random_crop_arr
from dart.dataset.build import build_dataset
from tqdm import tqdm

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains model
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device=device)

    # Setup an experiment folder:
    # Jingyi: will be changed
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.vq_model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        time_record = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        cloud_results_dir = f"{args.cloud_save_path}/{time_record}"
        cloud_checkpoint_dir = f"{cloud_results_dir}/{experiment_index:03d}-{model_string_name}/checkpoints"
        os.makedirs(cloud_checkpoint_dir, exist_ok=True)
        logger.info(f"Experiment directory created in cloud at {cloud_checkpoint_dir}")
        
    else:
        logger = create_logger(None)

    # training args
    logger.info(f"{args}")

    # training env
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    vae_model = DARTAutoEncoderWithDisc.from_pretrained(args.vae_path, ignore_mismatched_sizes=True).vae
    logger.info(f"VQ Model Parameters: {sum(p.numel() for p in vae_model.parameters()):,}")
    
    if args.ema:
        ema = deepcopy(vae_model).to(device)
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    vae_model = vae_model.to(device)
    
    vq_loss = VQLoss(
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        disc_type=args.disc_type,
        disc_loss=args.disc_loss,
        gen_adv_loss=args.gen_loss,
        image_size=args.image_size,
        perceptual_weight=args.perceptual_weight,
        reconstruction_weight=args.reconstruction_weight,
        reconstruction_loss=args.reconstruction_loss,
        codebook_weight=args.codebook_weight,
    ).to(device)
    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision =='fp16'))

    # Setup optimizer
    if not args.finetune_decoder:
        logger.info("Optimizing all parameters.")
        if args.kmeans:
            vqgan_parts_params = list(vae_model.encoder.parameters()) + \
                list(vae_model.decoder.parameters()) + \
                list(vae_model.quant_conv.parameters()) + \
                list(vae_model.post_quant_conv.parameters()) + \
                list(vae_model.quantize.embedding.parameters())
            all_params = list(vae_model.parameters())
            vqgan_parts_params_set = set(vqgan_parts_params)
            other_params = [p for p in all_params if p not in vqgan_parts_params_set]

            param_groups = [
                {'params': vqgan_parts_params, 'lr': args.lr * 10.0},  # use 1e-4 if kmeans
                {'params': other_params, 'lr': args.lr}  # use 1e-5 if kmeans
            ]
            logger.info(f"use keans, args.lr = {args.lr}")
            optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=1e-8, betas=(0.9, 0.99))
            optimizer_disc = torch.optim.AdamW(vq_loss.discriminator.parameters(), lr=args.lr * 10.0, eps=1e-8, betas=(0.9, 0.99))

        else:
            logger.info(f"no use keans, args.lr = {args.lr}")
            optimizer = torch.optim.Adam(vae_model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
            optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    else:
        logger.info("Optimizing only decoder parameters.")
        optimizer = torch.optim.Adam(vae_model.decoder.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        optimizer_disc = torch.optim.Adam(vq_loss.discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    
    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = build_dataset(args, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare model for training:
    if args.vae_ckpt:
        checkpoint = torch.load(args.vae_ckpt, map_location="cpu")
        model_state = checkpoint["model"]
        if args.finetune_decoder and args.enhanced_decoder:
            # if finetuning with enhanced decoder, you would expect the old shape not match
            try:
                # if you want to continue finetune the enhanced decoder
                missing, unexpected = vae_model.load_state_dict(model_state, strict=False)
                logger.info(f"Missing keys: {missing}")
                logger.info(f"Unexpected keys: {unexpected}")
            except:
                # if switching from small decoder to enhanced decoder, delete the old decoder keys first
                decoder_keys = [k for k in model_state.keys() if k.startswith("decoder.")]
                for k in decoder_keys:
                    del model_state[k]
                missing, unexpected = vae_model.load_state_dict(model_state, strict=False)
                logger.info(f"Missing keys: {missing}")
                logger.info(f"Unexpected keys: {unexpected}")
        else:
            vae_model.load_state_dict(model_state, strict=True)
            logger.info("Loaded model from checkpoint.")
        if args.ema:
            ema.load_state_dict(checkpoint["ema"])

        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except:
            logger.info("Optimizer starting from scratch.")
        try:
            vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        except:
            logger.info("Discriminator starting from scratch.")
        try:
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        except:
            logger.info("Discriminator optimizer starting from scratch.")

        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(
                args.vq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
            train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        else:
            train_steps = 0
            start_epoch = 0
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vae_model, decay=0)  # Ensure EMA is initialized with synced weights

    if args.finetune_decoder:
        for name, param in vae_model.named_parameters():
            if name.startswith("decoder."):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vae_model = torch.compile(vae_model)  # requires PyTorch 2.0

    vae_model = DDP(vae_model.to(device), device_ids=[args.gpu])
    vae_model.train()
    if args.ema:
        ema.eval()
    vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    logger.info(f"Training for {args.epochs} epochs...")
    # Initialize wandb
    if rank == 0:
        import wandb
        wandb.init(project="cloud-VQ", config=args)

    # Begin to trainging
    for epoch in tqdm(range(start_epoch, args.epochs)):
        sampler.set_epoch(epoch)
        logger.info(f"Begin epoch {epoch}...")

        for x, _ in tqdm(loader):
            imgs = x.to(device, non_blocking=True)

            # generator training
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                out = vae_model(imgs)
                recons_imgs, usages, codebook_loss = out["out"], out["usages"], out["vq_loss"]

                loss_gen, loss_dict_gen = vq_loss(
                    codebook_loss=codebook_loss,
                    usages=usages,
                    inputs=imgs,
                    reconstructions=recons_imgs,
                    optimizer_idx=0,
                    global_step=train_steps + 1,
                    last_layer=vae_model.module.decoder.last_layer,
                    logger=logger,
                    log_every=args.log_every,
                )

            scaler.scale(loss_gen).backward()

            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vae_model.parameters(), args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            if args.ema:
                update_ema(ema, vae_model.module._orig_mod if args.compile else vae_model)

            # discriminator training
            optimizer_disc.zero_grad()
            with torch.cuda.amp.autocast(dtype=ptdtype):
                loss_disc, loss_dict_disc = vq_loss(
                    codebook_loss=codebook_loss,
                    usages=usages,
                    inputs=imgs,
                    reconstructions=recons_imgs,
                    optimizer_idx=1,
                    global_step=train_steps + 1,
                    logger=logger,
                    log_every=args.log_every
                )
            scaler_disc.scale(loss_disc).backward()
            if args.max_grad_norm != 0.0:
                scaler_disc.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(vae_model.parameters(), args.max_grad_norm)
            scaler_disc.step(optimizer_disc)
            scaler_disc.update()

            # Log loss values:
            running_loss += loss_gen.item() + loss_disc.item()

            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                if rank == 0:
                    wandb.log({**loss_dict_gen, **loss_dict_disc}, step=train_steps)

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    if args.compile:
                        model_weight = vae_model.module._orig_mod.state_dict()
                    else:
                        model_weight = vae_model.module.state_dict()
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "discriminator": vq_loss.module.discriminator.state_dict(),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    if not args.no_local_save:
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")

                    cloud_checkpoint_path = f"{cloud_checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, cloud_checkpoint_path)
                    logger.info(f"Saved checkpoint in cloud to {cloud_checkpoint_path}")
                dist.barrier()

    vae_model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done")
    dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required data paths
    parser.add_argument("--data-path", type=str, required=True, default="", help="Path to training dataset") # should fill
    parser.add_argument("--cloud-save-path", type=str, default='', 
                      help='Cloud disk path for saving checkpoints') # should fill
    parser.add_argument("--results-dir", type=str, default="",
                      help="Local directory for saving results") # should fill

    # Model configuration
    parser.add_argument("--vq-model", type=str, default="DART_tokenizer")
    parser.add_argument("--vae-path", type=str, required=True, help="Path to pretrained VAE model") # should fill
    parser.add_argument("--vae-ckpt", type=str, default=None, help="Checkpoint path for resuming training")
    parser.add_argument("--ema", action='store_true', help="Whether to use EMA during training")
    parser.add_argument("--finetune", action='store_true', help="Whether to finetune a pre-trained model")
    parser.add_argument("--finetune-decoder", default=True, help="Whether to only finetune decoder")
    parser.add_argument("--enhanced-decoder", action='store_true', help="Whether to use enhanced decoder")
    parser.add_argument("--kmeans", default=True, help="Whether to use kmeans for codebook initialization")

    # Training parameters
    parser.add_argument("--dataset", type=str, default="dyb", help="Name of training dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.95, help="Beta2 for Adam optimizer")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--global-batch-size", type=int, default=512, help="Global batch size across all GPUs")
    parser.add_argument("--global-seed", type=int, default=0, help="Global random seed")
    parser.add_argument("--mixed-precision", type=str, default='none', 
                      choices=["none", "fp16", "bf16"], help="Mixed precision training mode")
    parser.add_argument("--compile", action='store_true', help="Whether to use torch.compile")
    parser.add_argument("--num-workers", type=int, default=64, help="Number of data loading workers")

    # Loss weights and configurations
    parser.add_argument("--disc-start", type=int, default=20000, help="Steps before starting discriminator training")
    parser.add_argument("--disc-weight", type=float, default=0.5, help="Weight for discriminator loss")
    parser.add_argument("--disc-type", type=str, default='patchgan', choices=['patchgan', 'stylegan'])
    parser.add_argument("--disc-loss", type=str, default='hinge', 
                      choices=['hinge', 'vanilla', 'non-saturating'])
    parser.add_argument("--gen-loss", type=str, default='hinge', choices=['hinge', 'non-saturating'])
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="Weight for perceptual loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="Weight for reconstruction loss")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="Type of reconstruction loss")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="Weight for codebook loss")

    # Image processing
    parser.add_argument("--image-size", type=int, default=256, choices=[224, 256, 384], 
                      help="Input image size")

    # Logging and checkpointing
    parser.add_argument("--log-every", type=int, default=100, help="Log frequency in steps")
    parser.add_argument("--ckpt-every", type=int, default=5000, help="Checkpoint saving frequency in steps")
    parser.add_argument("--no-local-save", action='store_true', 
                      help="Don't save checkpoints locally (for limited disk space)")
    parser.add_argument('--finetune_decoder', action='store_true', help='finetune decoder')
    args = parser.parse_args()
    main(args)


    



