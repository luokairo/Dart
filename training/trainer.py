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
from dart.models.router import (
    DARTRouterMlp,
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
        self.text_model.eval()
        self.max_token_length = max_token_length
        self.use_llm_system_prompt = use_llm_system_prompt
        # text embedding

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

            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW) # [(B, pn*pn) for pn in patch_nums]
            gt_Bl = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl) # 通过离散索引获得的不同大小的特征图
            

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
            self.dart_wo_ddp.forward
            logits_BLV = self.dart_wo_ddp(context_tenser, x_BLCv_wo_first_l, context_position_ids, context_mask)
        self.dart_wo_ddp.train(training)

        stats = L_mean.new_tensoer([L_mean.item(), L_tail.item(), acc_mean.item()], acc_tail.item(), tot)
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt

    def train_step(
        self, it: int, gt_it: int, stepping: bool, metric_lg: 
    )