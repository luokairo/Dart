"""
Adopted from Dart
"""
"""
References:
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L110
- GumbelQuantize
: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
- VQVAE (VQModel): https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/autoencoder.py#L14
"""
from pickle import FALSE
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from dart.models.autoencoder.dart_configuration import DARTAutoEncoderConfig

from dart.models.autoencoder.quantize.hart_hybrid_quantize import (
    HARTHybridQuantizer,
)
from dart.models.autoencoder.quantize.dart_quantize import (
    DartHybridQuantizer
)
from dart.models.autoencoder.quantize.var_quantize_multiple_res import (
    VectorQuantizer2 as VARQuantizer,
)
from dart.models.networks.basic_vae import Decoder, Encoder
from dart.models.router.mlp_router import DARTRouterMlp
from transformers import (
    AutoModel,
    AutoTokenizer
)



class DARTAutoEncoder(PreTrainedModel):
    config_class = DARTAutoEncoderConfig

    def __init__(
        self,
        config: DARTAutoEncoderConfig
    ):
        super().__init__(config)

        vocab_size = config.vocab_size
        z_channels = config.z_channels
        ch = config.ch
        dropout = config.dropout
        beta = config.beta
        using_znorm = config.using_znorm
        quant_conv_ks = config.quant_conv_ks
        quant_resi = config.quant_resi
        share_quant_resi = config.share_quant_resi
        default_qresi_counts = config.default_qresi_counts
        v_patch_nums = config.v_patch_nums
        test_mode = config.test_mode
        freeze_codebook_for_hybrid = config.freeze_codebook_for_hybrid

        text_model_path = config.text_model_path
        context_token = config.context_token
        context_dim = config.context_dim

        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        # ddconfig is copied from https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/models/first_stage_models/vq-f16/config.yaml
        ddconfig = dict(
            dropout=dropout,
            ch=ch,
            z_channels=z_channels,
            in_channels=3,
            ch_mult=config.ch_mult,
            num_res_blocks=2,  # from vq-f16/config.yaml above
            using_sa=True,
            using_mid_sa=True,  # from vq-f16/config.yaml above
            # resamp_with_conv=True,   # always True, removed.
        )
        ddconfig.pop("double_z", None)
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig["ch_mult"]) - 1)

        self.quant_conv = torch.nn.Conv2d(
            self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2
        )
        self.post_quant_conv = torch.nn.Conv2d(
            self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2
        )

        self.double_decoder = config.double_decoder
        if self.double_decoder:
            assert config.quantizer_type == "var_hybrid"
            self.decoder2 = Decoder(**ddconfig)
            self.post_quant_conv2 = torch.nn.Conv2d(
                self.Cvae,
                self.Cvae,
                quant_conv_ks,
                stride=1,
                padding=quant_conv_ks // 2,
            )
        
        if config.quantizer_type == "var":
            self.quantize: VARQuantizer = VARQuantizer(
                vocab_size=vocab_size,
                Cvae=self.Cvae,
                using_znorm=using_znorm,
                beta=beta,
                default_qresi_counts=default_qresi_counts,
                v_patch_nums=v_patch_nums,
                quant_resi=quant_resi,
                share_quant_resi=share_quant_resi,
                disable_quant_resi=config.disable_quant_resi,
            )
        elif config.quantizer_type == "var_hybrid":
            self.quantize: DartHybridQuantizer = DartHybridQuantizer(
                vocab_size=vocab_size,
                Cvae=self.Cvae,
                using_znorm=using_znorm,
                beta=beta,
                default_qresi_counts=default_qresi_counts,
                v_patch_nums=v_patch_nums,
                quant_resi=quant_resi,
                share_quant_resi=share_quant_resi,
            )
            if freeze_codebook_for_hybrid:
                self.encoder.requires_grad_(False)
                if self.double_decoder:
                    self.decoder2.requires_grad_(False)
                    self.post_quant_conv2.requires_grad_(False)
                self.quantize.requires_grad_(False)
                self.quant_conv.requires_grad_(False)
            else:
                print("Codebook will be tuned.")
        
        if test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
        # router (Jingyi):
        self.router = DARTRouterMlp()
        
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(
        self,
        input,
        context_tensor,
        ret_usages=True,
        patch_nums=None,
        skip_continuous_prob=0.33,
        use_router=True,
    ):
        con_h_BChw, usages, vq_loss, _ = self.quantize(
            self.quant_conv(self.encoder(input)),
            patch_nums=patch_nums,
            ret_usages=ret_usages,
            return_all_codes=False,
            skip_continuous_prob=0.0,
        )

        dis_h_BChw, usages, vq_loss, _ = self.quantize(
            self.quant_conv(self.encoder(input)),
            patch_nums=patch_nums,
            ret_usages=ret_usages,
            return_all_codes=False,
            skip_continuous_prob=1.0,
        )
        if use_router:
            h_BChw, routing_outcome = self.router(
                con_h_BChw,
                dis_h_BChw,
                context_tensor
            )
        else:
            p = random.random()
            if p < skip_continuous_prob:
                h_BChw = dis_h_BChw
            elif p < 2 * skip_continuous_prob:
                h_BChw = 0.5 * dis_h_BChw + 0.5 * con_h_BChw
            else:
                h_BChw = con_h_BChw

        out = self.decoder(self.post_quant_conv(h_BChw))

        return {
            "out": out,
            "usages": usages,
            "vq_loss": vq_loss,
            "routing_outcome": routing_outcome if routing_outcome is not None else None,
        }
    
    def forward_with_double_decoder(
        self,
        inp,
        ret_usages=True,
        patch_nums=None,
        skip_continuous_prob=0.33,
    ):
        p = random.random()
        if p < skip_continuous_prob:
            h_BChw, usages, vq_loss, _ = self.quantize(
                self.quant_conv(self.encoder(inp)),
                ret_usages=ret_usages,
                patch_nums=patch_nums,
                return_all_codes=False,
                skip_continuous_prob=1.0,
            )
            out = self.decoder2(self.post_quant_conv2(h_BChw))
        else:
            h_BChw, usages, vq_loss, _ = self.quantize(
                self.quant_conv(self.encoder(inp)),
                ret_usages=ret_usages,
                patch_nums=patch_nums,
                return_all_codes=False,
                skip_continuous_prob=0.0,
            )
            out = self.decoder(self.post_quant_conv(h_BChw))
        return {
            "out": out,
            "usages": usages,
            "vq_loss": vq_loss,
        }
    # ===================== `forward` is only used in VAE training =====================

    def encode(self, inp):
        z = self.encoder(inp)
        z_q = self.quantize(self.quant_conv(z))

        return z_q
    
    def compute_loss(self, out, usages, vq_loss, recon_weight=1.0, xs=None):
        loss_recon = F.mse_loss(out, xs, reduction="mean")

        loss_latent = vq_loss

        loss_total = loss_recon * recon_weight + loss_latent

        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_latent": loss_latent,
        }
    
    def fhat_to_img(self, f_hat: torch.Tensor):
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)

    def img_to_idxBl(
        self,
        inp_img_no_grad: torch.Tensor,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        exception_stages: Optional[Dict[int, torch.Tensor]] = None,
    ) -> List[torch.LongTensor]:  # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad)) # (B, Cvae, last_pn, last_pn)
        return self.quantize.f_to_idxBl_or_fhat(
            f,
            to_fhat=False,
            v_patch_nums=v_patch_nums,
            exception_stages=exception_stages,
        )

    def img_to_x_BLCv(
        self,
        inp_img_no_grad: torch.Tensor,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    )  -> torch.Tensor:
        f = self.quant_conv(self.ecnoder(inp_img_no_grad))
        B, C, H, W = f.shape
        f_no_grad = f.detach()
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)]
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f'{patch_hws[-1]=} != ({H=}, {W=})'

        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws): # from small to large
            if 0 <= self.prog_si < si: break
    
    def img_to_idxBl_and_frest(
        self,
        inp_img_no_grad: torch.Tensor,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    ) -> List[torch.LongTensor]:  # return List[Bl]
        assert self.config.quantizer_type == "var_hybrid"
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_and_frest(
            f, to_fhat=False, v_patch_nums=v_patch_nums
        )
    
    def img_to_gt_BLCv(
        self,
        inp_img_no_grad: torch.Tensor,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    ) -> List[torch.LongTensor]:
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_gt_BLCv(f, v_patch_nums=v_patch_nums)

    def idxBl_to_img(
        self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l**0.5)
            ms_h_BChw.append(
                self.quantize.embedding(idx_Bl)
                .transpose(1, 2)
                .view(B, self.Cvae, pn, pn)
            )
        return self.embed_to_img(
            ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one
        )

    def idxBl_to_fhat(
        self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l**0.5)
            ms_h_BChw.append(
                self.quantize.embedding(idx_Bl)
                .transpose(1, 2)
                .view(B, self.Cvae, pn, pn)
            )

        return self.quantize.embed_to_fhat(
            ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one
        )
    
    # Jingyi: To get gt_Bl_dis
    def idxBl_to_gt(
        self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False
    ) -> torch.Tensor:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l**0.5)
            ms_h_BChw.append(
                self.quantize.embedding(idx_Bl)
            )
        gt_Bl_dis = torch.cat(ms_h_BChw, dim=1)
        return gt_Bl_dis

    def embed_to_img(
        self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(
                self.post_quant_conv(
                    self.quantize.embed_to_fhat(
                        ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True
                    )
                )
            ).clamp_(-1, 1)
        else:
            return [
                self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
                for f_hat in self.quantize.embed_to_fhat(
                    ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False
                )
            ]

    def img_to_reconstructed_img(
        self,
        x,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        last_one=False,
    ) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(
            f, to_fhat=True, v_patch_nums=v_patch_nums
        )
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [
                self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
                for f_hat in ls_f_hat_BChw
            ]
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if (
            "quantize.ema_vocab_hit_SV" in state_dict
            and state_dict["quantize.ema_vocab_hit_SV"].shape[0]
            != self.quantize.ema_vocab_hit_SV.shape[0]
        ):
            state_dict["quantize.ema_vocab_hit_SV"] = self.quantize.ema_vocab_hit_SV
        super().load_state_dict(state_dict=state_dict, strict=strict)

        if self.double_decoder:
            self.decoder2 = deepcopy(self.decoder)
            self.post_quant_conv2 = deepcopy(self.post_quant_conv2)

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
AutoConfig.register("dart_autoencoder", DARTAutoEncoderConfig)
AutoModel.register(DARTAutoEncoderConfig, DARTAutoEncoder)
