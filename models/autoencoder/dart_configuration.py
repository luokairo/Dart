from typing import Optional, Tuple

from transformers import PretrainedConfig

__all__ = ["DARTAutoEncoderConfig", "DARTAutoEncoderWithDiscConfig"]

class DARTAutoEncoderConfig(PretrainedConfig):
    model_type = "dart_autoencoder"

    def __init__(
        self,
        vocab_size=4096,
        z_channels=32,
        ch=160,
        dropout=0.0,
        beta=0.25,
        using_znorm=False,
        quant_conv_ks=3,
        quant_resi=0.5,
        share_quant_resi=4,
        default_qresi_counts=0,
        v_patch_nums=(1, 2, 3, 4, 6, 9, 13, 18, 24, 32),
        test_mode=False,
        ch_mult=(1, 1, 2, 2, 4),
        levels=[8, 8, 8, 6, 5],
        quantizer_type: str = "var_hybrid",
        hybrid: bool = False,
        disable_quant_resi: bool = False,
        freeze_codebook_for_hybrid: bool = True,
        double_decoder=False,
        text_model_path = "/fs/scratch/PAS2473/ICML2025/hart/hart/Qwen2-VL-1.5B-Instruct",
        context_token=300,
        context_dim=1536,
        
        **kwargs,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.z_channels = z_channels
        self.ch = ch
        self.dropout = dropout
        self.beta = beta
        self.using_znorm = using_znorm
        self.quant_conv_ks = quant_conv_ks
        self.quant_resi = quant_resi
        self.share_quant_resi = share_quant_resi
        self.default_qresi_counts = default_qresi_counts
        self.v_patch_nums = v_patch_nums
        self.test_mode = test_mode
        self.ch_mult = ch_mult
        self.levels = levels
        self.quantizer_type = quantizer_type
        self.hybrid = hybrid
        self.disable_quant_resi = disable_quant_resi
        self.freeze_codebook_for_hybrid = freeze_codebook_for_hybrid
        self.double_decoder = double_decoder
        self.text_model_path = text_model_path
        self.context_dim = context_dim
        self.context_token = context_token


class DARTAutoEncoderWithDiscConfig(DARTAutoEncoderConfig):
    model_type = "dart_autoencoder_with_disc"

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)