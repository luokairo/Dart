# build models
from torch.nn.parallel import DistributedDataParallel as DDP
from dart.models import build_vae_dart

from dart.utils.lr_control import filter_params
from dart.models.transformer.configuration import DARTForT2IConfig
from dart.models.autoencoder.dart_configuration import DARTAutoEncoderWithDiscConfig
import dart.dist as dist
import torch
import os

dart_autoencoder_config = DARTAutoEncoderWithDiscConfig()  # 创建实例
dart_config = DARTForT2IConfig()  # 同样实例化 dart_config

vae_local, dart_wo_ddp = build_vae_dart(
    device=dist.get_device(), dart_autoencoder_config=dart_autoencoder_config,
    dart_config=dart_config
)
vae_ckpt = "/fs/scratch/PAS2473/ICML2025/logs/vqgan/512/2025-01-21-09-40-44/008-DART_tokenizer/checkpoints/0012600.pt"
if dist.is_local_master():
    if not os.path.exists(vae_ckpt):
        os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}')
dist.barrier()

vae_checkpoint = torch.load(vae_ckpt, map_location='cpu')
vae_local.load_state_dict(vae_checkpoint['model'], strict=True)
print(vae_local)

# 加载 HART 模型的权重
hart_ckpt = "/fs/scratch/PAS2473/ICML2025/hart/hart/hart-0.7b-1024px/llm/pytorch_model.bin"
hart_checkpoint = torch.load(hart_ckpt, map_location='cpu')

# 检查点中的权重
checkpoint_state_dict = hart_checkpoint  # 确保检查点文件的键是 'model'

# 当前模型的权重
model_state_dict = dart_wo_ddp.state_dict()

# 保留匹配的权重
filtered_state_dict = {
    k: v for k, v in checkpoint_state_dict.items()
    if k in model_state_dict and model_state_dict[k].shape == v.shape
}

# 加载匹配的权重并记录未匹配的键
load_info = dart_wo_ddp.load_state_dict(filtered_state_dict, strict=False)

# 输出加载成功和未成功的信息
print("Successfully loaded matching parameters!")
if load_info.missing_keys:
    print("\nMissing keys (not found in the checkpoint):")
    for key in load_info.missing_keys:
        print(f"  - {key}")
if load_info.unexpected_keys:
    print("\nUnexpected keys (found in the checkpoint but not in the model):")
    for key in load_info.unexpected_keys:
        print(f"  - {key}")