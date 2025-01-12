import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from dart.models.autoencoder import (
    DartHybridQuantizer,
    DARTAutoEncoder,
    DARTAutoEncoderWithDisc
)
from skimage.metrics import structural_similarity as ssim
import numpy as np

# 加载自编码器模型
vae_path = "/Users/kairoliu/Documents/Dart/hart/tokenizer"
vae = DARTAutoEncoderWithDisc.from_pretrained(vae_path, ignore_mismatched_sizes=True).vae

# 加载图像并进行预处理
image_path = "/Users/kairoliu/Downloads/DYB-201203161316-0-118.jpg"  # 修改为你的图像路径
image = Image.open(image_path)

# 保存原始图像的尺寸
original_size = image.size

# 定义图像预处理（如调整大小、转换为张量、归一化）
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整为合适的输入尺寸
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 根据模型的训练标准化
])

image_tensor = transform(image).unsqueeze(0)  # 扩展维度以适应模型输入

# 将图像输入到编码器（加密）得到潜在表示
encoded = vae.encoder(image_tensor)  # 假设 `encode` 方法是模型编码器的一部分
print(encoded.shape)
encoded = vae.quant_conv(encoded)
print(encoded.shape)

# 使用 embed_to_img (离散tokens) 和 img_to_reconstructed_img (连续tokens) 重构图像
discrete_decoded_images = vae.decoder(vae.post_quant_conv(encoded)).clamp(-1, 1)
continuous = vae.quantize.f_to_idxBl_or_fhat(encoded, to_fhat=True)[-1]
continuous_decoded_images = vae.decoder(vae.post_quant_conv(0.5 * encoded + 0.5 * continuous).clamp(-1, 1))
# continuous_decoded_images = vae.img_to_reconstructed_img(image_tensor, last_one=True)  # 连续tokens

# 如果返回的解码图像是多个，选择最后一个重构图像进行显示
discrete_decoded_image = discrete_decoded_images[-1]
continuous_decoded_image = continuous_decoded_images[-1]

# 转换为 NumPy 数组并处理图像以便显示
discrete_decoded_image = discrete_decoded_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)  # 转换为 NumPy 数组
discrete_decoded_image = (discrete_decoded_image * 0.5 + 0.5) * 255  # 恢复到原始范围 [0, 255]
discrete_decoded_image = discrete_decoded_image.astype("uint8")

continuous_decoded_image = continuous_decoded_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)  # 转换为 NumPy 数组
continuous_decoded_image = (continuous_decoded_image * 0.5 + 0.5) * 255  # 恢复到原始范围 [0, 255]
continuous_decoded_image = continuous_decoded_image.astype("uint8")

# 将重构图像调整为与原始图像相同的大小
discrete_decoded_image_pil = Image.fromarray(discrete_decoded_image)
discrete_decoded_image_resized = discrete_decoded_image_pil.resize(original_size, Image.LANCZOS)

continuous_decoded_image_pil = Image.fromarray(continuous_decoded_image)
continuous_decoded_image_resized = continuous_decoded_image_pil.resize(original_size, Image.LANCZOS)

# 计算 MSE（均方误差）
original_image_np = np.array(image.resize(original_size))
discrete_mse = np.mean((original_image_np - discrete_decoded_image_resized) ** 2)
continuous_mse = np.mean((original_image_np - continuous_decoded_image_resized) ** 2)

# 计算 SSIM（结构相似性指数），并显式设置窗口大小和通道轴
discrete_ssim = ssim(
    original_image_np,
    np.array(discrete_decoded_image_resized),
    multichannel=True,
    win_size=3,  # 设置一个较小的窗口大小，适用于小尺寸图像
    channel_axis=2  # 指定颜色通道轴
)

continuous_ssim = ssim(
    original_image_np,
    np.array(continuous_decoded_image_resized),
    multichannel=True,
    win_size=3,  # 设置一个较小的窗口大小，适用于小尺寸图像
    channel_axis=2  # 指定颜色通道轴
)

# 输出重构指标
print(f"Discrete Tokens - Mean Squared Error (MSE): {discrete_mse:.4f}")
print(f"Discrete Tokens - Structural Similarity Index (SSIM): {discrete_ssim:.4f}")
print(f"Continuous Tokens - Mean Squared Error (MSE): {continuous_mse:.4f}")
print(f"Continuous Tokens - Structural Similarity Index (SSIM): {continuous_ssim:.4f}")

# 显示原始图像与两种重构图像
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

axes[1].imshow(discrete_decoded_image_resized)
axes[1].set_title("Continuous Tokens Reconstructed Image")
axes[1].axis('off')

axes[2].imshow(continuous_decoded_image_resized)
axes[2].set_title("Discrete Tokens Reconstructed Image")
axes[2].axis('off')

plt.show()