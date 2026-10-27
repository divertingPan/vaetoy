#ifndef MCU_CODE_VAE_ENGINE_H
#define MCU_CODE_VAE_ENGINE_H

#include <stdint.h>

#define VAE_LATENT_DIM 16
#define VAE_IMAGE_SIZE 16

#ifdef __cplusplus
extern "C" {
#endif

// 初始化VAE引擎
void vae_init();

// 获取某数字的几何均值中心锚点 (固定字形)
void vae_get_anchor(int digit, float z_out[VAE_LATENT_DIM]);

// 从潜空间的真实多维高斯分布 N(mu, sigma^2) 进行连续有界随机采样
// diversity_scale: 笔画多样性缩放因子 (推荐 0.5 ~ 0.75，默认 0.65)
// 即使连续对同一数字调用，每次也会产生完全不同的全新手写体潜向量！
void vae_sample_digit(int digit, float z_out[VAE_LATENT_DIM], float diversity_scale);

// VAE Decoder 前向推理：输入 8 维潜向量 z，输出 16x16 灰度图像矩阵（范围 [0.0, 1.0]）
void vae_decode(const float z[VAE_LATENT_DIM], float output_img[VAE_IMAGE_SIZE][VAE_IMAGE_SIZE]);

#ifdef __cplusplus
}
#endif

#endif // MCU_CODE_VAE_ENGINE_H

