#include "mcu_code_vae_engine.h"
#include "mcu_code_decoder_weights.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#if defined(ESP32)
#include <Arduino.h>
#if __has_include(<esp_random.h>)
#include <esp_random.h>
#else
#include <esp_system.h>
#endif
#endif

// 静态内部缓存区（总计约 11KB，无需动态内存分配，防止单片机内存碎片）
static float fc_out[256];
static float x_4x4[48][4][4];
static float x_8x8[24][8][8];
static float x_16x16[16][16];

// Box-Muller 变换生成标准正态分布随机数 N(0, 1)
static float rand_gaussian() {
#if defined(ESP32)
    // 直接使用 ESP32/ESP32-S3 硬件真随机数发生器 (Hardware RNG)，熵源真实且不重复
    uint32_t r1 = esp_random();
    uint32_t r2 = esp_random();
    float u1 = ((float)(r1 % 100000) + 1.0f) / 100002.0f;
    float u2 = ((float)(r2 % 100000) + 1.0f) / 100002.0f;
#else
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
#endif
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307f * u2);
}

void vae_init() {
    memset(fc_out, 0, sizeof(fc_out));
    memset(x_4x4, 0, sizeof(x_4x4));
    memset(x_8x8, 0, sizeof(x_8x8));
    memset(x_16x16, 0, sizeof(x_16x16));
}

void vae_get_anchor(int digit, float z_out[VAE_LATENT_DIM]) {
    if (digit < 0) digit = 0;
    if (digit > 9) digit = 9;
    for (int i = 0; i < VAE_LATENT_DIM; i++) {
        z_out[i] = ((float)anchor_points[digit][i]) / 256.0f;
    }
}

void vae_sample_digit(int digit, float z_out[VAE_LATENT_DIM], float diversity_scale) {
    if (digit < 0) digit = 0;
    if (digit > 9) digit = 9;
    for (int i = 0; i < VAE_LATENT_DIM; i++) {
        float mean = ((float)anchor_points[digit][i]) / 256.0f;
        float std_val = ((float)anchor_stds[digit][i]) / 256.0f;
        float eps = rand_gaussian();
        // 有界截断（Truncated Normal）在 2 个标准差以内，确保字形稳定可辨，不产生离群坏点
        if (eps > 2.0f) eps = 2.0f;
        if (eps < -2.0f) eps = -2.0f;
        z_out[i] = mean + diversity_scale * std_val * eps;
    }
}

void vae_decode(const float z[VAE_LATENT_DIM], float output_img[VAE_IMAGE_SIZE][VAE_IMAGE_SIZE]) {
    // ------------------------------------------------------------------------
    // 第 1 步: 全连接层 FC (16 -> 256) + ReLU
    // ------------------------------------------------------------------------
    for (int i = 0; i < 256; i++) {
        float sum = ((float)layer1_fc_bias[i]) / 256.0f;
        const int16_t* w_row = &layer0_fc_weight[i * 16];
        for (int j = 0; j < 16; j++) {
            sum += z[j] * (((float)w_row[j]) / 256.0f);
        }
        fc_out[i] = sum > 0.0f ? sum : 0.0f;
    }

    // ------------------------------------------------------------------------
    // 第 2 步: 反卷积 Deconv1 (64x2x2 -> 48x4x4, k=3, s=2, p=1, op=1) + ReLU
    // ------------------------------------------------------------------------
    for (int oc = 0; oc < 48; oc++) {
        float b = ((float)layer3_deconv1_bias[oc]) / 256.0f;
        for (int h = 0; h < 4; h++) {
            for (int w = 0; w < 4; w++) {
                x_4x4[oc][h][w] = b;
            }
        }
    }

    for (int ic = 0; ic < 64; ic++) {
        for (int hi = 0; hi < 2; hi++) {
            for (int wi = 0; wi < 2; wi++) {
                float val = fc_out[ic * 4 + hi * 2 + wi];
                if (val <= 0.0f) continue; // ReLU 稀疏性加速跳过

                for (int oc = 0; oc < 48; oc++) {
                    const int16_t* k_base = &layer2_deconv1_weight[(ic * 48 + oc) * 9];
                    for (int kh = 0; kh < 3; kh++) {
                        int ho = hi * 2 - 1 + kh;
                        if (ho < 0 || ho >= 4) continue;
                        for (int kw = 0; kw < 3; kw++) {
                            int wo = wi * 2 - 1 + kw;
                            if (wo < 0 || wo >= 4) continue;
                            x_4x4[oc][ho][wo] += val * (((float)k_base[kh * 3 + kw]) / 256.0f);
                        }
                    }
                }
            }
        }
    }

    for (int oc = 0; oc < 48; oc++) {
        for (int h = 0; h < 4; h++) {
            for (int w = 0; w < 4; w++) {
                if (x_4x4[oc][h][w] < 0.0f) x_4x4[oc][h][w] = 0.0f;
            }
        }
    }

    // ------------------------------------------------------------------------
    // 第 3 步: 反卷积 Deconv2 (48x4x4 -> 24x8x8, k=3, s=2, p=1, op=1) + ReLU
    // ------------------------------------------------------------------------
    for (int oc = 0; oc < 24; oc++) {
        float b = ((float)layer5_deconv2_bias[oc]) / 256.0f;
        for (int h = 0; h < 8; h++) {
            for (int w = 0; w < 8; w++) {
                x_8x8[oc][h][w] = b;
            }
        }
    }

    for (int ic = 0; ic < 48; ic++) {
        for (int hi = 0; hi < 4; hi++) {
            for (int wi = 0; wi < 4; wi++) {
                float val = x_4x4[ic][hi][wi];
                if (val <= 0.0f) continue;

                for (int oc = 0; oc < 24; oc++) {
                    const int16_t* k_base = &layer4_deconv2_weight[(ic * 24 + oc) * 9];
                    for (int kh = 0; kh < 3; kh++) {
                        int ho = hi * 2 - 1 + kh;
                        if (ho < 0 || ho >= 8) continue;
                        for (int kw = 0; kw < 3; kw++) {
                            int wo = wi * 2 - 1 + kw;
                            if (wo < 0 || wo >= 8) continue;
                            x_8x8[oc][ho][wo] += val * (((float)k_base[kh * 3 + kw]) / 256.0f);
                        }
                    }
                }
            }
        }
    }

    for (int oc = 0; oc < 24; oc++) {
        for (int h = 0; h < 8; h++) {
            for (int w = 0; w < 8; w++) {
                if (x_8x8[oc][h][w] < 0.0f) x_8x8[oc][h][w] = 0.0f;
            }
        }
    }

    // ------------------------------------------------------------------------
    // 第 4 步: 反卷积 Deconv3 (24x8x8 -> 1x16x16, k=3, s=2, p=1, op=1) + Sigmoid
    // ------------------------------------------------------------------------
    float b3 = ((float)layer7_deconv3_bias[0]) / 256.0f;
    for (int h = 0; h < 16; h++) {
        for (int w = 0; w < 16; w++) {
            x_16x16[h][w] = b3;
        }
    }

    for (int ic = 0; ic < 24; ic++) {
        const int16_t* k_base = &layer6_deconv3_weight[ic * 9];
        for (int hi = 0; hi < 8; hi++) {
            for (int wi = 0; wi < 8; wi++) {
                float val = x_8x8[ic][hi][wi];
                if (val <= 0.0f) continue;

                for (int kh = 0; kh < 3; kh++) {
                    int ho = hi * 2 - 1 + kh;
                    if (ho < 0 || ho >= 16) continue;
                    for (int kw = 0; kw < 3; kw++) {
                        int wo = wi * 2 - 1 + kw;
                        if (wo < 0 || wo >= 16) continue;
                        x_16x16[ho][wo] += val * (((float)k_base[kh * 3 + kw]) / 256.0f);
                    }
                }
            }
        }
    }

    // Sigmoid 映射到亮度 [0.0, 1.0]
    for (int h = 0; h < 16; h++) {
        for (int w = 0; w < 16; w++) {
            output_img[h][w] = 1.0f / (1.0f + expf(-x_16x16[h][w]));
        }
    }
}

