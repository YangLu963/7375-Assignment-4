#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// 实现作业提供的 diag_inv_scale 逻辑
void diag_inv_scale(float* L, float* O, int Br, int d) {
    for (int i = 0; i < Br; ++i) {
        float inv_li = 1.0f / L[i]; // L 存储的是累加和
        for (int j = 0; j < d; ++j) {
            O[i * d + j] *= inv_li; // 对应算法第 12 行
        }
    }
}

// FlashAttention-2 Algorithm 1 (Unparallelized C)
void flash_attention_cpu(float* Q, float* K, float* V, float* O, int N, int d, int Br, int Bc) {
    float sqrt_d = sqrtf((float)d);
    int Tr = N / Br;
    int Tc = N / Bc;

    for (int i = 0; i < Tr; i++) {
        // 加载 Qi 并初始化局部统计量
        float* Qi = &Q[i * Br * d];
        float* Oi = &O[i * Br * d];
        float mi[Br]; // 存储当前行最大值
        float li[Br]; // 存储当前行累加和

        for (int k = 0; k < Br; k++) {
            mi[k] = -FLT_MAX;
            li[k] = 0.0f;
            for (int x = 0; x < d; x++) Oi[k * d + x] = 0.0f;
        }

        for (int j = 0; j < Tc; j++) {
            float* Kj = &K[j * Bc * d];
            float* Vj = &V[j * Bc * d];

            // 1. 计算 Sij = Qi * Kj^T / sqrt_d
            for (int r = 0; r < Br; r++) {
                float row_max_s = -FLT_MAX;
                float S_row[Bc];
                for (int c = 0; c < Bc; c++) {
                    float val = 0;
                    for (int x = 0; x < d; x++) val += Qi[r * d + x] * Kj[c * d + x];
                    S_row[c] = val / sqrt_d;
                    if (S_row[c] > row_max_s) row_max_s = S_row[c];
                }

                // 2. 更新在线 Softmax 统计量
                float mi_new = fmaxf(mi[r], row_max_s);
                float alpha = expf(mi[r] - mi_new);
                
                float li_new = 0;
                for (int c = 0; c < Bc; c++) {
                    float P_ij_hat = expf(S_row[c] - mi_new);
                    li_new += P_ij_hat;
                    // 3. 更新输出 Oi
                    for (int x = 0; x < d; x++) {
                        // 这一步结合了旧值的重缩放和新值的加入
                        if (j == 0) Oi[r * d + x] += P_ij_hat * Vj[c * d + x];
                        else Oi[r * d + x] = alpha * Oi[r * d + x] + P_ij_hat * Vj[c * d + x];
                    }
                }
                li[r] = alpha * li[r] + li_new;
                mi[r] = mi_new;
            }
        }
        // 4. 最终归一化
        diag_inv_scale(li, Oi, Br, d);
    }
}
