#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Section 1: Unparallelized C Implementation
// This follows Algorithm 1 logic for CPU-based verification
void flash_attn_cpu(float* Q, float* K, float* V, float* O, int N, int d, int Br, int Bc) {
    float scale = 1.0f / sqrtf((float)d);
    int Tr = N / Br;
    int Tc = N / Bc;

    for (int i = 0; i < Tr; i++) {
        float* Qi = &Q[i * Br * d];
        float* Oi = &O[i * Br * d];
        float mi[1024], li[1024]; 

        for (int r = 0; r < Br; r++) {
            mi[r] = -FLT_MAX;
            li[r] = 0.0f;
            for (int x = 0; x < d; x++) Oi[r * d + x] = 0.0f;
        }

        for (int j = 0; j < Tc; j++) {
            float* Kj = &K[j * Bc * d];
            float* Vj = &V[j * Bc * d];
            
            for (int r = 0; r < Br; r++) {
                float row_s[1024]; 
                float mij = -FLT_MAX;
                // Line 8: Compute Sij = Qi * Kj^T
                for (int c = 0; c < Bc; c++) {
                    float sum = 0;
                    for (int x = 0; x < d; x++) sum += Qi[r * d + x] * Kj[c * d + x];
                    row_s[c] = sum * scale;
                    if (row_s[c] > mij) mij = row_s[c];
                }

                // Line 9: Update Softmax statistics
                float mi_new = fmaxf(mi[r], mij);
                float alpha = expf(mi[r] - mi_new);
                float lij = 0.0f;
                for (int c = 0; c < Bc; c++) {
                    float p = expf(row_s[c] - mi_new);
                    row_s[c] = p; // P_ij_hat
                    lij += p;
                }

                // Line 10: Update Output Chunk
                for (int x = 0; x < d; x++) {
                    float pv = 0;
                    for (int c = 0; c < Bc; c++) pv += row_s[c] * Vj[c * d + x];
                    Oi[r * d + x] = alpha * Oi[r * d + x] + pv;
                }
                li[r] = alpha * li[r] + lij;
                mi[r] = mi_new;
            }
        }
        // Line 12: Final normalization
        for (int r = 0; r < Br; r++) {
            for (int x = 0; x < d; x++) Oi[r * d + x] /= li[r];
        }
    }
}

// Section 2: Parallelized CUDA Implementation
// Implements Algorithm 1 with IO-awareness and Shared Memory Tiling
__global__ void flash_attn_v2_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int N, int d, int Br, int Bc, float scale
) {
    int i = blockIdx.x; 
    int tx = threadIdx.x; 

    // Dynamic Shared Memory Allocation (SharedMem struct equivalent)
    extern __shared__ float sram[];
    float* qi = sram;                          // Size: Br * d
    float* kj = &qi[Br * d];                   // Size: Bc * d
    float* vj = &kj[Bc * d];                   // Size: Bc * d
    float* Sij = &vj[Bc * d];                  // Size: Br * Bc

    // Local registers for online softmax
    float mi = -FLT_MAX;
    float li = 0.0f;
    
    // Accumulator for output row (Assumes max d=128 for register safety)
    float local_oi[128]; 
    for(int n=0; n<d; n++) local_oi[n] = 0.0f;

    // Load Qi tile to Shared Memory (stays for all j)
    for (int n = 0; n < d; n++) {
        qi[tx * d + n] = Q[(i * Br + tx) * d + n];
    }
    __syncthreads();

    int Tc = N / Bc;
    for (int j = 0; j < Tc; j++) {
        // Collaborative load of Kj and Vj tiles
        for (int n = 0; n < (Bc * d + Br - 1) / Br; n++) {
            int idx = n * Br + tx;
            if (idx < Bc * d) {
                kj[idx] = K[(j * Bc) * d + idx];
                vj[idx] = V[(j * Bc) * d + idx];
            }
        }
        __syncthreads();

        // Compute Sij tile and find row max mij
        float mij = -FLT_MAX;
        for (int c = 0; c < Bc; c++) {
            float sum = 0.0f;
            for (int n = 0; n < d; n++) {
                sum += qi[tx * d + n] * kj[c * d + n];
            }
            sum *= scale;
            Sij[tx * Bc + c] = sum;
            if (sum > mij) mij = sum;
        }

        // Online Softmax rescaling factors
        float mi_new = fmaxf(mi, mij);
        float alpha = expf(mi - mi_new);
        
        float lij = 0.0f;
        for (int c = 0; c < Bc; c++) {
            float p = expf(Sij[tx * Bc + c] - mi_new);
            Sij[tx * Bc + c] = p; // P_ij_hat
            lij += p;
        }

        // Update local output row: Oi = alpha * Oi + Pij * Vj
        for (int n = 0; n < d; n++) {
            float pv = 0.0f;
            for (int c = 0; c < Bc; c++) {
                pv += Sij[tx * Bc + c] * vj[c * d + n];
            }
            local_oi[n] = local_oi[n] * alpha + pv;
        }

        // Update running statistics
        li = li * alpha + lij;
        mi = mi_new;
        __syncthreads();
    }

    // Final normalization and write-back to Global Memory
    for (int n = 0; n < d; n++) {
        O[(i * Br + tx) * d + n] = local_oi[n] / li;
    }
}

// PyTorch binding
torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int Br, int Bc) {
    int N = Q.size(0);
    int d = Q.size(1);
    auto O = torch::zeros_like(Q);
    float scale = 1.0f / sqrtf((float)d);

    // Dynamic Shared Memory size based on assignment formula
    size_t sram_size = (Br * d + 2 * Bc * d + Br * Bc) * sizeof(float);
    
    dim3 grid(N / Br);
    dim3 block(Br);

    flash_attn_v2_kernel<<<grid, block, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), N, d, Br, Bc, scale
    );
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "FlashAttention V2 Forward Implementation");
}
