import torch
from torch.utils.cpp_extension import load
import os


cuda_code = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void flash_attn_v2_fwd_kernel(
    const float* Q, const float* K, const float* V,
    float* O, float* L,
    int N, int d, int Br, int Bc, float softmax_scale
) {
    int i = blockIdx.x; 
    int tx = threadIdx.x; 

    extern __shared__ float sram[];
    float* qi = sram;                             
    float* kj = &qi[Br * d];                      
    float* vj = &kj[Bc * d];                      
    float* Sij = &vj[Bc * d];                     

    float mi = -FLT_MAX;
    float li = 0.0f;
    float oi[128]; // 支持到 d=128
    for(int k=0; k<d; k++) oi[k] = 0.0f;

    for(int k=tx; k<d; k+=Br) {
        qi[tx * d + k] = Q[(i * Br + tx) * d + k];
    }
    __syncthreads();

    int Tc = (N + Bc - 1) / Bc;
    for (int j = 0; j < Tc; j++) {
        for(int k=tx; k<d; k+=Br) {
            kj[tx * d + k] = K[(j * Bc + tx) * d + k];
            vj[tx * d + k] = V[(j * Bc + tx) * d + k];
        }
        __syncthreads();

        float mij = -FLT_MAX;
        for (int col = 0; col < Bc; col++) {
            float sum = 0;
            for (int k = 0; k < d; k++) {
                sum += qi[tx * d + k] * kj[col * d + k];
            }
            sum *= softmax_scale;
            Sij[tx * Bc + col] = sum;
            if (sum > mij) mij = sum;
        }

        float mi_new = fmaxf(mi, mij);
        float p_scale = expf(mi - mi_new);
        float p_curr_scale = expf(mij - mi_new);

        float lij = 0.0f;
        for (int col = 0; col < Bc; col++) {
            float p = expf(Sij[tx * Bc + col] - mi_new);
            Sij[tx * Bc + col] = p; 
            lij += p;
        }

        for (int k = 0; k < d; k++) {
            float pv = 0;
            for (int col = 0; col < Bc; col++) {
                pv += Sij[tx * Bc + col] * vj[col * d + k];
            }
            oi[k] = oi[k] * p_scale + pv * p_curr_scale; 
        }
        li = li * p_scale + lij;
        mi = mi_new;
        __syncthreads();
    }

    for(int k=0; k<d; k++) {
        O[(i * Br + tx) * d + k] = oi[k] / li;
    }
    L[i * Br + tx] = mi + logf(li);
}

torch::Tensor flash_attn_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V, int Br, int Bc) {
    int N = Q.size(0);
    int d = Q.size(1);
    auto O = torch::zeros_like(Q);
    auto L = torch::zeros({N}, Q.options());
    float scale = 1.0 / sqrt(d);

    int sram_size = (Br * d + 2 * Bc * d + Br * Bc) * sizeof(float);
    
    const int threads = Br;
    const int blocks = (N + Br - 1) / Br;

    flash_attn_v2_fwd_kernel<<<blocks, threads, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), L.data_ptr<float>(),
        N, d, Br, Bc, scale
    );
    return O;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn_fwd, "FlashAttention forward");
}
"""

with open("flash_attn.cu", "w") as f:
    f.write(cuda_code)

print(" CUDA Kernel ( 30-60 seconds)...")
flash_attn_lib = load(
    name="flash_attn_lib",
    sources=["flash_attn.cu"],
    verbose=True
)


N, d = 512, 64 
Br, Bc = 32, 32
Q, K, V = [torch.randn(N, d, device="cuda") for _ in range(3)]


output_custom = flash_attn_lib.forward(Q, K, V, Br, Bc)

output_ref = torch.nn.functional.scaled_dot_product_attention(
    Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0)
).squeeze(0)


diff = torch.abs(output_custom - output_ref).max().item()
print(f"\\n: {diff:.6e}")
if diff < 1e-4:
    print("✅ Result Match！")
else:
    print("❌ Discrepancy detected; suggest checking the scaling logic")
