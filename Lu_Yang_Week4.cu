#include <iostream>
#include <vector>
#include <cmath>
#include <float.h>
#include <cuda_runtime.h>

// --- 1. Unparallelized C Implementation ---
void flash_attn_cpu(float* Q, float* K, float* V, float* O, int N, int d, int Br, int Bc) {
    float scale = 1.0f / sqrtf(d);
    for (int i = 0; i < N / Br; i++) {
        float* Qi = &Q[i * Br * d];
        float* Oi = &O[i * Br * d];
        std::vector<float> mi(Br, -FLT_MAX), li(Br, 0.0f);

        for (int j = 0; j < N / Bc; j++) {
            float* Kj = &K[j * Bc * d];
            float* Vj = &V[j * Bc * d];

            for (int r = 0; r < Br; r++) {
                float mij = -FLT_MAX;
                std::vector<float> Sij(Bc);
                for (int c = 0; c < Bc; c++) {
                    float sum = 0;
                    for (int k = 0; k < d; k++) sum += Qi[r * d + k] * Kj[c * d + k];
                    Sij[c] = sum * scale;
                    if (Sij[c] > mij) mij = Sij[c];
                }

                float mi_new = fmaxf(mi[r], mij);
                float exp_old = expf(mi[r] - mi_new);
                float exp_curr = expf(mij - mi_new);
                float lij = 0;
                for (int c = 0; c < Bc; c++) lij += expf(Sij[c] - mij);
                float li_new = exp_old * li[r] + exp_curr * lij;

                for (int k = 0; k < d; k++) {
                    float pv = 0;
                    for (int c = 0; c < Bc; c++) pv += expf(Sij[c] - mij) * Vj[c * d + k];
                    Oi[r * d + k] = (li[r] * exp_old * Oi[r * d + k] + exp_curr * pv) / li_new;
                }
                mi[r] = mi_new; li[r] = li_new;
            }
        }
    }
}

// --- 2. Parallelized CUDA Implementation ---
__global__ void flash_attn_2_kernel(float* Q, float* K, float* V, float* O, int N, int d, int Br, int Bc, float scale) {
    int i = blockIdx.x; 
    int tid = threadIdx.x; 
    extern __shared__ float sram[];
    float* qi = sram;               
    float* kj = &qi[Br * d];        
    float* vj = &kj[Bc * d];        

    float mi = -1e20f, li = 0.0f;
    for (int k = 0; k < d; k++) qi[tid * d + k] = Q[(i * Br + tid) * d + k];
    __syncthreads();

    for (int j = 0; j < N / Bc; j++) {
        for (int k = 0; k < d; k++) {
            kj[tid * d + k] = K[(j * Bc + tid) * d + k];
            vj[tid * d + k] = V[(j * Bc + tid) * d + k];
        }
        __syncthreads();

        float mij = -1e20f;
        for (int c = 0; c < Bc; c++) {
            float sum = 0;
            for (int k = 0; k < d; k++) sum += qi[tid * d + k] * kj[c * d + k];
            if (sum * scale > mij) mij = sum * scale;
        }

        float mi_new = fmaxf(mi, mij);
        float exp_old = expf(mi - mi_new), exp_curr = expf(mij - mi_new);
        float lij = 0;
        for (int c = 0; c < Bc; c++) {
            float sum = 0;
            for (int k = 0; k < d; k++) sum += qi[tid * d + k] * kj[c * d + k];
            lij += expf(sum * scale - mij);
        }
        float li_new = exp_old * li + exp_curr * lij;

        for (int k = 0; k < d; k++) {
            float pv = 0;
            for (int c = 0; c < Bc; c++) {
                float sum = 0;
                for (int sk = 0; sk < d; sk++) sum += qi[tid * d + sk] * kj[c * d + sk];
                pv += expf(sum * scale - mij) * vj[c * d + k];
            }
            int out_idx = (i * Br + tid) * d + k;
            O[out_idx] = (li * exp_old * O[out_idx] + exp_curr * pv) / li_new;
        }
        mi = mi_new; li = li_new;
        __syncthreads();
    }
}

int main() {
    int N = 256, d = 64, Br = 32, Bc = 32;
    float scale = 1.0f / sqrtf(d);
    size_t size = N * d * sizeof(float);
    
    std::vector<float> h_Q(N*d), h_K(N*d), h_V(N*d), h_Oc(N*d, 0), h_Og(N*d, 0);
    for(int i=0; i<N*d; i++) {
        h_Q[i] = (float)rand()/RAND_MAX;
        h_K[i] = (float)rand()/RAND_MAX;
        h_V[i] = (float)rand()/RAND_MAX;
    }

    float *d_Q, *d_K, *d_V, *d_O;
    cudaMalloc(&d_Q, size); cudaMalloc(&d_K, size); cudaMalloc(&d_V, size); cudaMalloc(&d_O, size);
    cudaMemcpy(d_Q, h_Q.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), size, cudaMemcpyHostToDevice);
    cudaMemset(d_O, 0, size);

    flash_attn_cpu(h_Q.data(), h_K.data(), h_V.data(), h_Oc.data(), N, d, Br, Bc);
    flash_attn_2_kernel<<<N/Br, Br, (Br*d + 2*Bc*d)*4>>>(d_Q, d_K, d_V, d_O, N, d, Br, Bc, scale);
    cudaMemcpy(h_Og.data(), d_O, size, cudaMemcpyDeviceToHost);

    float err = 0;
    for(int i=0; i<N*d; i++) err = fmaxf(err, fabs(h_Oc[i] - h_Og[i]));
    std::cout << "Verification Max Error: " << err << std::endl;

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    return 0;
}
