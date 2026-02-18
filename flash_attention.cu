__global__ void flash_attention_kernel(float* Q, float* K, float* V, float* O, int N, int d, int Br, int Bc, float sqrt_d) {
    // 每个 Threadblock 负责输出矩阵 O 的一个分块 Oi
    int i = blockIdx.x; 
    int tid = threadIdx.x; // 假设 tid 遍历 Br 行

    // 声明 Shared Memory (SRAM)
    extern __shared__ float sram[];
    float* qi = sram; // Br * d
    float* kj = &sram[Br * d]; // Bc * d
    float* vj = &sram[Br * d + Bc * d]; // Bc * d

    // 1. 将 Qi 加载到 SRAM
    for (int x = 0; x < d; x++) qi[tid * d + x] = Q[i * Br * d + tid * d + x];
    
    float mi = -CUDART_INF_F;
    float li = 0.0f;
    // 初始化局部输出寄存器或共享内存

    int Tc = N / Bc;
    for (int j = 0; j < Tc; j++) {
        // 2. 将 Kj, Vj 加载到 SRAM
        // 此处需要 __syncthreads() 确保加载完成
        __syncthreads(); 
        
        // 3. 计算并更新 Online Softmax (逻辑同 CPU 版)
        // ... 此处省略重复的数学累加逻辑 ...
        
        __syncthreads();
    }

    // 4. 写回 HBM
    for (int x = 0; x < d; x++) O[i * Br * d + tid * d + x] = Oi_local[x] / li;
}
