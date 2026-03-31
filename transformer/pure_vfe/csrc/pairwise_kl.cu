/*
 * Fused pairwise KL divergence kernel for the Pure VFE Transformer.
 *
 * Key decomposition (from constant_gauge_kl_derivation.md):
 *   KL(q_i || Ω_ij · q_j) = ½[tr(P_j · Q_i) + (ρ_i - ν_j)ᵀ P_j (ρ_i - ν_j)
 *                              - K + ldc_q_i + ldc_k_j]
 *
 * where (precomputed per token):
 *   Q_i = Ω_i⁻¹ Σ_i Ω_i⁻ᵀ           (pulled-back query covariance)
 *   ρ_i = Ω_i⁻¹ μ_i                   (pulled-back query mean)
 *   P_j = Ω_jᵀ Σ_j⁻¹ Ω_j             (rotated key precision)
 *   ν_j = Ω_j⁻¹ μ_j                   (pulled-back key mean)
 *   ldc_q_i = 2 ln|det Ω_i| - ln det Σ_i  (query log-det)
 *   ldc_k_j = -2 ln|det Ω_j| + ln det Σ_j (key log-det)
 *
 * For K_h = 8: all matrices fit in registers (64 floats).
 * One thread block per (batch, head, query_i). Threads handle keys j.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Template parameter K for compile-time unrolling
template <int K>
__global__ void pairwise_kl_forward_kernel(
    const float* __restrict__ Q,        // [B, H, N, K, K]
    const float* __restrict__ rho,      // [B, H, N, K]
    const float* __restrict__ P,        // [B, H, N, K, K]
    const float* __restrict__ nu,       // [B, H, N, K]
    const float* __restrict__ ldc_q,    // [B, H, N]
    const float* __restrict__ ldc_k,    // [B, H, N]
    float* __restrict__ kl_out,         // [B, H, N, N]
    const int N,
    const bool causal
) {
    // Block: one (batch, head, query_i)
    const int bhi = blockIdx.x;
    const int BH = gridDim.x / N;
    const int bh = bhi / N;
    const int i = bhi % N;
    const int j = threadIdx.x;

    if (j >= N) return;
    if (causal && j > i) {
        // Future tokens get +inf KL (masked out in softmax)
        kl_out[bh * N * N + i * N + j] = 1e9f;
        return;
    }

    // Load Q_i into shared memory (shared across all j threads)
    __shared__ float s_Q[K * K];
    __shared__ float s_rho[K];
    __shared__ float s_ldc_q;

    const int q_mat_off = (bh * N + i) * K * K;
    const int q_vec_off = (bh * N + i) * K;

    // Cooperative loading
    for (int idx = threadIdx.x; idx < K * K; idx += blockDim.x)
        s_Q[idx] = Q[q_mat_off + idx];
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x)
        s_rho[idx] = rho[q_vec_off + idx];
    if (threadIdx.x == 0)
        s_ldc_q = ldc_q[bh * N + i];
    __syncthreads();

    // Load P_j, nu_j, ldc_k_j into registers
    const int p_mat_off = (bh * N + j) * K * K;
    const int p_vec_off = (bh * N + j) * K;

    float P_j[K * K];
    float nu_j[K];

    #pragma unroll
    for (int idx = 0; idx < K * K; idx++)
        P_j[idx] = P[p_mat_off + idx];
    #pragma unroll
    for (int idx = 0; idx < K; idx++)
        nu_j[idx] = nu[p_vec_off + idx];
    float ldc_k_j = ldc_k[bh * N + j];

    // 1. Compute tr(P_j @ Q_i) = Σ_{a,b} P_j[a,b] * Q_i[b,a]
    float trace_val = 0.0f;
    #pragma unroll
    for (int a = 0; a < K; a++) {
        #pragma unroll
        for (int b = 0; b < K; b++) {
            trace_val += P_j[a * K + b] * s_Q[b * K + a];
        }
    }

    // 2. Compute d = rho_i - nu_j
    float d[K];
    #pragma unroll
    for (int a = 0; a < K; a++)
        d[a] = s_rho[a] - nu_j[a];

    // 3. Compute P_j @ d (matrix-vector product)
    float Pd[K];
    #pragma unroll
    for (int a = 0; a < K; a++) {
        Pd[a] = 0.0f;
        #pragma unroll
        for (int b = 0; b < K; b++)
            Pd[a] += P_j[a * K + b] * d[b];
    }

    // 4. Compute Mahalanobis: d^T P_j d
    float mahal = 0.0f;
    #pragma unroll
    for (int a = 0; a < K; a++)
        mahal += d[a] * Pd[a];

    // 5. Full KL = 0.5 * (trace + mahal - K + ldc_q_i + ldc_k_j)
    float kl = 0.5f * (trace_val + mahal - (float)K + s_ldc_q + ldc_k_j);

    // Clamp to non-negative (numerical safety)
    kl = fmaxf(kl, 0.0f);

    kl_out[bh * N * N + i * N + j] = kl;
}


/*
 * Fused attention-weighted gradient aggregation kernel.
 *
 * Computes the alignment contribution to ∂F/∂μ_i:
 *   grad_i = Σ_j w_ij * Ω_i⁻ᵀ P_j (ρ_i - ν_j)
 *
 * where w_ij = β_ij [1 + (E_β[KL] - KL_ij)/τ]  (includes softmax correction)
 *
 * One block per (batch, head, query_i). Threads reduce over keys j.
 */
template <int K>
__global__ void grad_mu_alignment_kernel(
    const float* __restrict__ P,            // [B, H, N, K, K]
    const float* __restrict__ rho,          // [B, H, N, K]
    const float* __restrict__ nu,           // [B, H, N, K]
    const float* __restrict__ Omega_i_invT, // [B, H, N, K, K] — Ω_i⁻ᵀ
    const float* __restrict__ beta,         // [B, H, N, N]
    const float* __restrict__ kl_vals,      // [B, H, N, N]
    float* __restrict__ grad_out,           // [B, H, N, K]
    const int N,
    const float tau
) {
    const int bhi = blockIdx.x;
    const int bh = bhi / N;
    const int i = bhi % N;

    // Shared memory for reduction
    __shared__ float s_rho[K];
    __shared__ float s_OiT[K * K];   // Ω_i⁻ᵀ
    __shared__ float s_grad[K];       // accumulated gradient

    const int vec_off = (bh * N + i) * K;
    const int mat_off = (bh * N + i) * K * K;

    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
        s_rho[idx] = rho[vec_off + idx];
        s_grad[idx] = 0.0f;
    }
    for (int idx = threadIdx.x; idx < K * K; idx += blockDim.x)
        s_OiT[idx] = Omega_i_invT[mat_off + idx];
    __syncthreads();

    // Compute E_β[KL] = Σ_j β_ij * KL_ij
    const int beta_row = bh * N * N + i * N;
    float e_kl = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        e_kl += beta[beta_row + j] * kl_vals[beta_row + j];
    }
    // Warp-level reduction for e_kl
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        e_kl += __shfl_down_sync(0xffffffff, e_kl, offset);
    __shared__ float s_e_kl;
    if (threadIdx.x == 0) s_e_kl = 0.0f;
    __syncthreads();
    if (threadIdx.x % warpSize == 0)
        atomicAdd(&s_e_kl, e_kl);
    __syncthreads();
    e_kl = s_e_kl;

    // Each thread handles a subset of j values, accumulates locally
    float local_grad[K];
    #pragma unroll
    for (int a = 0; a < K; a++) local_grad[a] = 0.0f;

    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float b_ij = beta[beta_row + j];
        if (b_ij < 1e-10f) continue;

        float kl_ij = kl_vals[beta_row + j];
        // Weight with softmax correction (Eq. 24), clamped to match Python path
        float correction = (e_kl - kl_ij) / tau;
        correction = fminf(fmaxf(correction, -1.0f), 2.0f);
        float w = b_ij * (1.0f + correction);

        // Load P_j, nu_j
        const int pj_off = (bh * N + j) * K * K;
        const int nj_off = (bh * N + j) * K;

        // Compute d = rho_i - nu_j, then P_j @ d
        float Pd[K];
        #pragma unroll
        for (int a = 0; a < K; a++) {
            float d_b;
            Pd[a] = 0.0f;
            #pragma unroll
            for (int b = 0; b < K; b++) {
                d_b = s_rho[b] - nu[nj_off + b];
                Pd[a] += P[pj_off + a * K + b] * d_b;
            }
        }

        // Accumulate w * Pd (in pulled-back frame)
        #pragma unroll
        for (int a = 0; a < K; a++)
            local_grad[a] += w * Pd[a];
    }

    // Reduce across threads using shared memory
    __shared__ float s_reduce[K * 64]; // max 64 threads

    #pragma unroll
    for (int a = 0; a < K; a++)
        s_reduce[a * blockDim.x + threadIdx.x] = local_grad[a];
    __syncthreads();

    // Thread 0 accumulates and applies Ω_i⁻ᵀ
    if (threadIdx.x == 0) {
        float pulled_back[K];
        #pragma unroll
        for (int a = 0; a < K; a++) {
            pulled_back[a] = 0.0f;
            for (int t = 0; t < blockDim.x && t < N; t++)
                pulled_back[a] += s_reduce[a * blockDim.x + t];
        }

        // Push forward: grad_i = Ω_i⁻ᵀ @ pulled_back
        const int out_off = (bh * N + i) * K;
        #pragma unroll
        for (int a = 0; a < K; a++) {
            float val = 0.0f;
            #pragma unroll
            for (int b = 0; b < K; b++)
                val += s_OiT[a * K + b] * pulled_back[b];
            grad_out[out_off + a] = val;
        }
    }
}


/*
 * Fused attention-weighted covariance gradient kernel.
 *
 * ∂KL_ij/∂Σ_i = ½(transported_precision_ij - Σ_i⁻¹)
 *              = ½(Ω_i⁻ᵀ P_j Ω_i⁻¹ - Σ_i⁻¹)
 *
 * Alignment gradient: ∂F_align/∂Σ_i = ½[Σ_j β_ij Ω_i⁻ᵀ P_j Ω_i⁻¹ - Σ_i⁻¹]
 *   (ignoring softmax correction for covariance — it's second-order small)
 */
template <int K>
__global__ void grad_sigma_alignment_kernel(
    const float* __restrict__ P,            // [B, H, N, K, K]
    const float* __restrict__ Omega_i_inv,  // [B, H, N, K, K]
    const float* __restrict__ beta,         // [B, H, N, N]
    float* __restrict__ weighted_prec_out,  // [B, H, N, K, K] — Σ_j β_ij (Ω⁻ᵀ P_j Ω⁻¹)
    const int N
) {
    const int bhi = blockIdx.x;
    const int bh = bhi / N;
    const int i = bhi % N;

    // Load Ω_i⁻¹ into shared memory
    __shared__ float s_Oi_inv[K * K];
    const int mat_off = (bh * N + i) * K * K;
    for (int idx = threadIdx.x; idx < K * K; idx += blockDim.x)
        s_Oi_inv[idx] = Omega_i_inv[mat_off + idx];
    __syncthreads();

    // Each thread accumulates for a subset of (a, b) entries
    const int beta_row = bh * N * N + i * N;

    // Accumulate: Σ_j β_ij P_j into weighted_P (in pulled-back frame)
    // Then transform: Ω_i⁻ᵀ weighted_P Ω_i⁻¹
    __shared__ float s_wP[K * K]; // weighted precision in original frame
    for (int idx = threadIdx.x; idx < K * K; idx += blockDim.x)
        s_wP[idx] = 0.0f;
    __syncthreads();

    // Sum β_ij * P_j
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float b_ij = beta[beta_row + j];
        if (b_ij < 1e-10f) continue;

        const int pj_off = (bh * N + j) * K * K;
        #pragma unroll
        for (int ab = 0; ab < K * K; ab++)
            atomicAdd(&s_wP[ab], b_ij * P[pj_off + ab]);
    }
    __syncthreads();

    // Now compute Ω_i⁻ᵀ @ s_wP @ Ω_i⁻¹
    // First: temp = s_wP @ Ω_i⁻¹ (right multiply)
    __shared__ float s_temp[K * K];
    for (int idx = threadIdx.x; idx < K * K; idx += blockDim.x) {
        int a = idx / K, b = idx % K;
        float val = 0.0f;
        #pragma unroll
        for (int c = 0; c < K; c++)
            val += s_wP[a * K + c] * s_Oi_inv[c * K + b];
        s_temp[idx] = val;
    }
    __syncthreads();

    // Then: result = Ω_i⁻ᵀ @ temp  (Ω_i⁻ᵀ[a,c] = Ω_i⁻¹[c,a])
    const int out_off = (bh * N + i) * K * K;
    for (int idx = threadIdx.x; idx < K * K; idx += blockDim.x) {
        int a = idx / K, b = idx % K;
        float val = 0.0f;
        #pragma unroll
        for (int c = 0; c < K; c++)
            val += s_Oi_inv[c * K + a] * s_temp[c * K + b];  // Ω⁻ᵀ[a,c] = Ω⁻¹[c,a]
        weighted_prec_out[out_off + idx] = val;
    }
}


/*
 * Precomputation kernel: compute per-token quantities.
 *
 * For each token i:
 *   Ω_i⁻¹            (via batched inverse, done in PyTorch)
 *   ρ_i = Ω_i⁻¹ μ_i
 *   ν_i = Ω_i⁻¹ μ_i  (same — i plays both query and key roles)
 *   Q_i = Ω_i⁻¹ Σ_i Ω_i⁻ᵀ
 *   P_i = Ω_iᵀ Σ_i⁻¹ Ω_i
 *   ldc_q_i = 2 ln|det Ω_i| - ln det Σ_i
 *   ldc_k_i = -2 ln|det Ω_i| + ln det Σ_i
 *
 * One block per (batch, head, token).
 */
template <int K>
__global__ void precompute_kernel(
    const float* __restrict__ mu,           // [B, H, N, K]
    const float* __restrict__ Sigma,        // [B, H, N, K, K]
    const float* __restrict__ Sigma_inv,    // [B, H, N, K, K]
    const float* __restrict__ Omega_inv,    // [B, H, N, K, K]
    const float* __restrict__ ln_det_Omega, // [B, H, N]
    const float* __restrict__ ln_det_Sigma, // [B, H, N]
    float* __restrict__ Q_out,              // [B, H, N, K, K]
    float* __restrict__ rho_out,            // [B, H, N, K]
    float* __restrict__ P_out,              // [B, H, N, K, K]
    float* __restrict__ nu_out,             // [B, H, N, K]
    float* __restrict__ ldc_q_out,          // [B, H, N]
    float* __restrict__ ldc_k_out,          // [B, H, N]
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;  // not used; idx is the token index
    // Reinterpret: each block handles some tokens
    // Actually, let's do one thread per (b, h, n) and iterate over matrix elements

    const int bhn = blockIdx.x;  // linear index into [B*H*N]
    if (bhn >= (gridDim.x)) return;  // guard

    const int vec_off = bhn * K;
    const int mat_off = bhn * K * K;

    // Load Ω_i⁻¹ into registers
    float Oi[K * K];
    #pragma unroll
    for (int ab = 0; ab < K * K; ab++)
        Oi[ab] = Omega_inv[mat_off + ab];

    // ρ = Ω⁻¹ @ μ
    float mu_local[K];
    #pragma unroll
    for (int a = 0; a < K; a++)
        mu_local[a] = mu[vec_off + a];

    float rho_local[K];
    #pragma unroll
    for (int a = 0; a < K; a++) {
        rho_local[a] = 0.0f;
        #pragma unroll
        for (int b = 0; b < K; b++)
            rho_local[a] += Oi[a * K + b] * mu_local[b];
    }

    #pragma unroll
    for (int a = 0; a < K; a++) {
        rho_out[vec_off + a] = rho_local[a];
        nu_out[vec_off + a] = rho_local[a];  // Same: both query and key
    }

    // Q = Ω⁻¹ Σ Ω⁻ᵀ
    float Sig[K * K];
    #pragma unroll
    for (int ab = 0; ab < K * K; ab++)
        Sig[ab] = Sigma[mat_off + ab];

    // temp = Σ @ Ω⁻ᵀ  (Ω⁻ᵀ[a,b] = Ω⁻¹[b,a])
    float temp[K * K];
    #pragma unroll
    for (int a = 0; a < K; a++) {
        #pragma unroll
        for (int b = 0; b < K; b++) {
            float val = 0.0f;
            #pragma unroll
            for (int c = 0; c < K; c++)
                val += Sig[a * K + c] * Oi[b * K + c];  // Ω⁻ᵀ[c,b] = Ω⁻¹[b,c]
            temp[a * K + b] = val;
        }
    }

    // Q = Ω⁻¹ @ temp
    #pragma unroll
    for (int a = 0; a < K; a++) {
        #pragma unroll
        for (int b = 0; b < K; b++) {
            float val = 0.0f;
            #pragma unroll
            for (int c = 0; c < K; c++)
                val += Oi[a * K + c] * temp[c * K + b];
            Q_out[mat_off + a * K + b] = val;
        }
    }

    // P = Ωᵀ Σ⁻¹ Ω  — but we need Ω, not Ω⁻¹
    // Alternative: P = (Ω⁻¹)⁻ᵀ Σ⁻¹ (Ω⁻¹)⁻¹ ... that's circular.
    // Better: we pass Sigma_inv and compute P in Python. Or pass Omega too.
    // For now, compute P = (Ω⁻ᵀ)⁻¹ Σ⁻¹ (Ω⁻¹)⁻¹ ... we need Ω itself.
    //
    // Actually let's just compute P in Python as Ωᵀ @ Σ⁻¹ @ Ω
    // and only compute Q, rho, nu in this kernel.

    // Log-det contributions
    float ld_omega = ln_det_Omega[bhn];
    float ld_sigma = ln_det_Sigma[bhn];
    ldc_q_out[bhn] = 2.0f * ld_omega - ld_sigma;
    ldc_k_out[bhn] = -2.0f * ld_omega + ld_sigma;
}


// C++ dispatch functions

torch::Tensor pairwise_kl_cuda(
    torch::Tensor Q, torch::Tensor rho,
    torch::Tensor P, torch::Tensor nu,
    torch::Tensor ldc_q, torch::Tensor ldc_k,
    bool causal
) {
    const int BH = Q.size(0) * Q.size(1);
    const int N = Q.size(2);
    const int K = Q.size(3);

    // Reshape to [BH, N, ...] for kernel
    auto Q_flat = Q.reshape({BH, N, K, K}).contiguous();
    auto rho_flat = rho.reshape({BH, N, K}).contiguous();
    auto P_flat = P.reshape({BH, N, K, K}).contiguous();
    auto nu_flat = nu.reshape({BH, N, K}).contiguous();
    auto ldc_q_flat = ldc_q.reshape({BH, N}).contiguous();
    auto ldc_k_flat = ldc_k.reshape({BH, N}).contiguous();

    auto kl_out = torch::empty({BH, N, N}, Q.options());

    const int threads = (N <= 64) ? 64 : ((N <= 128) ? 128 : 256);
    const int blocks = BH * N;

    // Dispatch based on K
    switch (K) {
        case 4:
            pairwise_kl_forward_kernel<4><<<blocks, threads>>>(
                Q_flat.data_ptr<float>(), rho_flat.data_ptr<float>(),
                P_flat.data_ptr<float>(), nu_flat.data_ptr<float>(),
                ldc_q_flat.data_ptr<float>(), ldc_k_flat.data_ptr<float>(),
                kl_out.data_ptr<float>(), N, causal);
            break;
        case 8:
            pairwise_kl_forward_kernel<8><<<blocks, threads>>>(
                Q_flat.data_ptr<float>(), rho_flat.data_ptr<float>(),
                P_flat.data_ptr<float>(), nu_flat.data_ptr<float>(),
                ldc_q_flat.data_ptr<float>(), ldc_k_flat.data_ptr<float>(),
                kl_out.data_ptr<float>(), N, causal);
            break;
        case 16:
            pairwise_kl_forward_kernel<16><<<blocks, threads>>>(
                Q_flat.data_ptr<float>(), rho_flat.data_ptr<float>(),
                P_flat.data_ptr<float>(), nu_flat.data_ptr<float>(),
                ldc_q_flat.data_ptr<float>(), ldc_k_flat.data_ptr<float>(),
                kl_out.data_ptr<float>(), N, causal);
            break;
        default:
            TORCH_CHECK(false, "pairwise_kl_cuda: unsupported K=", K, ". Supported: 4, 8, 16.");
    }

    // Reshape back to [B, H, N, N]
    return kl_out.reshape({Q.size(0), Q.size(1), N, N});
}


torch::Tensor grad_mu_alignment_cuda(
    torch::Tensor P, torch::Tensor rho, torch::Tensor nu,
    torch::Tensor Omega_i_invT, torch::Tensor beta,
    torch::Tensor kl_vals, float tau
) {
    const int BH = P.size(0) * P.size(1);
    const int N = P.size(2);
    const int K = P.size(3);

    auto P_flat = P.reshape({BH, N, K, K}).contiguous();
    auto rho_flat = rho.reshape({BH, N, K}).contiguous();
    auto nu_flat = nu.reshape({BH, N, K}).contiguous();
    auto OiT_flat = Omega_i_invT.reshape({BH, N, K, K}).contiguous();
    auto beta_flat = beta.reshape({BH, N, N}).contiguous();
    auto kl_flat = kl_vals.reshape({BH, N, N}).contiguous();

    auto grad_out = torch::zeros({BH, N, K}, P.options());

    const int threads = min(N, 64);
    const int blocks = BH * N;

    switch (K) {
        case 8:
            grad_mu_alignment_kernel<8><<<blocks, threads>>>(
                P_flat.data_ptr<float>(), rho_flat.data_ptr<float>(),
                nu_flat.data_ptr<float>(), OiT_flat.data_ptr<float>(),
                beta_flat.data_ptr<float>(), kl_flat.data_ptr<float>(),
                grad_out.data_ptr<float>(), N, tau);
            break;
        default:
            TORCH_CHECK(false, "grad_mu_alignment_cuda: unsupported K=", K);
    }

    return grad_out.reshape({P.size(0), P.size(1), N, K});
}


torch::Tensor grad_sigma_alignment_cuda(
    torch::Tensor P, torch::Tensor Omega_inv,
    torch::Tensor beta
) {
    const int BH = P.size(0) * P.size(1);
    const int N = P.size(2);
    const int K = P.size(3);

    auto P_flat = P.reshape({BH, N, K, K}).contiguous();
    auto Oi_flat = Omega_inv.reshape({BH, N, K, K}).contiguous();
    auto beta_flat = beta.reshape({BH, N, N}).contiguous();

    auto out = torch::zeros({BH, N, K, K}, P.options());

    const int threads = min(N, 64);
    const int blocks = BH * N;

    switch (K) {
        case 8:
            grad_sigma_alignment_kernel<8><<<blocks, threads>>>(
                P_flat.data_ptr<float>(), Oi_flat.data_ptr<float>(),
                beta_flat.data_ptr<float>(), out.data_ptr<float>(), N);
            break;
        default:
            TORCH_CHECK(false, "grad_sigma_alignment_cuda: unsupported K=", K);
    }

    return out.reshape({P.size(0), P.size(1), N, K, K});
}
