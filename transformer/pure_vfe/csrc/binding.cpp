/*
 * PyTorch C++ binding for Pure VFE CUDA kernels.
 */
#include <torch/extension.h>

// Forward declarations from pairwise_kl.cu
torch::Tensor pairwise_kl_cuda(
    torch::Tensor Q, torch::Tensor rho,
    torch::Tensor P, torch::Tensor nu,
    torch::Tensor ldc_q, torch::Tensor ldc_k,
    bool causal);

torch::Tensor grad_mu_alignment_cuda(
    torch::Tensor P, torch::Tensor rho, torch::Tensor nu,
    torch::Tensor Omega_i_invT, torch::Tensor beta,
    torch::Tensor kl_vals, float tau);

torch::Tensor grad_sigma_alignment_cuda(
    torch::Tensor P, torch::Tensor Omega_inv,
    torch::Tensor beta);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pairwise_kl", &pairwise_kl_cuda,
          "Fused pairwise KL divergence (CUDA)",
          py::arg("Q"), py::arg("rho"),
          py::arg("P"), py::arg("nu"),
          py::arg("ldc_q"), py::arg("ldc_k"),
          py::arg("causal") = true);

    m.def("grad_mu_alignment", &grad_mu_alignment_cuda,
          "Attention-weighted mean gradient with softmax correction (CUDA)",
          py::arg("P"), py::arg("rho"), py::arg("nu"),
          py::arg("Omega_i_invT"), py::arg("beta"),
          py::arg("kl_vals"), py::arg("tau"));

    m.def("grad_sigma_alignment", &grad_sigma_alignment_cuda,
          "Attention-weighted covariance gradient (CUDA)",
          py::arg("P"), py::arg("Omega_inv"), py::arg("beta"));
}
