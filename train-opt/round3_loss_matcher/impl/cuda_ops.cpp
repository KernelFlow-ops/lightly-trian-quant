#include <torch/extension.h>

// CUDA implementations are defined in cuda_ops.cu.  Keeping this small C++
// translation unit makes torch.utils.cpp_extension's ABI checks predictable.
torch::Tensor point_loss_forward_cuda(torch::Tensor pred, torch::Tensor target,
                                      torch::Tensor coords,
                                      torch::Tensor bce, torch::Tensor dice,
                                      torch::Tensor sampled_pred,
                                      torch::Tensor sampled_target,
                                      torch::Tensor numerator,
                                      torch::Tensor denominator);
torch::Tensor point_loss_backward_cuda(torch::Tensor coords,
                                       torch::Tensor sampled_pred,
                                       torch::Tensor sampled_target,
                                       torch::Tensor numerator,
                                       torch::Tensor denominator,
                                       torch::Tensor grad_bce,
                                       torch::Tensor grad_dice,
                                       torch::Tensor grad_pred);
torch::Tensor matcher_exact_cuda(torch::Tensor cost, torch::Tensor valid);

torch::Tensor point_loss_forward(torch::Tensor pred, torch::Tensor target,
                                 torch::Tensor coords, torch::Tensor bce,
                                 torch::Tensor dice,
                                 torch::Tensor sampled_pred,
                                 torch::Tensor sampled_target,
                                 torch::Tensor numerator,
                                 torch::Tensor denominator) {
  TORCH_CHECK(pred.is_cuda(), "pred must be CUDA");
  return point_loss_forward_cuda(pred, target, coords, bce, dice,
                                 sampled_pred, sampled_target, numerator,
                                 denominator);
}

torch::Tensor point_loss_backward(torch::Tensor coords, torch::Tensor sampled_pred,
                                  torch::Tensor sampled_target,
                                  torch::Tensor numerator,
                                  torch::Tensor denominator,
                                  torch::Tensor grad_bce,
                                  torch::Tensor grad_dice,
                                  torch::Tensor grad_pred) {
  TORCH_CHECK(grad_pred.is_cuda(), "grad_pred must be CUDA");
  return point_loss_backward_cuda(coords, sampled_pred, sampled_target,
                                  numerator, denominator, grad_bce, grad_dice,
                                  grad_pred);
}

torch::Tensor matcher_exact(torch::Tensor cost, torch::Tensor valid) {
  TORCH_CHECK(cost.is_cuda(), "cost must be CUDA");
  return matcher_exact_cuda(cost, valid);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("point_loss_forward", &point_loss_forward, "CUDA point loss forward");
  m.def("point_loss_backward", &point_loss_backward, "CUDA point loss backward");
  m.def("matcher_exact", &matcher_exact, "CUDA exact T<=4 matcher");
}
