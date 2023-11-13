#include <torch/extension.h>
#include <vector>

torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{
    torch::Tensor output = input.mm(weight.transpose(0, 1)) + bias.unsqueeze(0);
    return output;
}

std::vector<torch::Tensor> linear_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight)
{
    torch::Tensor grad_input = grad_output.mm(weight);
    torch::Tensor grad_weight = grad_output.transpose(0, 1).mm(input);
    torch::Tensor grad_bias = grad_output.sum(0);
    return {grad_input, grad_weight, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &linear_forward, "Linear forward");
  m.def("backward", &linear_backward, "Linear backward");
}