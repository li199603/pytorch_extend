#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void add_bias(torch::Tensor &output, const torch::Tensor &bias);

torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(bias);

    torch::Tensor output = input.mm(weight.transpose(0, 1));
    add_bias(output, bias);
    return output;
}

std::vector<torch::Tensor> linear_backward(torch::Tensor grad_output, torch::Tensor input, torch::Tensor weight)
{   
    torch::Tensor grad_input = grad_output.mm(weight);
    torch::Tensor grad_weight = grad_output.transpose(0, 1).mm(input);
    torch::Tensor grad_bias = grad_output.sum(0);
    return {grad_input, grad_weight, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &linear_forward, "Linear forward (CUDA)");
    m.def("backward", &linear_backward, "Linear backward (CUDA)");
}