import torch
from torch import Tensor
from torch.autograd import Function
from typing import Tuple
import copy
from torch.autograd import gradcheck
import linear_cuda_impl

class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        output = linear_cuda_impl.forward(input, weight, bias)
        ctx.save_for_backward(input, weight)
        return output  # [B, N]
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # grad_output -- [B, N]
        input, weight = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = linear_cuda_impl.backward(grad_output, input, weight)
        return grad_input, grad_weight, grad_bias

class MyLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), dtype=dtype))
        self.bias = torch.nn.Parameter(torch.empty((out_features,), dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.weight)
        torch.nn.init.uniform_(self.bias)
    
    def forward(self, input: Tensor) -> Tensor:
        return LinearFunction.apply(input, self.weight, self.bias)
        
        
if __name__ == "__main__":
    in_features = 512
    out_features = 256
    batch_size = 16
    dtype = torch.float64
    device = "cuda"
    
    linear = torch.nn.Linear(in_features, out_features, dtype=dtype).to(device)
    my_linear = MyLinear(in_features, out_features, dtype=dtype).to(device)
    my_linear.weight = copy.deepcopy(linear.weight)
    my_linear.bias = copy.deepcopy(linear.bias)
    
    input = torch.randn(batch_size, in_features, dtype=dtype).to(device)
    label = torch.randn(batch_size, out_features, dtype=dtype).to(device)
    
    output1 = linear(input)
    loss1 = torch.nn.functional.mse_loss(output1, label)
    loss1.backward()
    
    output2 = my_linear(input)
    loss2 = torch.nn.functional.mse_loss(output2, label)
    loss2.backward()
    
    assert torch.all(torch.isclose(output1, output2))
    assert torch.all(torch.isclose(linear.weight.grad, my_linear.weight.grad))
    assert torch.all(torch.isclose(linear.bias.grad, my_linear.bias.grad))
    
    input_ = torch.randn(batch_size, in_features, requires_grad=True, dtype=dtype).to(device)
    assert gradcheck(my_linear, input_)

    print("success")
    
    