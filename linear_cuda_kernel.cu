#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void add_bias_kernel(torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
                                const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> bias)
{
    const int b = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bias.size(0))
    {
        output[b][i] += bias[i];
    }
}

void add_bias(torch::Tensor &output, const torch::Tensor &bias)
{
    auto batch_size = output.size(0);
    auto feat_dim = output.size(1);

    dim3 block_size(1024);
    dim3 grid_size((feat_dim - 1) / 1024 + 1, batch_size);
    AT_DISPATCH_FLOATING_TYPES(output.type(), "add_bias_kernel", ([&] {
                                   add_bias_kernel<scalar_t><<<grid_size, block_size>>>(
                                       output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                                       bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>());
                               }));
}