import torch
import linear_pytorch
import linear_cpp
import linear_cuda
import copy
import time

in_features = 512
out_features = 256
batch_size = 16
dtype = torch.float32
device = "cuda"
N = 10000

linear_map = {
    "linear": torch.nn.Linear(in_features, out_features, dtype=dtype).to(device),
    "linear_pytorch": linear_pytorch.MyLinear(in_features, out_features, dtype=dtype).to(device),
    "linear_cpp": linear_cpp.MyLinear(in_features, out_features, dtype=dtype).to(device),
    "linear_cuda": linear_cuda.MyLinear(in_features, out_features, dtype=dtype).to(device)
}

linear = linear_map["linear"]
for name, linear2 in linear_map.items():
    if name == "linear":
        continue
    linear2.weight = copy.deepcopy(linear.weight)
    linear2.bias = copy.deepcopy(linear.bias)

input = torch.randn(batch_size, in_features, dtype=dtype).to(device)
label = torch.randn(batch_size, out_features, dtype=dtype).to(device)

for name, linear in linear_map.items():
    forward = 0
    backward = 0
    for _ in range(N):
        start = time.time()
        output = linear(input)
        torch.cuda.synchronize()
        forward += time.time() - start

        output = output.sum()
        
        start = time.time()
        output.backward()
        torch.cuda.synchronize()
        backward += time.time() - start
    print("{:15s} Forward: {:.3f} us | Backward {:.3f} us".format(name, forward * 1e6 / N, backward * 1e6 / N))
    