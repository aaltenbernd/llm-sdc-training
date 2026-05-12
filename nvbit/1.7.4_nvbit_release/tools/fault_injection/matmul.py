import torch
import time

start = time.time()

torch.manual_seed(42)

dim = 512

D = 0

for i in range(0, 100):
    A = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)
    B = torch.randn(dim, dim, device='cuda', dtype=torch.bfloat16)

    C = torch.matmul(A, B)

    D += C.sum()

print(D)
print(time.time() - start)
