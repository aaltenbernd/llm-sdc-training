import torch
import numpy as np

torch.manual_seed(42)

shape = (256, 8, 256, 256)
A_fp32 = (torch.rand(shape, dtype=torch.float32, device='cuda') * 2) - 1
B_fp32 = (torch.rand(shape, dtype=torch.float32, device='cuda') * 2) - 1

A = A_fp32.to(dtype=torch.bfloat16)
B = B_fp32.to(dtype=torch.bfloat16)

C = torch.matmul(A, B)

np.set_printoptions(precision=5, suppress=True, linewidth=1000, threshold=np.inf)

C_cpu = C.cpu().to(torch.float32)
#print(np.array(C_cpu)[0, 0])
print(C_cpu.sum().item())

'''shape = (16, 16)

A_fp32 = torch.rand(shape, dtype=torch.float32, device='cuda')
B_fp32 = torch.rand(shape, dtype=torch.float32, device='cuda')

A = A_fp32.to(dtype=torch.bfloat16)
B = B_fp32.to(dtype=torch.bfloat16)'''

'''A = torch.arange(1, 16 * 16 + 1, dtype=torch.float32, device='cuda') \
        .reshape(16, 16) \
        .to(torch.bfloat16)
print(A)

B = torch.arange(1, 16 * 16 + 1, dtype=torch.float32, device='cuda') \
        .reshape(16, 16) \
        .to(torch.bfloat16)'''

'''shape = (256, 8, 256, 256)
#shape = (1, 1, 256, 256)
A = torch.ones(shape, dtype=torch.bfloat16, device='cuda')
B = torch.ones(shape, dtype=torch.bfloat16, device='cuda')

# Define tile size
tile_m, tile_n = 16, 16

# Select tile indices, for example the tile at batch=0, channel=0, tile row=0, tile col=0
batch_idx, channel_idx = 0, 0
tile_row_idx, tile_col_idx = 0, 0

# Compute start and end indices for the tile
row_start = tile_row_idx * tile_m
row_end = row_start + tile_m
col_start = tile_col_idx * tile_n
col_end = col_start + tile_n

# Fill the selected tile with increasing numbers from 1 to tile_m*tile_n
increasing_tile = torch.arange(1, tile_m * tile_n + 1, device='cuda', dtype=torch.float32).reshape(tile_m, tile_n)

# Cast to bfloat16 and assign
#A[batch_idx, channel_idx, row_start:row_end, col_start:col_end] = increasing_tile.to(torch.bfloat16)
#B[batch_idx, channel_idx, row_start:row_end, col_start:col_end] = increasing_tile.to(torch.bfloat16)

# The rest remains zero

#print(A[batch_idx, channel_idx, row_start:row_end, col_start:col_end])

#C = torch.ones(shape, dtype=torch.bfloat16, device='cuda')

#for b in range(256):
#    for h in range(8):
#        C[b, h] = A[b, h] @ B[b, h]

C = torch.matmul(A, B)

C_cpu = C.cpu().to(torch.float32)

np.set_printoptions(precision=5, suppress=True, linewidth=1000, threshold=np.inf)

print(np.array(C_cpu)[0, 0])

#for b in range(256):
#    for h in range(8):
#        print(C[b, h].sum())

##for b in range(256):
#    for h in range(8):
#        print(C[b, h].mean().item())

print(C_cpu.sum().item())

'''