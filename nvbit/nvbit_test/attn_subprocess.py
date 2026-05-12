import torch
import os

for i in range(1):
    device = os.environ.get('CUDA_DEVICE', 'cuda:0')
    device = torch.device(device)

    Q = torch.load('../attn_subprocess_Q.pt').to(device)
    K = torch.load('../attn_subprocess_K.pt').to(device)
    attn_scores = torch.matmul(Q, K.transpose(-2, -1))

    torch.save(attn_scores.cpu(), '../attn_subprocess_result.pt')

    base = torch.load('../attn_subprocess_result_base.pt').to(device)

    print(f"[Run {i}] Shape:", base.shape)

    # --- Step 1: view as int16 to inspect bits ---
    base_bits = base.view(torch.int16)
    attn_bits = attn_scores.view(torch.int16)

    # --- Step 2: compute bit difference ---
    bit_flips = base_bits ^ attn_bits  # XOR shows which bits flipped

    # --- Step 3: identify all changed positions ---
    diff_mask = bit_flips != 0
    diff_indices = torch.nonzero(diff_mask)

    print(f"Total bit flips: {diff_indices.shape[0]}")

    # --- Step 4: print diff info per element ---
    for idx in diff_indices:
        b, h, l, d = idx
        base_val = base[b, h, l, d].item()
        fault_val = attn_scores[b, h, l, d].item()
        flip = bit_flips[b, h, l, d].item()

        print(f"[B={b.item()} H={h.item()} L={l.item()} D={d.item()}] "
              f"Base={base_val:.6f} → Fault={fault_val:.6f}, "
              f"Bit flips: {flip:016b}")
