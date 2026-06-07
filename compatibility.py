import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))   # must be (12, 0)

import bitsandbytes as bnb
print("bitsandbytes:", bnb.__version__)
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", (x @ x).shape)                      # exercises a real kernel