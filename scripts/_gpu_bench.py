"""One-off GPU per-batch benchmark (does NOT touch running training).

Measures the pure GPU time of one training step (AMP forward + backward +
optimizer step) for the UNet3D base16/depth4 at batch=2, D=16, 640x640 —
the exact config of the running trainings.  Data already lives on GPU, so
this isolates GPU compute time from any data-loading cost.
"""
import time
import torch

torch.cuda.set_device(0)
from cbct_n2n.models import UNet3D

model = UNet3D(base_channels=16, depth=4).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

x = torch.randn(2, 1, 16, 640, 640, device="cuda")
target = torch.randn(2, 1, 16, 640, 640, device="cuda")


def step():
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast():
        y = model(x)
        loss = torch.nn.functional.l1_loss(y, target)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()


for _ in range(5):
    step()
torch.cuda.synchronize()

N = 20
t0 = time.time()
for _ in range(N):
    step()
torch.cuda.synchronize()
dt = (time.time() - t0) / N

print(f"GPU per-batch (AMP fwd+bwd+step) = {dt*1000:.1f} ms")
print(f"  -> 2 samples/batch, so per-sample GPU = {dt/2*1000:.1f} ms")
