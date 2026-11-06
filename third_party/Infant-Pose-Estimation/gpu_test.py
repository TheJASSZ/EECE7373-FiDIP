# GPU sanity check (EECE 7373, STUDENT_GUIDE Part 1): versions, device, capability, a small GPU matmul.
#   python gpu_test.py
import platform
import sys
import time

import torch

print('python', platform.python_version(), '| torch', torch.__version__, '| CUDA (build)', torch.version.cuda,
      '| cuDNN', torch.backends.cudnn.version())
if not torch.cuda.is_available():
    print('CUDA not available')
    sys.exit(1)
print('arch list', torch.cuda.get_arch_list())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print('device %d: %s | capability %d.%d | %.1f GB' % (i, p.name, p.major, p.minor, p.total_memory / 2**30))
a = torch.randn(4096, 4096, device='cuda')
b = torch.randn(4096, 4096, device='cuda')
c = a @ b  # warm-up (cuBLAS init)
torch.cuda.synchronize()
t = time.time()
c = a @ b
torch.cuda.synchronize()
dt = time.time() - t
err = (c[:64, :64].cpu() - a[:64].cpu() @ b[:, :64].cpu()).abs().max().item()
print('4096x4096 fp32 matmul: %.1f ms, %.1f TFLOP/s, max |GPU-CPU| on a 64x64 block = %.2e' % (
    dt * 1e3, 2 * 4096 ** 3 / dt / 1e12, err))
print('GPU OK')
