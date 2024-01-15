import os

import numpy as np
import torch
from torch import nn

from models import Generator, ConvTranspose2dQuant
from utils import load_checkpoint, measure_model, save_checkpoint

nz = 100
nc = 3
student = Generator(nz=nz, ngf=64, nc=nc, quant=True)
ckpt = load_checkpoint(
    path="./out/checkpoints/student_20e16b/ckpt_19e.pth",
    generator=student,
)
student_ops = measure_model(student, (64, nz, 1, 1))
print(f'Before Compression: {student_ops / 1024. / 1024.:.6f}M ops')

# Get non-zero channel numbers
non_zero_list = []
for m in student.modules():
    if isinstance(m, torch.nn.GroupNorm) and m.weight is not None:
        gamma = m.weight.data.cpu().numpy()
        channel_num = np.sum(gamma != 0)
        non_zero_list.append(channel_num)
print(f"Non-zero channels: {non_zero_list}")

# Construct compressed Generator
compressed = Generator(nz=nz, ngf=64, nc=nc, dims=non_zero_list, quant=True)
print(compressed)

weight_list = {}
pre_m = None
pre_name = ""
input_nc = range(nz)

isConvTransposed = lambda m: isinstance(m, nn.ConvTranspose2d) or isinstance(m, ConvTranspose2dQuant)

for layer, (name, m) in enumerate(student.named_modules()):
    # print(m)
    if isinstance(m, nn.GroupNorm) and m.weight is not None and pre_m is not None:
        gamma = m.weight.data
        channel_indicator = torch.nonzero(gamma).flatten().cpu().numpy().tolist()

        if isConvTransposed(pre_m):
            temp = pre_m.weight.data[:, channel_indicator, :, :]
            if pre_m.bias is not None:
                bias = pre_m.bias.data[channel_indicator]
            else:
                bias = None
            weight_list[pre_name] = (temp[input_nc, :, :, :], bias)

        if m.affine:
            bias = m.bias.data[channel_indicator]
        else:
            bias = None

        weight_list[name] = (m.weight.data[channel_indicator], bias)
        pre_m = m
        pre_name = name
        input_nc = list(channel_indicator)
    if isConvTransposed(m):  # convolutional layers
        pre_m = m
        pre_name = name
    elif isinstance(m, nn.GroupNorm) and m.weight is None:  # uninitialised GroupNorm layers
        weight_list[pre_name] = (pre_m.weight.data[:, input_nc, :, :], pre_m.bias.data)
        input_nc = [i for i in range(pre_m.weight.data.shape[0])]
        pre_m = m
        pre_name = name
    elif isinstance(m, nn.Tanh) and isConvTransposed(pre_m):  # last layer
        if pre_m.bias is not None:
            bias = pre_m.bias
        else:
            bias = None
        weight_list[pre_name] = (pre_m.weight.data[channel_indicator, :, :, :], bias)
        pre_m = m
        pre_name = name

print("Starting pruned weights copying procedure (all shapes must match!)")
for name, m in compressed.named_modules():
    if name in weight_list:
        print(f"[{name}] Copying incoming {weight_list[name][0].shape} into {m.weight.data.shape}")
        m.weight.data.copy_(weight_list[name][0])
        if weight_list[name][1] is not None:
            m.bias.data.copy_(weight_list[name][1])

print("Compressed weights loaded into sub-architecture.")

compressed_ops = measure_model(compressed, (64, nz, 1, 1))
print(
    f"After Compression: {compressed_ops / 1024. / 1024.:.6f}M ops ({(compressed_ops / student_ops) * 100 - 100:.2f}%) "
    f"r_s={student_ops / compressed_ops:.2f} r_c="
)

torch.save({
    "weights": compressed.state_dict(),
    "channels": non_zero_list,
}, os.path.join(".", "out", "compressed16b.pth"))
