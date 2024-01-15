import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from utils import find_largest_divisor

NBITS = 8


class LinearQuantEstimatorOp(torch.autograd.Function):
    """
    Linear Quantization Estimator Operator
    """

    @staticmethod
    def forward(ctx, input, signed, nbits, max_val):
        assert max_val > 0

        # Determine the quantization range
        if signed:
            qmin = -2 ** (nbits - 1)
            qmax = 2 ** (nbits - 1) - 1
        else:
            qmin = 0
            qmax = 2 ** nbits - 1

        # Scale input to the quantization range and clamp
        scale = max_val / qmax
        output = torch.round(input / scale).clamp(qmin, qmax)

        # Rescale to the original range
        return output * scale

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE)
        return grad_output.clone(), None, None, None


class ConvTranspose2dQuant(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,
                 bias=True, dilation=1, padding_mode='zeros'):
        super(ConvTranspose2dQuant, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                                   output_padding, groups, bias, dilation, padding_mode)
        self.nbits = NBITS
        self.input_signed = False
        self.input_quant = True
        self.input_max = 4.0

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError("Use padding mode 'zeros' for ConvTranspose2dQuant.")
        num_spatial_dims = 2
        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size,
                                              num_spatial_dims, self.dilation)
        if self.input_quant:
            max_val = self.input_max
            min_val = -max_val if self.input_signed else 0.0
            input = LinearQuantEstimatorOp.apply(
                input.clamp(min=min_val, max=max_val), self.input_signed, self.nbits, max_val
            )

        max_val = self.weight.abs().max().item()
        weight = LinearQuantEstimatorOp.apply(self.weight, True, self.nbits, max_val)
        return F.conv_transpose2d(
            input, weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)


class Generator(nn.Module):
    def __init__(self, nz: int, ngf: int, nc: int, quant: Optional[bool] = False, n_layers=4, dims=None):
        """
        :param nz: size of z latent vector (i.e. size of generator input)
        :param ngf: size of feature maps in generator
        :param nc: number of channels in the training images (color = 3)
        :param n_layers: number of layers in the network
        :param dims: list of channel dimensions for each layer
        """

        super(Generator, self).__init__()

        if quant:
            print("[!] Running quantised model.")
            convtranspose_class = ConvTranspose2dQuant
        else:
            convtranspose_class = nn.ConvTranspose2d

        if dims is None or len(dims) != n_layers:
            default_ngf_multipliers = [8, 4, 2, 1]  # default architecture multipliers
            dims = [ngf * m for m in default_ngf_multipliers[:n_layers]]

        # Initial layer
        layers = [
            convtranspose_class(nz, dims[0], 4, 1, 0, bias=False),
            nn.GroupNorm(find_largest_divisor(dims[0], 32), dims[0]),
            nn.ReLU(inplace=True),
        ]

        # Intermediate layers
        for i in range(1, n_layers):
            layers += [
                convtranspose_class(dims[i-1], dims[i], 4, 2, 1, bias=False),
                nn.GroupNorm(find_largest_divisor(dims[i], 32), dims[i]),
                nn.ReLU(inplace=True),
            ]

        # Final layer (match the output channel size)
        layers += [
            convtranspose_class(dims[-1], nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        ]

        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, ndf: int, nc: int):
        """
        :param ndf: size of feature maps in the discriminator
        :param nc: number of channels in the training images (color = 3)
        """

        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input = (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size = (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 2), ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size = (ndf * 2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 4), ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size = (ndf * 4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.GroupNorm(min(32, ndf * 8), ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size = (ndf * 8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1, 1).squeeze(1)


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
