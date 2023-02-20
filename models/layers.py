__author__ = 'marvinler'

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.modules.utils import _single, _pair, _triple


# from torch._jit_internal import weak_module, weak_script_method


# @weak_module
class PolarConvNd(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, dimensions=2, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        self.init_kernel_size = kernel_size
        assert kernel_size % 2 == 1, 'expected kernel size to be odd, found %d' % kernel_size
        self.init_dimensions = dimensions

        self.base_vectors = torch.from_numpy(self.build_base_vectors()).float()
        self.true_base_vectors_shape = self.base_vectors.shape
        self.base_vectors = self.base_vectors.view(self.true_base_vectors_shape[0],
                                                   np.prod(self.true_base_vectors_shape[1:]).astype(int))

        if torch.cuda.is_available():
            self.base_vectors = self.base_vectors.cuda()

        inferred_kernel_size = self.true_base_vectors_shape[0]
        _kernel_size = _single(inferred_kernel_size)
        _stride = _single(stride)
        _padding = _single(padding)
        _dilation = _single(dilation)
        super(PolarConvNd, self).__init__(
            in_channels, out_channels, _kernel_size, _stride, _padding, _dilation,
            False, _single(0), groups, bias, padding_mode)

        if dimensions == 2:
            self.reconstructed_stride = _pair(stride)
            self.reconstructed_padding = _pair(padding)
            self.reconstructed_dilation = _pair(dilation)
            self.reconstructed_conv_op = F.conv2d
        elif dimensions == 3:
            self.reconstructed_stride = _triple(stride)
            self.reconstructed_padding = _triple(padding)
            self.reconstructed_dilation = _triple(dilation)
            self.reconstructed_conv_op = F.conv3d
        else:
            raise ValueError('dimension %d not supported' % dimensions)

    def build_base_vectors(self):
        kernel_size = self.init_kernel_size
        middle = kernel_size // 2
        dimensions = self.init_dimensions

        base_vectors = []
        # Burning phase: determine the number of base vectors
        unique_distances = []
        if dimensions == 2:
            for i in range(kernel_size):
                for j in range(kernel_size):
                    i_ = abs(i - middle)
                    j_ = abs(j - middle)
                    unique_distances.append(int(i_ * i_ + j_ * j_))
        elif dimensions == 3:
            for i in range(kernel_size):
                for j in range(kernel_size):
                    for k in range(kernel_size):
                        i_ = abs(i - middle)
                        j_ = abs(j - middle)
                        k_ = abs(k - middle)
                        unique_distances.append(int(i_ * i_ + j_ * j_ + k_ * k_))
        unique_distances, distances_counts = np.unique(unique_distances, return_counts=True)
        unique_distances = np.sort(unique_distances)

        for unique_distance, n in zip(unique_distances, distances_counts):  # number of base vectors
            base_vector = np.zeros([kernel_size] * dimensions)
            if dimensions == 2:
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        i_ = abs(i - middle)
                        j_ = abs(j - middle)
                        if int(i_ * i_ + j_ * j_) == unique_distance:
                            base_vector[i, j] = 1. / n
            elif dimensions == 3:
                for i in range(kernel_size):
                    for j in range(kernel_size):
                        for k in range(kernel_size):
                            i_ = abs(i - middle)
                            j_ = abs(j - middle)
                            k_ = abs(k - middle)
                            if int(i_ * i_ + j_ * j_ + k_ * k_) == unique_distance:
                                base_vector[i, j, k] = 1. / n
            base_vectors.append(base_vector)
        base_vectors = np.asarray(base_vectors)
        return base_vectors

    # @weak_script_method
    def forward(self, input):
        weight_size = self.weight.shape
        weight = torch.mm(self.weight.view(np.prod(weight_size[:-1]), weight_size[-1]), self.base_vectors).view(*weight_size[:-1], *self.true_base_vectors_shape[1:])
        return self.reconstructed_conv_op(input, weight, self.bias, self.reconstructed_stride, self.reconstructed_padding, self.reconstructed_dilation, self.groups)

    def __repr__(self):
        return ('PolarConv%dd' % self.init_dimensions) + '(' + self.extra_repr() + ')'
