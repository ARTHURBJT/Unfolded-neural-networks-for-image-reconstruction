"""
The function `torch_conv_layer_to_affine` takes a `torch.nn.Conv2d` layer `conv`
and produces an equivalent `torch.nn.Linear` layer `fc`.
Specifically, this means that the following holds for `x` of a valid shape:
    torch.flatten(conv(x)) == fc(torch.flatten(x))
Or equivalently:
    conv(x) == fc(torch.flatten(x)).reshape(conv(x).shape)
allowing of course for some floating-point error.
"""
from typing import Tuple

import torch
import torch.nn as nn
import numpy as np


def torch_conv_layer_to_affine(
    conv: torch.nn.Conv2d, input_size: Tuple[int, int]
) -> torch.nn.Linear:
    w, h = input_size

    # Formula from the Torch docs:
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    output_size = [
        (input_size[i] + 2 * conv.padding[i] - conv.kernel_size[i]) // conv.stride[i]
        + 1
        for i in [0, 1]
    ]

    in_shape = (conv.in_channels, w, h)
    out_shape = (conv.out_channels, output_size[0], output_size[1])

    fc = nn.Linear(in_features=np.product(in_shape), out_features=np.product(out_shape))
    fc.weight.data.fill_(0.0)

    # Output coordinates
    with torch.no_grad():
        for xo, yo in range2d(output_size[0], output_size[1]):
            # The upper-left corner of the filter in the input tensor
            xi0 = -conv.padding[0] + conv.stride[0] * xo
            yi0 = -conv.padding[1] + conv.stride[1] * yo

            # Position within the filter
            for xd, yd in range2d(conv.kernel_size[0], conv.kernel_size[1]):
                # Output channel
                for co in range(conv.out_channels):
                    if conv.bias != None:
                        fc.bias[enc_tuple((co, xo, yo), out_shape)] = conv.bias[co]
                    for ci in range(conv.in_channels):
                        # Make sure we are within the input image (and not in the padding)
                        if 0 <= xi0 + xd < w and 0 <= yi0 + yd < h:
                            cw = conv.weight[co, ci, xd, yd]
                            # Flatten the weight position to 1d in "canonical ordering",
                            # i.e. guaranteeing that:
                            # FC(img.reshape(-1)) == Conv(img).reshape(-1)
                            fc.weight[
                                enc_tuple((co, xo, yo), out_shape),
                                enc_tuple((ci, xi0 + xd, yi0 + yd), in_shape),
                            ] = cw

    return fc


def range2d(to_a, to_b):
    for a in range(to_a):
        for b in range(to_b):
            yield a, b


def enc_tuple(tup: Tuple, shape: Tuple) -> int:
    res = 0
    coef = 1
    for i in reversed(range(len(shape))):
        assert tup[i] < shape[i]
        res += coef * tup[i]
        coef *= shape[i]

    return res


def dec_tuple(x: int, shape: Tuple) -> Tuple:
    res = []
    for i in reversed(range(len(shape))):
        res.append(x % shape[i])
        x //= shape[i]

    return tuple(reversed(res))


def test_tuple_encoding():
    x = enc_tuple((3, 2, 1), (5, 6, 7))
    assert dec_tuple(x, (5, 6, 7)) == (3, 2, 1)
    print("Tuple encoding ok")


def test_layer_conversion():
    for stride in [1, 2]:
        for padding in [0, 1, 2]:
            for filter_size in [3, 4]:
                img = torch.rand((1, 2, 6, 7))
                conv = nn.Conv2d(2, 5, filter_size, stride=stride, padding=padding)
                fc = torch_conv_layer_to_affine(conv, img.shape[2:])

                # Also checks that our encoding flattens the inputs/outputs such that
                # FC(flatten(img)) == flatten(Conv(img))
                res1 = fc(img.reshape((-1))).reshape(conv(img).shape)
                res2 = conv(img)
                worst_error = (res1 - res2).max()

                print("Output shape", res2.shape, "Worst error: ", float(worst_error))
                assert worst_error <= 1.0e-6

    print("Layer conversion ok")


if __name__ == "__main__":
    test_tuple_encoding()
    test_layer_conversion()