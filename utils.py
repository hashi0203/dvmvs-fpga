import numpy as np
import nngen as ng

def rshift_round_and_clip(act, rshift, dtype):
    return ng.clip(ng.rshift_round(act, rshift), asymmetric_clip=True, dtype=dtype)


def clip(input, dtype):
    info = np.iinfo(dtype)
    return np.clip(input.astype(np.int64), info.min, info.max).astype(dtype)


class interpolate():
    def __init__(self, out_height, out_width, rshift, mode):
        self.out_height, self.out_width = out_height, out_width
        self.rshift = rshift
        self.mode = mode

    def __call__(self, input):
        batchsize, in_height, in_width, channels = input.shape
        out_height, out_width = self.out_height, self.out_width
        rshift = self.rshift
        mode = self.mode
        output = np.zeros((batchsize, out_height, out_width, channels), dtype=input.dtype)

        if rshift != 0:
            print("interpolate:", rshift)
        else:
            import torch
            out_shape = (out_height, out_width)
            mid = torch.tensor(input.astype(np.float32).transpose(0, 3, 1, 2))
            if mode == "nearest":
                mid = torch.nn.functional.interpolate(mid, size=out_shape, mode="nearest")
            elif mode == "bilinear":
                mid = torch.nn.functional.interpolate(mid, size=out_shape, mode='bilinear', align_corners=True)
            output = mid.detach().numpy().copy().transpose(0, 2, 3, 1).astype(input.dtype)

        # if mode == "nearest":
        #     fy = in_height / out_height
        #     fx = in_width / out_width
        #     for j in range(out_height):
        #         for k in range(out_width):
        #             y = int(j * fy)
        #             x = int(k * fx)
        #             for i in range(channels):
        #                 # input_idx = (i * in_height + y) * in_width + x
        #                 # output_idx = (i * out_height + j) * out_width + k
        #                 output[0][j][k][i] = input[0][y][x][i]
        # elif mode == "bilinear":
        #     if in_height < out_height:
        #         fy = (in_height - 1) / (out_height - 1)
        #         fx = (in_width - 1) / (out_width - 1)
        #         for j in range(out_height):
        #             for k in range(out_width):
        #                 y = j * fy
        #                 x = k * fx
        #                 y_int = int(y) if y < in_height-1 else int(y) - 1
        #                 x_int = int(x) if x < in_width-1 else int(x) - 1
        #                 ys = [y_int, y_int + 1]
        #                 xs = [x_int, x_int + 1]
        #                 dys = [y - ys[0], ys[1] - y]
        #                 dxs = [x - xs[0], xs[1] - x]
        #                 for i in range(channels):
        #                     sum = 0
        #                     for yi in range(2):
        #                         for xi in range(2):
        #                             sum += dys[1-yi] * dxs[1-xi] * input[0][ys[yi]][xs[xi]][i]
        #                     output[0][j][k][i] = int(round(sum)) >> rshift
        #     else:
        #         print("in_height is larger than out_height")
        # else:
        #     print("The 'mode' option in interpolation should be 'nearest' or 'bilinear,' but it is", mode)

        return output
