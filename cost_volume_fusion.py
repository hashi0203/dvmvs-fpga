import numpy as np
import nngen as ng
from utils import clip

n_depth_levels = 64

# def grid_sample(image, warping):
#     _, height, width, channels = image.shape
#     warped_image = np.zeros(image.shape)

#     for j in range(height):
#         for k in range(width):
#             x = (warping[j][k][0] + 1) * (width - 1) / 2.0
#             y = (warping[j][k][1] + 1) * (height - 1) / 2.0
#             y_int = int(y)
#             x_int = int(x)
#             ys = [y_int, y_int + 1]
#             xs = [x_int, x_int + 1]
#             dys = [y - ys[0], ys[1] - y]
#             dxs = [x - xs[0], xs[1] - x]
#             for i in range(channels):
#                 for yi in range(2):
#                     for xi in range(2):
#                         if not(ys[yi] < 0 or height-1 < ys[yi] or xs[xi] < 0 or width-1 < xs[xi]):
#                             warped_image[0][j][k][i] += dys[1-yi] * dxs[1-xi] * image[0][ys[yi]][xs[xi]][i]

#     return warped_image


def calculate_cost_volume_by_warping(image1, image2, warping, out_shape):
    cost_volume = np.zeros(out_shape)

    for depth_i in range(n_depth_levels):
        # warped_image2 = grid_sample(image2, warping[depth_i][0])
        import torch
        warped_image2 = torch.nn.functional.grid_sample(input=torch.tensor(image2.astype(np.float32).transpose(0, 3, 1, 2)),
                                                        grid=torch.tensor(warping[depth_i]),
                                                        mode='bilinear',
                                                        padding_mode='zeros',
                                                        align_corners=True)
        warped_image2 = warped_image2.detach().numpy().copy().transpose(0, 2, 3, 1)
        cost_volume[:,:,:,depth_i] = np.sum(image1 * warped_image2, axis=3) / image1.shape[-1]

    return cost_volume


class fusion():
    def __init__(self, rshift, warpings, n_measurement_frames):
        self.rshift = rshift
        self.warpings = warpings
        self.n_measurement_frames = n_measurement_frames

    def __call__(self, image1, *image2s):
        rshift = self.rshift
        warpings = self.warpings
        n_measurement_frames = self.n_measurement_frames

        batchsize, height, width, _ = image1.shape
        out_shape = (batchsize, height, width, n_depth_levels)
        fused_cost_volume = np.zeros(out_shape)
        for m in range(n_measurement_frames):
            fused_cost_volume += calculate_cost_volume_by_warping(image1, image2s[m], warpings[m], out_shape)

        return clip(np.round((fused_cost_volume / n_measurement_frames) / (1 << rshift)), image1.dtype)


def cost_volume_fusion(act78, act79s, warpings, n_measurement_frames=2):
    # [80] conv
    act80 = ng.extern([act78, *act79s], opcode=0x80, func=fusion(11, warpings, n_measurement_frames))
    act80.shape = (1, 32, 48, 64)

    return act80
