import numpy as np
import nngen as ng
import torch
from utils import round_and_clip

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


class fusion():
    n_depth_levels = 64

    def __init__(self, rshift, K, pose1s, pose2ss):
        self.rshift = rshift
        self.K = torch.tensor(K)
        self.pose1s = torch.tensor(pose1s)
        self.pose2ss = torch.tensor(pose2ss)

        test_image_width = 96
        test_image_height = 64
        self.warp_grid = self.get_warp_grid_for_cost_volume_calculation(int(test_image_width / 2), int(test_image_height / 2))


    def get_warp_grid_for_cost_volume_calculation(self, width, height):
        x = np.linspace(0, width - 1, num=int(width))
        y = np.linspace(0, height - 1, num=int(height))
        ones = np.ones(shape=(height, width))
        x_grid, y_grid = np.meshgrid(x, y)
        warp_grid = np.stack((x_grid, y_grid, ones), axis=-1)
        warp_grid = torch.from_numpy(warp_grid).float()
        warp_grid = warp_grid.view(-1, 3).t()
        return warp_grid


    def calc_warpings(self, frame_number, batchsize, height, width, n_measurement_frames):
        n_depth_levels = self.n_depth_levels
        K = self.K
        pose1 = self.pose1s[frame_number]
        pose2s = self.pose2ss[frame_number]
        warp_grid = self.warp_grid

        min_depth = 0.25
        max_depth = 20.0

        warp_grid = torch.cat(batchsize * [warp_grid.unsqueeze(dim=0)])

        warpings = [[] for _ in range(n_measurement_frames)]
        for m in range(n_measurement_frames):
            pose2 = pose2s[m]

            extrinsic2 = torch.inverse(pose2).bmm(pose1)
            R = extrinsic2[:, 0:3, 0:3]
            t = extrinsic2[:, 0:3, 3].unsqueeze(-1)

            Kt = K.bmm(t)
            K_R_Kinv = K.bmm(R).bmm(torch.inverse(K))
            K_R_Kinv_UV = K_R_Kinv.bmm(warp_grid)

            inverse_depth_base = 1.0 / max_depth
            inverse_depth_step = (1.0 / min_depth - 1.0 / max_depth) / (n_depth_levels - 1)

            width_normalizer = width / 2.0
            height_normalizer = height / 2.0

            for depth_i in range(n_depth_levels):
                this_depth = 1 / (inverse_depth_base + depth_i * inverse_depth_step)

                warping = K_R_Kinv_UV + (Kt / this_depth)
                warping = warping.transpose(dim0=1, dim1=2)
                warping = warping[:, :, 0:2] / (warping[:, :, 2].unsqueeze(-1) + 1e-8)
                warping = warping.view(batchsize, height, width, 2)
                warping[:, :, :, 0] = (warping[:, :, :, 0] - width_normalizer) / width_normalizer
                warping[:, :, :, 1] = (warping[:, :, :, 1] - height_normalizer) / height_normalizer
                warpings[m].append(warping.cpu().detach().numpy().copy())

        return np.array(warpings)


    def calculate_cost_volume_by_warping(self, image1, image2, warping, out_shape):
        n_depth_levels = self.n_depth_levels
        cost_volume = np.zeros(out_shape)

        for depth_i in range(n_depth_levels):
            # warped_image2 = grid_sample(image2, warping[depth_i][0])
            warped_image2 = torch.nn.functional.grid_sample(input=torch.tensor(image2.astype(np.float32).transpose(0, 3, 1, 2)),
                                                            grid=torch.tensor(warping[depth_i]),
                                                            mode='bilinear',
                                                            padding_mode='zeros',
                                                            align_corners=True)
            warped_image2 = warped_image2.detach().numpy().copy().transpose(0, 2, 3, 1)
            cost_volume[:,:,:,depth_i] = np.sum(image1 * warped_image2, axis=3) / image1.shape[-1]

        return cost_volume


    def __call__(self, frame_number, image1, n_measurement_frames, *image2s):
        n_depth_levels = self.n_depth_levels
        rshift = self.rshift

        batchsize, height, width, _ = image1.shape
        out_shape = (batchsize, height, width, n_depth_levels)
        fused_cost_volume = np.zeros(out_shape)
        warpings = self.calc_warpings(frame_number[0], batchsize, height, width, n_measurement_frames[0])
        for m in range(n_measurement_frames[0]):
            fused_cost_volume += self.calculate_cost_volume_by_warping(image1, image2s[m], warpings[m], out_shape)

        return round_and_clip((fused_cost_volume / n_measurement_frames) / (1 << rshift), image1.dtype)


def cost_volume_fusion(frame_number, act78, n_measurement_frames, act79s, K, pose1s, pose2ss, act_dtype=ng.int16):

    externs = []

    # [80] conv
    act80 = ng.extern([frame_number, act78, n_measurement_frames, *act79s], shape=(1, 32, 48, 64), dtype=act_dtype, opcode=0x80, func=fusion(11, K, pose1s, pose2ss))
    externs.append((act80, [act78, *act79s], "act80 = fusion(11, K, pose1s, pose2ss)(frame_number, act78, n_measurement_frames, *act79s)"))


    return (act80,), externs
