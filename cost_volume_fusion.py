import numpy as np
import nngen as ng
import torch
from utils import round_and_clip

class Fusion():
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


    def prep(self, frame_number, n_measurement_frames, image2s):
        self.n_measurement_frames = n_measurement_frames
        n_depth_levels = self.n_depth_levels

        batchsize, height, width, channels = image2s[0].shape
        warpings = self.calc_warpings(frame_number[0], batchsize, height, width, n_measurement_frames[0])

        self.warped_image2s = [[] for _ in range(n_measurement_frames[0])]
        for m in range(n_measurement_frames[0]):
            warping = warpings[m]
            image2 = image2s[m]
            for depth_i in range(n_depth_levels):
                # warped_image2 = grid_sample(image2, warping[depth_i][0])
                warped_image2 = torch.nn.functional.grid_sample(input=torch.tensor(image2.astype(np.float32).transpose(0, 3, 1, 2)),
                                                                grid=torch.tensor(warping[depth_i]),
                                                                mode='bilinear',
                                                                padding_mode='zeros',
                                                                align_corners=True)
                self.warped_image2s[m].append(warped_image2.detach().numpy().copy().transpose(0, 2, 3, 1) / channels)


    def __call__(self, image1):
        n_depth_levels = self.n_depth_levels
        n_measurement_frames = self.n_measurement_frames
        rshift = self.rshift

        batchsize, height, width, _ = image1.shape
        fused_cost_volume = np.zeros((batchsize, height, width, n_depth_levels))
        for m in range(n_measurement_frames[0]):
            for depth_i in range(n_depth_levels):
                warped_image2 = self.warped_image2s[m][depth_i]
                fused_cost_volume[:,:,:,depth_i] += np.sum(image1 * warped_image2, axis=3)

        return round_and_clip((fused_cost_volume / n_measurement_frames) / (1 << rshift), image1.dtype)


def cost_volume_fusion(act78, K, pose1s, pose2ss, act_dtype):

    externs = []

    # [79] cost volume fusion
    fusion = Fusion(11, K, pose1s, pose2ss)
    act79 = ng.extern([act78], shape=(1, 32, 48, 64), dtype=act_dtype, opcode=0x79, func=fusion)
    externs.append((act79, [act78], "act79 = fusion(act78)"))


    return (act79,), externs, fusion

