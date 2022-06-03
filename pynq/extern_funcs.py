import torch
import numpy as np
import kornia


def round_and_clip(input, dtype):
    info = np.iinfo(dtype)
    return np.clip(np.round(input).astype(np.int64), info.min, info.max).astype(dtype)


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


class lstm_state_calculator():
    def __init__(self, inputs, prepare_input_value, hshift, cshift):
        self.org_lstm_state = prepare_input_value(inputs["hidden_state"][0].transpose(0, 2, 3, 1), hshift), prepare_input_value(inputs["cell_state"][0].transpose(0, 2, 3, 1), cshift)
        self.full_K_torch = torch.tensor(inputs["full_K"])
        self.half_K_torch = torch.tensor(inputs["half_K"])
        self.camera_matrix = torch.tensor(inputs["lstm_K"])
        self.prepare_input_value = prepare_input_value
        self.hshift = hshift
        self.cshift = cshift
        self.test_image_width = 96
        self.test_image_height = 64


    def get_non_differentiable_rectangle_depth_estimation(self, reference_pose_torch, measurement_pose_torch,
                                                          previous_depth_torch, full_K_torch, half_K_torch,
                                                          original_width, original_height):
        batch_size, _, _ = reference_pose_torch.shape
        half_width = int(original_width / 2)
        half_height = int(original_height / 2)

        trans = torch.bmm(torch.inverse(reference_pose_torch), measurement_pose_torch)
        points_3d_src = kornia.depth_to_3d(previous_depth_torch, full_K_torch, normalize_points=False)
        points_3d_src = points_3d_src.permute(0, 2, 3, 1)
        points_3d_dst = kornia.transform_points(trans[:, None], points_3d_src)

        points_3d_dst = points_3d_dst.view(batch_size, -1, 3)

        z_values = points_3d_dst[:, :, -1]
        z_values = torch.relu(z_values)
        sorting_indices = torch.argsort(z_values, descending=True)
        z_values = torch.gather(z_values, dim=1, index=sorting_indices)

        sorting_indices_for_points = torch.stack([sorting_indices] * 3, dim=-1)
        points_3d_dst = torch.gather(points_3d_dst, dim=1, index=sorting_indices_for_points)

        projections = torch.round(kornia.project_points(points_3d_dst, half_K_torch.unsqueeze(1))).long()
        is_valid_below = (projections[:, :, 0] >= 0) & (projections[:, :, 1] >= 0)
        is_valid_above = (projections[:, :, 0] < half_width) & (projections[:, :, 1] < half_height)
        is_valid = is_valid_below & is_valid_above

        depth_hypothesis = torch.zeros(size=(batch_size, 1, half_height, half_width))
        for projection_index in range(0, batch_size):
            valid_points_zs = z_values[projection_index][is_valid[projection_index]]
            valid_projections = projections[projection_index][is_valid[projection_index]]
            i_s = valid_projections[:, 1]
            j_s = valid_projections[:, 0]
            ij_combined = i_s * half_width + j_s
            _, ij_combined_unique_indices = np.unique(ij_combined.cpu().numpy(), return_index=True)
            ij_combined_unique_indices = torch.from_numpy(ij_combined_unique_indices).long()
            i_s = i_s[ij_combined_unique_indices]
            j_s = j_s[ij_combined_unique_indices]
            valid_points_zs = valid_points_zs[ij_combined_unique_indices]
            torch.index_put_(depth_hypothesis[projection_index, 0], (i_s, j_s), valid_points_zs)
        return depth_hypothesis


    def warp_frame_depth(
            self,
            image_src: torch.Tensor,
            depth_dst: torch.Tensor,
            src_trans_dst: torch.Tensor,
            camera_matrix: torch.Tensor,
            normalize_points: bool = False,
            sampling_mode='bilinear') -> torch.Tensor:
        # TAKEN FROM KORNIA LIBRARY
        if not isinstance(image_src, torch.Tensor):
            raise TypeError(f"Input image_src type is not a torch.Tensor. Got {type(image_src)}.")

        if not len(image_src.shape) == 4:
            raise ValueError(f"Input image_src musth have a shape (B, D, H, W). Got: {image_src.shape}")

        if not isinstance(depth_dst, torch.Tensor):
            raise TypeError(f"Input depht_dst type is not a torch.Tensor. Got {type(depth_dst)}.")

        if not len(depth_dst.shape) == 4 and depth_dst.shape[-3] == 1:
            raise ValueError(f"Input depth_dst musth have a shape (B, 1, H, W). Got: {depth_dst.shape}")

        if not isinstance(src_trans_dst, torch.Tensor):
            raise TypeError(f"Input src_trans_dst type is not a torch.Tensor. "
                            f"Got {type(src_trans_dst)}.")

        if not len(src_trans_dst.shape) == 3 and src_trans_dst.shape[-2:] == (3, 3):
            raise ValueError(f"Input src_trans_dst must have a shape (B, 3, 3). "
                             f"Got: {src_trans_dst.shape}.")

        if not isinstance(camera_matrix, torch.Tensor):
            raise TypeError(f"Input camera_matrix type is not a torch.Tensor. "
                            f"Got {type(camera_matrix)}.")

        if not len(camera_matrix.shape) == 3 and camera_matrix.shape[-2:] == (3, 3):
            raise ValueError(f"Input camera_matrix must have a shape (B, 3, 3). "
                             f"Got: {camera_matrix.shape}.")
        # unproject source points to camera frame
        points_3d_dst: torch.Tensor = kornia.depth_to_3d(depth_dst, camera_matrix, normalize_points)  # Bx3xHxW

        # transform points from source to destination
        points_3d_dst = points_3d_dst.permute(0, 2, 3, 1)  # BxHxWx3

        # apply transformation to the 3d points
        points_3d_src = kornia.transform_points(src_trans_dst[:, None], points_3d_dst)  # BxHxWx3
        points_3d_src[:, :, :, 2] = torch.relu(points_3d_src[:, :, :, 2])

        # project back to pixels
        camera_matrix_tmp: torch.Tensor = camera_matrix[:, None, None]  # Bx1x1xHxW
        points_2d_src: torch.Tensor = kornia.project_points(points_3d_src, camera_matrix_tmp)  # BxHxWx2

        # normalize points between [-1 / 1]
        height, width = depth_dst.shape[-2:]
        points_2d_src_norm: torch.Tensor = kornia.normalize_pixel_coordinates(points_2d_src, height, width)  # BxHxWx2

        return torch.nn.functional.grid_sample(image_src, points_2d_src_norm, align_corners=True, mode=sampling_mode)


    def __call__(self, lstm_state, previous_depth, previous_pose, current_pose):
        hshift = self.hshift
        cshift = self.cshift
        prepare_input_value = self.prepare_input_value

        if lstm_state is None:
            return self.org_lstm_state
        if previous_pose is None:
            if lstm_state[0].dtype == np.int64:
                return lstm_state
            else:
                return prepare_input_value(lstm_state[0].transpose(0, 2, 3, 1), hshift), prepare_input_value(lstm_state[1].transpose(0, 2, 3, 1), cshift)

        full_K_torch = self.full_K_torch
        half_K_torch = self.half_K_torch
        camera_matrix = self.camera_matrix
        test_image_width = self.test_image_width
        test_image_height = self.test_image_height

        previous_depth = torch.tensor(previous_depth)
        previous_pose = torch.tensor(previous_pose)
        current_pose = torch.tensor(current_pose)

        if previous_depth is not None:
            depth_estimation = self.get_non_differentiable_rectangle_depth_estimation(reference_pose_torch=current_pose,
                                                                                      measurement_pose_torch=previous_pose,
                                                                                      previous_depth_torch=previous_depth,
                                                                                      full_K_torch=full_K_torch,
                                                                                      half_K_torch=half_K_torch,
                                                                                      original_height=test_image_height,
                                                                                      original_width=test_image_width)
            depth_estimation = torch.nn.functional.interpolate(input=depth_estimation,
                                                               scale_factor=(1.0 / 16.0),
                                                               mode="nearest")
        else:
            depth_estimation = torch.zeros(size=(1, 1, int(test_image_height / 32.0), int(test_image_width / 32.0)))


        if lstm_state[0].dtype == np.int64:
            h_cur = lstm_state[0].transpose(0, 3, 1, 2).astype(np.float32) / (1 << hshift)
            c_cur = lstm_state[1]
        else:
            h_cur = lstm_state[0].astype(np.float32)
            c_cur = prepare_input_value(lstm_state[1].transpose(0, 2, 3, 1), cshift)


        h_cur = torch.tensor(h_cur)
        transformation = torch.bmm(torch.inverse(previous_pose), current_pose)

        non_valid = depth_estimation <= 0.01
        h_cur = self.warp_frame_depth(image_src=h_cur,
                                      depth_dst=depth_estimation,
                                      src_trans_dst=transformation,
                                      camera_matrix=camera_matrix,
                                      normalize_points=False,
                                      sampling_mode='bilinear')
        b, c, h, w = h_cur.size()
        non_valid = torch.cat([non_valid] * c, dim=1)
        h_cur.data[non_valid] = 0.0
        return prepare_input_value(h_cur.detach().numpy().copy().transpose(0, 2, 3, 1), hshift), c_cur



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



class celu():
    def __init__(self, xshift):
        self.xshift = xshift

    def __call__(self, x):
        # celu_table = [0, -2016, -3971, -5865, -7701, -9480, -11205, -12876, -14497, -16067, -17589, -19064, -20494, -21880, -23223, -24525, -25786, -27009, -28195, -29344, -30457, -31536, -32582, -33596, -34579, -35531, -36455, -37349, -38217, -39057, -39872, -40661, -41427, -42168, -42887, -43584, -44260, -44914, -45549, -46164, -46760, -47337, -47897, -48440, -48966, -49476, -49970, -50449, -50913, -51363, -51799, -52222, -52631, -53028, -53413, -53786, -54148, -54498, -54838, -55167, -55486, -55795, -56095, -56385, -56667, -56940, -57204, -57460, -57709, -57950, -58183, -58409, -58629, -58841, -59047, -59247, -59440, -59628, -59810, -59986, -60156, -60322, -60482, -60638, -60789, -60935, -61076, -61213, -61346, -61475, -61600, -61721, -61839, -61952, -62063, -62170, -62273, -62374, -62471, -62565, -62657, -62745, -62831, -62914, -62995, -63073, -63149, -63222, -63293, -63362, -63429, -63494, -63557, -63618, -63677, -63734, -63790, -63843, -63895, -63946, -63995, -64042, -64088, -64133, -64176, -64218, -64258, -64298, -64336, -64373, -64408, -64443, -64477, -64509, -64541, -64572, -64601, -64630, -64658, -64685, -64711, -64736, -64761, -64785, -64808, -64830, -64852, -64873, -64894, -64913, -64932, -64951, -64969, -64986, -65003, -65020, -65036, -65051, -65066, -65080, -65094, -65108, -65121, -65134, -65146, -65158, -65170, -65181, -65192, -65203, -65213, -65223, -65233, -65242, -65251, -65260, -65268, -65276, -65284, -65292, -65300, -65307, -65314, -65321, -65327, -65334, -65340, -65346, -65352, -65358, -65363, -65368, -65374, -65379, -65383, -65388, -65393, -65397, -65401, -65405, -65409, -65413, -65417, -65421, -65424, -65428, -65431, -65434, -65437, -65441, -65443, -65446, -65449, -65452, -65454, -65457, -65459, -65462, -65464, -65466, -65468, -65470, -65472, -65474, -65476, -65478, -65480, -65482, -65483, -65485, -65486, -65488, -65489, -65491, -65492, -65494, -65495, -65496, -65497, -65499, -65500, -65501, -65502, -65503, -65504, -65505, -65506, -65507, -65508, -65509, -65509, -65510, -65511, -65512, -65513, -65513]
        celu_table = [0, -126, -248, -367, -481, -593, -700, -805, -906, -1004, -1099, -1192, -1281, -1367, -1451, -1533, -1612, -1688, -1762, -1834, -1904, -1971, -2036, -2100, -2161, -2221, -2278, -2334, -2389, -2441, -2492, -2541, -2589, -2636, -2680, -2724, -2766, -2807, -2847, -2885, -2922, -2959, -2994, -3027, -3060, -3092, -3123, -3153, -3182, -3210, -3237, -3264, -3289, -3314, -3338, -3362, -3384, -3406, -3427, -3448, -3468, -3487, -3506, -3524, -3542, -3559, -3575, -3591, -3607, -3622, -3636, -3651, -3664, -3678, -3690, -3703, -3715, -3727, -3738, -3749, -3760, -3770, -3780, -3790, -3799, -3808, -3817, -3826, -3834, -3842, -3850, -3858, -3865, -3872, -3879, -3886, -3892, -3898, -3904, -3910, -3916, -3922, -3927, -3932, -3937, -3942, -3947, -3951, -3956, -3960, -3964, -3968, -3972, -3976, -3980, -3983, -3987, -3990, -3993, -3997, -4000, -4003, -4006, -4008, -4011, -4014, -4016, -4019, -4021, -4023, -4026, -4028, -4030, -4032, -4034, -4036, -4038, -4039, -4041, -4043, -4044, -4046, -4048, -4049, -4050, -4052, -4053, -4055, -4056, -4057, -4058, -4059, -4061, -4062, -4063, -4064, -4065, -4066, -4067, -4068, -4068, -4069, -4070, -4071, -4072, -4072, -4073, -4074, -4075, -4075, -4076, -4076, -4077, -4078, -4078, -4079, -4079, -4080, -4080, -4081, -4081, -4082, -4082, -4083, -4083, -4083, -4084, -4084, -4084, -4085, -4085, -4086, -4086, -4086, -4086, -4087, -4087, -4087, -4088, -4088, -4088, -4088, -4089, -4089, -4089, -4089, -4089, -4090, -4090, -4090, -4090, -4090, -4091, -4091, -4091, -4091, -4091, -4091, -4091, -4092, -4092, -4092, -4092, -4092, -4092, -4092, -4092, -4093, -4093, -4093, -4093, -4093, -4093, -4093, -4093, -4093, -4093, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4095, -4095]
        tbbit = 8
        tbshift = 5
        celushift = 12
        xshift = self.xshift

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):
                    for l in range(x.shape[3]):
                        tb_idx = (-x[i][j][k][l]) >> (xshift - tbshift)
                        if x[i][j][k][l] > 0:
                            x[i][j][k][l] = x[i][j][k][l]
                        elif tb_idx >= (1 << tbbit):
                            x[i][j][k][l] = -1 << xshift
                        else:
                            x[i][j][k][l] = np.round(celu_table[tb_idx] / (float) (1 << (celushift - xshift))).astype(x.dtype)

        return x


class ln():
    def __init__(self, lshift):
        self.lshift = lshift

    def __call__(self, x):
        eps = 1e-5
        e = np.mean(x.reshape(-1, x.shape[-1]), axis=0)
        v = np.var(x.reshape(-1, x.shape[-1]), axis=0)
        return round_and_clip((x - e) / np.sqrt(v + eps) * (1 << self.lshift), x.dtype)