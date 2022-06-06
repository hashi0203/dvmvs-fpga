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
            mid = torch.nn.functional.interpolate(mid, size=out_shape, mode=mode, align_corners=True)
            output = mid.detach().numpy().copy().transpose(0, 2, 3, 1).astype(input.dtype)

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


class ln():
    def __init__(self, lshift):
        self.lshift = lshift

    def __call__(self, x):
        eps = 1e-5
        e = np.mean(x.reshape(-1, x.shape[-1]), axis=0)
        v = np.var(x.reshape(-1, x.shape[-1]), axis=0)
        return round_and_clip((x - e) / np.sqrt(v + eps) * (1 << self.lshift), x.dtype)