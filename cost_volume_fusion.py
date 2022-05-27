import numpy as np
import nngen as ng

n_depth_levels = 64

def grid_sample(image, warping):
    _, height, width, channels = image.shape
    warped_image = np.zeros(image.shape)

    for j in range(height):
        for k in range(width):
            x = (warping[j][k][0] + 1) * (width - 1) / 2.0
            y = (warping[j][k][1] + 1) * (height - 1) / 2.0
            y_int = int(y)
            x_int = int(x)
            ys = [y_int, y_int + 1]
            xs = [x_int, x_int + 1]
            dys = [y - ys[0], ys[1] - y]
            dxs = [x - xs[0], xs[1] - x]
            for i in range(channels):
                for yi in range(2):
                    for xi in range(2):
                        if not(ys[yi] < 0 or height-1 < ys[yi] or xs[xi] < 0 or width-1 < xs[xi]):
                            warped_image[0][j][k][i] += dys[1-yi] * dxs[1-xi] * image[0][ys[yi]][xs[xi]][i]

    return warped_image


def calculate_cost_volume_by_warping(image1, image2, warping, out_shape):
    cost_volume = np.zeros(out_shape)

    for i in range(image2.shape[2]):
        print(image2[0][0][i][0])
    print("aaaa")

    for i in range(warping.shape[3]):
        for j in range(warping.shape[4]):
            print(warping[0][0][0][i][j])


    for depth_i in range(n_depth_levels):
        warped_image2 = grid_sample(image2, warping[depth_i][0])
        cost_volume[:,:,:,depth_i] = np.mean(image1 * warped_image2, axis=3)

    return cost_volume

    # for (int idx = 0; idx < n_depth_levels * height_2 * width_2; idx++)
    #     cost_volume[idx] = 0;

    # for (int depth_i = 0; depth_i < n_depth_levels; depth_i++) {
    #     float warped_image2[fpn_output_channels * height_2 * width_2];
    #     grid_sample(image2, warping + depth_i * (height_2 * width_2 * 2), warped_image2, fpn_output_channels, height_2, width_2);

    #     for (int i = 0; i < fpn_output_channels; i++) for (int idx = 0; idx < height_2 * width_2; idx++)
    #         cost_volume[depth_i * (height_2 * width_2) + idx] += (image1[i * (height_2 * width_2) + idx] * warped_image2[i * (height_2 * width_2) + idx]) / fpn_output_channels;


class fusion():
    def __init__(self, rshift, warpings, n_measurement_frames):
        self.rshift = rshift
        self.warpings = warpings
        self.n_measurement_frames = n_measurement_frames

    def __call__(self, image1, *image2s):
    # def __call__(self, image1, image2_0, image2_1):
        rshift = self.rshift
        warpings = self.warpings
        n_measurement_frames = self.n_measurement_frames

        # float fused_cost_volume_float[n_depth_levels * height_2 * width_2];

        # for (int idx = 0; idx < n_depth_levels * height_2 * width_2; idx++)
        #     fused_cost_volume_float[idx] = 0;

        # image2s = [image2_0, image2_1]

        batchsize, height, width, _ = image1.shape
        out_shape = (batchsize, height, width, n_depth_levels)
        fused_cost_volume = np.zeros(out_shape)
        for m in range(n_measurement_frames):
            fused_cost_volume += calculate_cost_volume_by_warping(image1, image2s[m], warpings[m], out_shape)

            # float cost_volume[n_depth_levels * height_2 * width_2];
            # const qaint* image2 = image2s + m * (fpn_output_channels * height_2 * width_2);
            # const float* warping = warpings + m * (n_depth_levels * height_2 * width_2 * 2);
            # calculate_cost_volume_by_warping(image1, image2, warping, cost_volume);
            # for (int idx = 0; idx < n_depth_levels * height_2 * width_2; idx++)
            #     fused_cost_volume_float[idx] += cost_volume[idx];
        # }

        # const int xshift = cout_shifts[conv_cnt-1] * 2;
        # const int yshift = cin_shifts[conv_cnt];  // (not necessarily)
        # print_neg_shift("cost_volume_fusion", "yshift", yshift);
        # print_neg_shift("cost_volume_fusion", "xshift - yshift", xshift - yshift);
        # for (int idx = 0; idx < n_depth_levels * height_2 * width_2; idx++)
        #     fused_cost_volume[idx] = (fused_cost_volume_float[idx] / n_measurement_frames) / (1 << (xshift - yshift));

        return np.round((fused_cost_volume / n_measurement_frames) / (1 << rshift)).astype(image1.dtype)



def cost_volume_fusion(act78, act79s, warpings, n_measurement_frames=2):
# def cost_volume_fusion(act78, measurement_feature0, measurement_feature1, warpings, n_measurement_frames=2):
    # act79 = ng.extern([reference_feature_half, measurement_feature0, measurement_feature1], opcode=0x79, func=fusion(rshift, warpings, n_measurement_frames))
    # act79.shape = (batchsize, height_2, width_2, n_depth_levels)

    # [80] conv
    act80 = ng.extern([act78, *act79s], opcode=0x80, func=fusion(11, warpings, n_measurement_frames))
    # act80 = ng.extern([act78, measurement_feature0, measurement_feature1], opcode=0x80, func=fusion(11, warpings, n_measurement_frames))
    act80.shape = (1, 32, 48, 64)

    return act80
