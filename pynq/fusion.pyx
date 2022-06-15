# distutils: language=c++
# distutils: extra_compile_args = ["-O3", "-fopenmp"]
# distutils: extra_link_args=["-fopenmp"]
# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
# cython: cdivision=True

import numpy as np
from libc.math cimport round
from cython.parallel import prange


cpdef float[:, :, :, :, :] prep_cython(char n_measurement_frames, short[:, :, :, :, :] image2s,
                                       float[:, :] K, float[:, :] inv_K,
                                       float[:, :] pose1, float[:, :, :] inv_pose2s, float[:, :] warp_grid):

    cdef char batchsize = 1
    cdef char height = 32
    cdef char width = 48
    cdef char channels = 32
    cdef char n_depth_levels = 64

    cdef float min_depth = 0.25
    cdef float max_depth = 20.0

    cdef float inverse_depth_base = 1.0 / max_depth
    cdef float inverse_depth_step = (1.0 / min_depth - 1.0 / max_depth) / (n_depth_levels - 1)

    cdef float width_normalizer = width / 2.0
    cdef float height_normalizer = height / 2.0

    cdef float[:, :, :, :, :] warped_image2s = np.zeros((n_depth_levels, batchsize, height, width, channels), dtype=np.float32)

    cdef float[3][3] R
    cdef float[3] t
    cdef float[3] Kt
    cdef float[3][3] K_R
    cdef float[3][3] K_R_Kinv
    cdef float[1536][3] K_R_Kinv_UV

    cdef float this_depth
    cdef float[3] Ktd
    cdef float w2
    cdef float[32][48][2] warping

    cdef short[32][48][2] warping_short

    cdef short m
    cdef short i
    cdef short j
    cdef short k
    cdef short depth_i

    cdef short height1 = height-1
    cdef short width1 = width-1

    cdef short x_int
    cdef short y_int
    cdef short[2] xs
    cdef short[2] ys
    cdef short[2] dxs
    cdef short[2] dys
    cdef char xi
    cdef char yi
    cdef char rshift = ((n_measurement_frames-1) + 8 * 2)
    cdef int s

    for m in range(n_measurement_frames):
        for i in range(3):
            for j in range(3):
                R[i][j] = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    R[i][j] += inv_pose2s[m][i][k] * pose1[k][j]


        for i in range(3):
            t[i] = 0
        for i in range(3):
            for j in range(3):
                t[i] += inv_pose2s[m][i][j] * pose1[j][3]
            t[i] += inv_pose2s[m][i][3]

        for i in range(3):
            Kt[i] = 0
        for i in range(2):
            for j in range(3):
                Kt[i] += K[i][j] * t[j]
        Kt[2] = t[2]

        for i in range(3):
            for j in range(3):
                K_R[i][j] = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    K_R[i][j] += K[i][k] * R[k][j]

        for i in range(3):
            for j in range(3):
                K_R_Kinv[i][j] = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    K_R_Kinv[i][j] += K_R[i][k] * inv_K[k][j]

        for j in range(1536):
            for i in range(3):
                K_R_Kinv_UV[j][i] = 0
        for j in range(1536):
            for i in range(3):
                for k in range(3):
                    K_R_Kinv_UV[j][i] += K_R_Kinv[i][k] * warp_grid[k][j]


        for depth_i in range(n_depth_levels):
            this_depth = inverse_depth_base + depth_i * inverse_depth_step

            for i in range(3):
                Ktd[i] = Kt[i] * this_depth

            k = 0
            for i in range(height):
                for j in range(width):
                    k = i * width + j
                    w2 = K_R_Kinv_UV[k][2] + Ktd[2] + 1e-8
                    warping[i][j][0] = ((K_R_Kinv_UV[k][0] + Ktd[0]) / (w2 * width_normalizer)) - 1
                    warping[i][j][1] = ((K_R_Kinv_UV[k][1] + Ktd[1]) / (w2 * height_normalizer)) - 1

            for i in prange(height, nogil=True, schedule='static', chunksize=1):
            # for i in range(height):
                for j in range(width):
                    warping_short[i][j][0] = <short>round((warping[i][j][0] + 1) * (width1 / 2.0 * (1 << 8)))
                    warping_short[i][j][1] = <short>round((warping[i][j][1] + 1) * (height1 / 2.0 * (1 << 8)))

            # for j in prange(height, nogil=True, schedule='static', chunksize=1):
            for j in range(height):
                for k in range(width):
                    x_int = (warping_short[j][k][0] >> 8)
                    y_int = (warping_short[j][k][1] >> 8)
                    xs[0] = x_int
                    xs[1] = x_int + 1
                    ys[0] = y_int
                    ys[1] = y_int + 1
                    dxs[0] = warping_short[j][k][0] & ((1 << 8) - 1)
                    dys[0] = warping_short[j][k][1] & ((1 << 8) - 1)
                    dxs[1] = (1 << 8) - dxs[0]
                    dys[1] = (1 << 8) - dys[0]
                    for i in range(channels):
                        s = 1 << (rshift-1)
                        for yi in range(2):
                            for xi in range(2):
                                if not(ys[yi] < 0 or height1 < ys[yi] or xs[xi] < 0 or width1 < xs[xi]):
                                    s += dys[1-yi] * dxs[1-xi] * image2s[m][0][ys[yi]][xs[xi]][i]
                        warped_image2s[depth_i][0][j][k][i] += <short>(s >> rshift)

    return warped_image2s


cpdef short[:, :, :, :] fusion_quantize_cython(short[:, :, :, :] image1, short[:, :, :, :, :] warped_image2s):

    cdef char batchsize = 1
    cdef char height = 32
    cdef char width = 48
    cdef char channels = 32
    cdef char n_depth_levels = 64
    cdef char rshift = 16

    cdef short short_max = 32767

    cdef short[:, :, :, :] fused_cost_volume = np.zeros((batchsize, height, width, n_depth_levels), dtype=np.int16)
    cdef int s
    cdef char depth_i
    cdef char b
    cdef char h
    cdef char w
    cdef char c

    for depth_i in prange(n_depth_levels, nogil=True, schedule='static', chunksize=1):
        for b in range(batchsize):
            for h in range(height):
                for w in range(width):
                    s = (1 << (rshift-1))
                    for c in range(channels):
                        s += image1[b][h][w][c] * warped_image2s[depth_i][b][h][w][c]
                    s = s >> rshift
                    fused_cost_volume[b][h][w][depth_i] = <short>s

    return fused_cost_volume
