import numpy as np
import nngen as ng
from utils import sigmoid, rshift_round_and_clip, round_and_clip

# class celu():
#     def __init__(self, xshift):
#         self.xshift = xshift

#     def __call__(self, x):
#         # celu_table = [0, -2016, -3971, -5865, -7701, -9480, -11205, -12876, -14497, -16067, -17589, -19064, -20494, -21880, -23223, -24525, -25786, -27009, -28195, -29344, -30457, -31536, -32582, -33596, -34579, -35531, -36455, -37349, -38217, -39057, -39872, -40661, -41427, -42168, -42887, -43584, -44260, -44914, -45549, -46164, -46760, -47337, -47897, -48440, -48966, -49476, -49970, -50449, -50913, -51363, -51799, -52222, -52631, -53028, -53413, -53786, -54148, -54498, -54838, -55167, -55486, -55795, -56095, -56385, -56667, -56940, -57204, -57460, -57709, -57950, -58183, -58409, -58629, -58841, -59047, -59247, -59440, -59628, -59810, -59986, -60156, -60322, -60482, -60638, -60789, -60935, -61076, -61213, -61346, -61475, -61600, -61721, -61839, -61952, -62063, -62170, -62273, -62374, -62471, -62565, -62657, -62745, -62831, -62914, -62995, -63073, -63149, -63222, -63293, -63362, -63429, -63494, -63557, -63618, -63677, -63734, -63790, -63843, -63895, -63946, -63995, -64042, -64088, -64133, -64176, -64218, -64258, -64298, -64336, -64373, -64408, -64443, -64477, -64509, -64541, -64572, -64601, -64630, -64658, -64685, -64711, -64736, -64761, -64785, -64808, -64830, -64852, -64873, -64894, -64913, -64932, -64951, -64969, -64986, -65003, -65020, -65036, -65051, -65066, -65080, -65094, -65108, -65121, -65134, -65146, -65158, -65170, -65181, -65192, -65203, -65213, -65223, -65233, -65242, -65251, -65260, -65268, -65276, -65284, -65292, -65300, -65307, -65314, -65321, -65327, -65334, -65340, -65346, -65352, -65358, -65363, -65368, -65374, -65379, -65383, -65388, -65393, -65397, -65401, -65405, -65409, -65413, -65417, -65421, -65424, -65428, -65431, -65434, -65437, -65441, -65443, -65446, -65449, -65452, -65454, -65457, -65459, -65462, -65464, -65466, -65468, -65470, -65472, -65474, -65476, -65478, -65480, -65482, -65483, -65485, -65486, -65488, -65489, -65491, -65492, -65494, -65495, -65496, -65497, -65499, -65500, -65501, -65502, -65503, -65504, -65505, -65506, -65507, -65508, -65509, -65509, -65510, -65511, -65512, -65513, -65513]
#         celu_table = [0, -126, -248, -367, -481, -593, -700, -805, -906, -1004, -1099, -1192, -1281, -1367, -1451, -1533, -1612, -1688, -1762, -1834, -1904, -1971, -2036, -2100, -2161, -2221, -2278, -2334, -2389, -2441, -2492, -2541, -2589, -2636, -2680, -2724, -2766, -2807, -2847, -2885, -2922, -2959, -2994, -3027, -3060, -3092, -3123, -3153, -3182, -3210, -3237, -3264, -3289, -3314, -3338, -3362, -3384, -3406, -3427, -3448, -3468, -3487, -3506, -3524, -3542, -3559, -3575, -3591, -3607, -3622, -3636, -3651, -3664, -3678, -3690, -3703, -3715, -3727, -3738, -3749, -3760, -3770, -3780, -3790, -3799, -3808, -3817, -3826, -3834, -3842, -3850, -3858, -3865, -3872, -3879, -3886, -3892, -3898, -3904, -3910, -3916, -3922, -3927, -3932, -3937, -3942, -3947, -3951, -3956, -3960, -3964, -3968, -3972, -3976, -3980, -3983, -3987, -3990, -3993, -3997, -4000, -4003, -4006, -4008, -4011, -4014, -4016, -4019, -4021, -4023, -4026, -4028, -4030, -4032, -4034, -4036, -4038, -4039, -4041, -4043, -4044, -4046, -4048, -4049, -4050, -4052, -4053, -4055, -4056, -4057, -4058, -4059, -4061, -4062, -4063, -4064, -4065, -4066, -4067, -4068, -4068, -4069, -4070, -4071, -4072, -4072, -4073, -4074, -4075, -4075, -4076, -4076, -4077, -4078, -4078, -4079, -4079, -4080, -4080, -4081, -4081, -4082, -4082, -4083, -4083, -4083, -4084, -4084, -4084, -4085, -4085, -4086, -4086, -4086, -4086, -4087, -4087, -4087, -4088, -4088, -4088, -4088, -4089, -4089, -4089, -4089, -4089, -4090, -4090, -4090, -4090, -4090, -4091, -4091, -4091, -4091, -4091, -4091, -4091, -4092, -4092, -4092, -4092, -4092, -4092, -4092, -4092, -4093, -4093, -4093, -4093, -4093, -4093, -4093, -4093, -4093, -4093, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4094, -4095, -4095]
#         tbbit = 8
#         tbshift = 5
#         celushift = 12
#         xshift = self.xshift
#         y = np.zeros_like(x)

#         for i in range(x.shape[0]):
#             for j in range(x.shape[1]):
#                 for k in range(x.shape[2]):
#                     for l in range(x.shape[3]):
#                         tb_idx = (-x[i][j][k][l]) >> (xshift - tbshift)
#                         if x[i][j][k][l] > 0:
#                             y[i][j][k][l] = x[i][j][k][l]
#                         elif tb_idx >= (1 << tbbit):
#                             y[i][j][k][l] = -1 << xshift
#                         else:
#                             y[i][j][k][l] = np.round(celu_table[tb_idx] / (float) (1 << (celushift - xshift))).astype(x.dtype)

#         return y


class ln():
    def __init__(self, lshift):
        self.lshift = lshift

    def __call__(self, x):
        eps = 1e-5
        e = np.mean(x.reshape(-1, x.shape[-1]), axis=0)
        v = np.var(x.reshape(-1, x.shape[-1]), axis=0)
        return round_and_clip((x - e) / np.sqrt(v + eps) * (1 << self.lshift), x.dtype)


def LSTMFusion(act99, act100, act101, params, par_ich, par_och, par_och_k5, par,
               weight_dtype, bias_dtype, scale_dtype, act_dtype, mid_dtype):

    externs = []

    # [102] cat
    act102 = ng.concat([act99, act100], axis=3)


    # [103] conv
    weight103 = ng.variable(dtype=weight_dtype, shape=(2048, 3, 3, 1024), name="lstm_cell.conv.weight")
    weight103.set_value(params["lstm_cell.conv.weight"])

    rshift103 = ng.constant([11], dtype=ng.int8)
    act103 = ng.conv2d(act102, weight103, strides=(1, 1, 1, 1), rshift_out=rshift103, asymmetric_clip=True, par_ich=par_ich, par_och=par_och, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [104] sig_ln_celu
    slice104s = [ng.slice_(act103, (0, 0, 0, i * 512), (1, 2, 3, (i+1) * 512), (1, 1, 1, 1)) for i in range(4)]

    rshift104 = ng.constant([4], dtype=ng.int8)
    ii104, ff104, oo104 = [sigmoid(ng.rshift_round(slice104s[i], rshift104, par=par), par=par) for i in range(3)]

    ln104 = ng.extern([slice104s[3]], opcode=0x104, func=ln(12))
    externs.append((ln104, [slice104s[3]], "ln104 = ln(12)(slice104s[3])"))
    gg104 = ng.celu(ln104, rshift_lut_in=7, lut_clip=8.0, range_rate=0.125, dtype=act_dtype, par=par)


    # [105] cell_state
    in_rshift105 = ng.constant([2], dtype=ng.int8)
    out_rshift105 = ng.constant([12], dtype=ng.int8)
    sum105 = rshift_round_and_clip(ng.add(ng.multiply(ng.rshift_round(ff104, in_rshift105, par=par), act101, par=par, dtype=mid_dtype), ng.multiply(ng.rshift_round(ii104, in_rshift105, par=par), gg104, par=par, dtype=mid_dtype), par=par), out_rshift105, par=par, dtype=act_dtype)
    act105 = ng.extern([sum105], opcode=0x105, func=ln(12))
    externs.append((act105, [sum105], "act105 = ln(12)(sum105)"))


    # [106] hidden_state
    celu106 = ng.celu(act105, rshift_lut_in=7, lut_clip=8.0, range_rate=0.125, dtype=act_dtype, par=par)
    rshift106 = ng.constant([13], dtype=ng.int8)
    act106 = rshift_round_and_clip(ng.multiply(celu106, oo104, par=par, dtype=mid_dtype), rshift106, par=par, dtype=act_dtype)


    return (act106, act105), externs
