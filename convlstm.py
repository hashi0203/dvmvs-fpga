import numpy as np
import nngen as ng


class celu():
    def __init__(self, xshift):
        self.xshift = xshift

    def __call__(self, x):
        celu_table = [0, -2016, -3971, -5865, -7701, -9480, -11205, -12876, -14497, -16067, -17589, -19064, -20494, -21880, -23223, -24525, -25786, -27009, -28195, -29344, -30457, -31536, -32582, -33596, -34579, -35531, -36455, -37349, -38217, -39057, -39872, -40661, -41427, -42168, -42887, -43584, -44260, -44914, -45549, -46164, -46760, -47337, -47897, -48440, -48966, -49476, -49970, -50449, -50913, -51363, -51799, -52222, -52631, -53028, -53413, -53786, -54148, -54498, -54838, -55167, -55486, -55795, -56095, -56385, -56667, -56940, -57204, -57460, -57709, -57950, -58183, -58409, -58629, -58841, -59047, -59247, -59440, -59628, -59810, -59986, -60156, -60322, -60482, -60638, -60789, -60935, -61076, -61213, -61346, -61475, -61600, -61721, -61839, -61952, -62063, -62170, -62273, -62374, -62471, -62565, -62657, -62745, -62831, -62914, -62995, -63073, -63149, -63222, -63293, -63362, -63429, -63494, -63557, -63618, -63677, -63734, -63790, -63843, -63895, -63946, -63995, -64042, -64088, -64133, -64176, -64218, -64258, -64298, -64336, -64373, -64408, -64443, -64477, -64509, -64541, -64572, -64601, -64630, -64658, -64685, -64711, -64736, -64761, -64785, -64808, -64830, -64852, -64873, -64894, -64913, -64932, -64951, -64969, -64986, -65003, -65020, -65036, -65051, -65066, -65080, -65094, -65108, -65121, -65134, -65146, -65158, -65170, -65181, -65192, -65203, -65213, -65223, -65233, -65242, -65251, -65260, -65268, -65276, -65284, -65292, -65300, -65307, -65314, -65321, -65327, -65334, -65340, -65346, -65352, -65358, -65363, -65368, -65374, -65379, -65383, -65388, -65393, -65397, -65401, -65405, -65409, -65413, -65417, -65421, -65424, -65428, -65431, -65434, -65437, -65441, -65443, -65446, -65449, -65452, -65454, -65457, -65459, -65462, -65464, -65466, -65468, -65470, -65472, -65474, -65476, -65478, -65480, -65482, -65483, -65485, -65486, -65488, -65489, -65491, -65492, -65494, -65495, -65496, -65497, -65499, -65500, -65501, -65502, -65503, -65504, -65505, -65506, -65507, -65508, -65509, -65509, -65510, -65511, -65512, -65513, -65513]
        tbbit = 8
        tbshift = 5
        celushift = 16
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
        return np.round((x - e) / np.sqrt(v + eps) * (1 << self.lshift)).astype(x.dtype)


def LSTMFusion(act100, act101, act102, params, weight_dtype=ng.int8, act_dtype=ng.int32):

    # [103] cat
    act103 = ng.concat([act100, act101], axis=3)


    # [104] conv
    weight104 = ng.variable(dtype=weight_dtype, shape=(2048, 3, 3, 1024), name="lstm_cell.conv.weight")
    weight104.set_value(params["lstm_cell.conv.weight"])

    # bias104 = ng.variable(dtype=bias_dtype, shape=(2048,), name="lstm_cell.conv.bias")
    # bias104.set_value(params["lstm_cell.conv.bias"])

    conv104 = ng.conv2d(act103, weight104, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32)

    # lshift104 = ng.constant([20], dtype=ng.int8)
    # sum104 = ng.add(conv104, ng.lshift(bias104, lshift104))
    rshift104 = ng.constant([11], dtype=ng.int8)
    act104 = ng.rshift_round(conv104, rshift104)


    # [105] sig_ln_celu
    slice105s = [ng.slice_(act104, (0, 0, 0, i * 512), (1, 2, 3, (i+1) * 512), (1, 1, 1, 1)) for i in range(4)]

    rshift105 = ng.constant([4], dtype=ng.int8)
    ii105, ff105, oo105 = [ng.sigmoid(ng.rshift_round(slice105s[i], rshift105), lut_addrwidth=9, lut_clip=8.0, range_rate=0.5, dtype=ng.int16) for i in range(3)]

    gg105 = ng.extern([slice105s[3]], opcode=0x105, func=lambda x : celu(12)(ln(12)(x)))


    # [106] cell_state
    # in_rshift106 = ng.constant([2], dtype=ng.int8)
    # in2_rshift106 = ng.constant([1], dtype=ng.int8)
    # out_rshift106 = ng.constant([12], dtype=ng.int8)
    out_rshift106 = ng.constant([14], dtype=ng.int8)
    sum106 = ng.clip(ng.rshift_round(ng.add(ng.multiply(ff105, act102, dtype=ng.int64), ng.multiply(ii105, gg105, dtype=ng.int64)), out_rshift106), dtype=act_dtype)
    # sum106 = ng.rshift_round(ng.add(ng.multiply(ng.rshift_round(ff105, in_rshift106), act102), ng.multiply(ng.rshift_round(ii105, in_rshift106), gg105)), out_rshift106)
    act106 = ng.extern([sum106], opcode=0x106, func=ln(12))


    # [107] hidden_state
    celu107 = ng.extern([act106], opcode=0x107, func=celu(12))
    # in_rshift107 = ng.constant([1], dtype=ng.int8)
    # rshift107 = ng.constant([15], dtype=ng.int8)
    rshift107 = ng.constant([12], dtype=ng.int8)
    act107 = ng.clip(ng.rshift_round(ng.multiply(celu107, oo105, dtype=ng.int64), rshift107), dtype=act_dtype)


    return act107, act106


