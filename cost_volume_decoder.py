import numpy as np
import nngen as ng
from utils import rshift_round_and_clip, interpolate

def cost_volume_decoder(act0, act82, act87, act92, act97, act107, params,
                        weight_dtype=ng.int8, bias_dtype=ng.int32, scale_dtype=ng.int8, act_dtype=ng.int16, mid_dtype=ng.int32):

    # [108] interpolate
    act108 = ng.extern([act107], shape=(1, 4, 6, 512), opcode=0x108, func=interpolate(4, 6, 0, "bilinear"))


    # [109] conv
    weight109 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 512), name="decoder_block1.up_convolution.conv.0.weight")
    weight109.set_value(params["decoder_block1.up_convolution.conv.0.weight"])

    bias109 = ng.variable(dtype=bias_dtype, shape=(256,), name="decoder_block1.up_convolution.conv.0.bias")
    bias109.set_value(np.round(params["decoder_block1.up_convolution.conv.0.bias"] / (float) (1 << 3)).astype(params["decoder_block1.up_convolution.conv.0.bias"].dtype))

    scale109 = ng.variable(dtype=scale_dtype, shape=(256,), name="decoder_block1.up_convolution.conv.0.scale")
    scale109.set_value(params["decoder_block1.up_convolution.conv.0.scale"])

    conv109 = ng.conv2d(act108, weight109, strides=(1, 1, 1, 1), bias=bias109, scale=scale109, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift109 = ng.constant([16], dtype=ng.int8)
    act109 = ng.relu(rshift_round_and_clip(conv109, rshift109, dtype=act_dtype))


    # [110] cat
    rshift110 = ng.constant([1], dtype=ng.int8)
    act110 = ng.concat([ng.rshift_round(act109, rshift110), act97], axis=3)


    # [111] conv
    weight111 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 512), name="decoder_block1.convolution1.0.weight")
    weight111.set_value(params["decoder_block1.convolution1.0.weight"])

    bias111 = ng.variable(dtype=bias_dtype, shape=(256,), name="decoder_block1.convolution1.0.bias")
    bias111.set_value(np.round(params["decoder_block1.convolution1.0.bias"] / (float) (1 << 5)).astype(params["decoder_block1.convolution1.0.bias"].dtype))

    scale111 = ng.variable(dtype=scale_dtype, shape=(256,), name="decoder_block1.convolution1.0.scale")
    scale111.set_value(params["decoder_block1.convolution1.0.scale"])

    conv111 = ng.conv2d(act110, weight111, strides=(1, 1, 1, 1), bias=bias111, scale=scale111, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift111 = ng.constant([12], dtype=ng.int8)
    act111 = ng.relu(rshift_round_and_clip(conv111, rshift111, dtype=act_dtype))


    # [112] conv
    weight112 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 256), name="decoder_block1.convolution2.0.weight")
    weight112.set_value(params["decoder_block1.convolution2.0.weight"])

    bias112 = ng.variable(dtype=bias_dtype, shape=(256,), name="decoder_block1.convolution2.0.bias")
    bias112.set_value(np.round(params["decoder_block1.convolution2.0.bias"] / (float) (1 << 4)).astype(params["decoder_block1.convolution2.0.bias"].dtype))

    scale112 = ng.variable(dtype=scale_dtype, shape=(256,), name="decoder_block1.convolution2.0.scale")
    scale112.set_value(params["decoder_block1.convolution2.0.scale"])

    conv112 = ng.conv2d(act111, weight112, strides=(1, 1, 1, 1), bias=bias112, scale=scale112, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift112 = ng.constant([11], dtype=ng.int8)
    act112 = ng.relu(rshift_round_and_clip(conv112, rshift112, dtype=act_dtype))


    # [113] conv
    weight113 = ng.variable(dtype=weight_dtype, shape=(1, 3, 3, 256), name="depth_layer_one_sixteen.0.weight")
    weight113.set_value(params["depth_layer_one_sixteen.0.weight"])

    bias113 = ng.variable(dtype=bias_dtype, shape=(1,), name="depth_layer_one_sixteen.0.bias")
    bias113.set_value(np.round(params["depth_layer_one_sixteen.0.bias"] / (float) (1 << 7)).astype(params["depth_layer_one_sixteen.0.bias"].dtype))

    conv113 = ng.conv2d(act112, weight113, strides=(1, 1, 1, 1), bias=bias113, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift113 = ng.constant([18], dtype=ng.int8)
    act113 = ng.sigmoid(ng.rshift_round(conv113, rshift113, dtype=act_dtype), lut_addrwidth=9, lut_clip=8.0, range_rate=0.5, dtype=act_dtype)


    # [114] interpolate
    act114 = ng.extern([act112], shape=(1, 8, 12, 256), opcode=0x114, func=interpolate(8, 12, 0, "bilinear"))


    # [115] conv
    weight115 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 256), name="decoder_block2.up_convolution.conv.0.weight")
    weight115.set_value(params["decoder_block2.up_convolution.conv.0.weight"])

    bias115 = ng.variable(dtype=bias_dtype, shape=(128,), name="decoder_block2.up_convolution.conv.0.bias")
    bias115.set_value(np.round(params["decoder_block2.up_convolution.conv.0.bias"] / (float) (1 << 2)).astype(params["decoder_block2.up_convolution.conv.0.bias"].dtype))

    scale115 = ng.variable(dtype=scale_dtype, shape=(128,), name="decoder_block2.up_convolution.conv.0.scale")
    scale115.set_value(params["decoder_block2.up_convolution.conv.0.scale"])

    conv115 = ng.conv2d(act114, weight115, strides=(1, 1, 1, 1), bias=bias115, scale=scale115, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift115 = ng.constant([13], dtype=ng.int8)
    act115 = ng.relu(rshift_round_and_clip(conv115, rshift115, dtype=act_dtype))


    # [116] interpolate
    act116 = ng.extern([act113], shape=(1, 8, 12, 1), opcode=0x116, func=interpolate(8, 12, 0, "bilinear"))


    # [117] cat
    rshift117 = ng.constant([2], dtype=ng.int8)
    act117 = ng.concat([act115, act92, ng.rshift_round(act116, rshift117)], axis=3)


    # [118] conv
    weight118 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 257), name="decoder_block2.convolution1.0.weight")
    weight118.set_value(params["decoder_block2.convolution1.0.weight"])

    bias118 = ng.variable(dtype=bias_dtype, shape=(128,), name="decoder_block2.convolution1.0.bias")
    bias118.set_value(np.round(params["decoder_block2.convolution1.0.bias"] / (float) (1 << 3)).astype(params["decoder_block2.convolution1.0.bias"].dtype))

    scale118 = ng.variable(dtype=scale_dtype, shape=(128,), name="decoder_block2.convolution1.0.scale")
    scale118.set_value(params["decoder_block2.convolution1.0.scale"])

    conv118 = ng.conv2d(act117, weight118, strides=(1, 1, 1, 1), bias=bias118, scale=scale118, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift118 = ng.constant([15], dtype=ng.int8)
    act118 = ng.relu(rshift_round_and_clip(conv118, rshift118, dtype=act_dtype))


    # [119] conv
    weight119 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 128), name="decoder_block2.convolution2.0.weight")
    weight119.set_value(params["decoder_block2.convolution2.0.weight"])

    bias119 = ng.variable(dtype=bias_dtype, shape=(128,), name="decoder_block2.convolution2.0.bias")
    bias119.set_value(np.round(params["decoder_block2.convolution2.0.bias"] / (float) (1 << 5)).astype(params["decoder_block2.convolution2.0.bias"].dtype))

    scale119 = ng.variable(dtype=scale_dtype, shape=(128,), name="decoder_block2.convolution2.0.scale")
    scale119.set_value(params["decoder_block2.convolution2.0.scale"])

    conv119 = ng.conv2d(act118, weight119, strides=(1, 1, 1, 1), bias=bias119, scale=scale119, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift119 = ng.constant([11], dtype=ng.int8)
    act119 = ng.relu(rshift_round_and_clip(conv119, rshift119, dtype=act_dtype))


    # [120] conv
    weight120 = ng.variable(dtype=weight_dtype, shape=(1, 3, 3, 128), name="depth_layer_one_eight.0.weight")
    weight120.set_value(params["depth_layer_one_eight.0.weight"])

    bias120 = ng.variable(dtype=bias_dtype, shape=(1,), name="depth_layer_one_eight.0.bias")
    bias120.set_value(np.round(params["depth_layer_one_eight.0.bias"] / (float) (1 << 7)).astype(params["depth_layer_one_eight.0.bias"].dtype))

    conv120 = ng.conv2d(act119, weight120, strides=(1, 1, 1, 1), bias=bias120, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift120 = ng.constant([17], dtype=ng.int8)
    act120 = ng.sigmoid(ng.rshift_round(conv120, rshift120, dtype=act_dtype), lut_addrwidth=9, lut_clip=8.0, range_rate=0.5, dtype=act_dtype)


    # [121] interpolate
    act121 = ng.extern([act119], shape=(1, 16, 24, 128), opcode=0x121, func=interpolate(16, 24, 0, "bilinear"))


    # [122] conv
    weight122 = ng.variable(dtype=weight_dtype, shape=(64, 3, 3, 128), name="decoder_block3.up_convolution.conv.0.weight")
    weight122.set_value(params["decoder_block3.up_convolution.conv.0.weight"])

    bias122 = ng.variable(dtype=bias_dtype, shape=(64,), name="decoder_block3.up_convolution.conv.0.bias")
    bias122.set_value(np.round(params["decoder_block3.up_convolution.conv.0.bias"] / (float) (1 << 4)).astype(params["decoder_block3.up_convolution.conv.0.bias"].dtype))

    scale122 = ng.variable(dtype=scale_dtype, shape=(64,), name="decoder_block3.up_convolution.conv.0.scale")
    scale122.set_value(params["decoder_block3.up_convolution.conv.0.scale"])

    conv122 = ng.conv2d(act121, weight122, strides=(1, 1, 1, 1), bias=bias122, scale=scale122, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift122 = ng.constant([13], dtype=ng.int8)
    act122 = ng.relu(rshift_round_and_clip(conv122, rshift122, dtype=act_dtype))


    # [123] interpolate
    act123 = ng.extern([act120], shape=(1, 16, 24, 1), opcode=0x123, func=interpolate(16, 24, 0, "bilinear"))


    # [124] cat
    rshift124 = ng.constant([3], dtype=ng.int8)
    act124 = ng.concat([act122, act87, ng.rshift_round(act123, rshift124)], axis=3)


    # [125] conv
    weight125 = ng.variable(dtype=weight_dtype, shape=(64, 3, 3, 129), name="decoder_block3.convolution1.0.weight")
    weight125.set_value(params["decoder_block3.convolution1.0.weight"])

    bias125 = ng.variable(dtype=bias_dtype, shape=(64,), name="decoder_block3.convolution1.0.bias")
    bias125.set_value(np.round(params["decoder_block3.convolution1.0.bias"] / (float) (1 << 3)).astype(params["decoder_block3.convolution1.0.bias"].dtype))

    scale125 = ng.variable(dtype=scale_dtype, shape=(64,), name="decoder_block3.convolution1.0.scale")
    scale125.set_value(params["decoder_block3.convolution1.0.scale"])

    conv125 = ng.conv2d(act124, weight125, strides=(1, 1, 1, 1), bias=bias125, scale=scale125, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift125 = ng.constant([14], dtype=ng.int8)
    act125 = ng.relu(rshift_round_and_clip(conv125, rshift125, dtype=act_dtype))


    # [126] conv
    weight126 = ng.variable(dtype=weight_dtype, shape=(64, 3, 3, 64), name="decoder_block3.convolution2.0.weight")
    weight126.set_value(params["decoder_block3.convolution2.0.weight"])

    bias126 = ng.variable(dtype=bias_dtype, shape=(64,), name="decoder_block3.convolution2.0.bias")
    bias126.set_value(np.round(params["decoder_block3.convolution2.0.bias"] / (float) (1 << 4)).astype(params["decoder_block3.convolution2.0.bias"].dtype))

    scale126 = ng.variable(dtype=scale_dtype, shape=(64,), name="decoder_block3.convolution2.0.scale")
    scale126.set_value(params["decoder_block3.convolution2.0.scale"])

    conv126 = ng.conv2d(act125, weight126, strides=(1, 1, 1, 1), bias=bias126, scale=scale126, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift126 = ng.constant([13], dtype=ng.int8)
    act126 = ng.relu(rshift_round_and_clip(conv126, rshift126, dtype=act_dtype))


    # [127] conv
    weight127 = ng.variable(dtype=weight_dtype, shape=(1, 3, 3, 64), name="depth_layer_quarter.0.weight")
    weight127.set_value(params["depth_layer_quarter.0.weight"])

    bias127 = ng.variable(dtype=bias_dtype, shape=(1,), name="depth_layer_quarter.0.bias")
    bias127.set_value(np.round(params["depth_layer_quarter.0.bias"] / (float) (1 << 5)).astype(params["depth_layer_quarter.0.bias"].dtype))

    conv127 = ng.conv2d(act126, weight127, strides=(1, 1, 1, 1), bias=bias127, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift127 = ng.constant([19], dtype=ng.int8)
    act127 = ng.sigmoid(ng.rshift_round(conv127, rshift127, dtype=act_dtype), lut_addrwidth=9, lut_clip=8.0, range_rate=0.5, dtype=act_dtype)


    # [128] interpolate
    act128 = ng.extern([act126], shape=(1, 32, 48, 64), opcode=0x128, func=interpolate(32, 48, 0, "bilinear"))


    # [129] conv
    weight129 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 64), name="decoder_block4.up_convolution.conv.0.weight")
    weight129.set_value(params["decoder_block4.up_convolution.conv.0.weight"])

    bias129 = ng.variable(dtype=bias_dtype, shape=(32,), name="decoder_block4.up_convolution.conv.0.bias")
    bias129.set_value(np.round(params["decoder_block4.up_convolution.conv.0.bias"] / (float) (1 << 2)).astype(params["decoder_block4.up_convolution.conv.0.bias"].dtype))

    scale129 = ng.variable(dtype=scale_dtype, shape=(32,), name="decoder_block4.up_convolution.conv.0.scale")
    scale129.set_value(params["decoder_block4.up_convolution.conv.0.scale"])

    conv129 = ng.conv2d(act128, weight129, strides=(1, 1, 1, 1), bias=bias129, scale=scale129, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift129 = ng.constant([15], dtype=ng.int8)
    act129 = ng.relu(rshift_round_and_clip(conv129, rshift129, dtype=act_dtype))


    # [130] interpolate
    act130 = ng.extern([act127], shape=(1, 32, 48, 1), opcode=0x130, func=interpolate(32, 48, 0, "bilinear"))


    # [131] cat
    rshift131 = ng.constant([3], dtype=ng.int8)
    act131 = ng.concat([act129, act82, ng.rshift_round(act130, rshift131)], axis=3)


    # [132] conv
    weight132 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 65), name="decoder_block4.convolution1.0.weight")
    weight132.set_value(params["decoder_block4.convolution1.0.weight"])

    bias132 = ng.variable(dtype=bias_dtype, shape=(32,), name="decoder_block4.convolution1.0.bias")
    bias132.set_value(np.round(params["decoder_block4.convolution1.0.bias"] / (float) (1 << 3)).astype(params["decoder_block4.convolution1.0.bias"].dtype))

    scale132 = ng.variable(dtype=scale_dtype, shape=(32,), name="decoder_block4.convolution1.0.scale")
    scale132.set_value(params["decoder_block4.convolution1.0.scale"])

    conv132 = ng.conv2d(act131, weight132, strides=(1, 1, 1, 1), bias=bias132, scale=scale132, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift132 = ng.constant([14], dtype=ng.int8)
    act132 = ng.relu(rshift_round_and_clip(conv132, rshift132, dtype=act_dtype))


    # [133] conv
    weight133 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 32), name="decoder_block4.convolution2.0.weight")
    weight133.set_value(params["decoder_block4.convolution2.0.weight"])

    bias133 = ng.variable(dtype=bias_dtype, shape=(32,), name="decoder_block4.convolution2.0.bias")
    bias133.set_value(np.round(params["decoder_block4.convolution2.0.bias"] / (float) (1 << 4)).astype(params["decoder_block4.convolution2.0.bias"].dtype))

    scale133 = ng.variable(dtype=scale_dtype, shape=(32,), name="decoder_block4.convolution2.0.scale")
    scale133.set_value(params["decoder_block4.convolution2.0.scale"])

    conv133 = ng.conv2d(act132, weight133, strides=(1, 1, 1, 1), bias=bias133, scale=scale133, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift133 = ng.constant([13], dtype=ng.int8)
    act133 = ng.relu(rshift_round_and_clip(conv133, rshift133, dtype=act_dtype))


    # [134] conv
    weight134 = ng.variable(dtype=weight_dtype, shape=(1, 3, 3, 32), name="depth_layer_half.0.weight")
    weight134.set_value(params["depth_layer_half.0.weight"])

    bias134 = ng.variable(dtype=bias_dtype, shape=(1,), name="depth_layer_half.0.bias")
    bias134.set_value(np.round(params["depth_layer_half.0.bias"] / (float) (1 << 7)).astype(params["depth_layer_half.0.bias"].dtype))

    conv134 = ng.conv2d(act133, weight134, strides=(1, 1, 1, 1), bias=bias134, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift134 = ng.constant([18], dtype=ng.int8)
    act134 = ng.sigmoid(ng.rshift_round(conv134, rshift134, dtype=act_dtype), lut_addrwidth=9, lut_clip=8.0, range_rate=0.5, dtype=act_dtype)


    # [135] interpolate
    act135 = ng.extern([act134], shape=(1, 64, 96, 1), opcode=0x135, func=interpolate(64, 96, 0, "bilinear"))


    # [136] interpolate
    act136 = ng.extern([act133], shape=(1, 64, 96, 32), opcode=0x136, func=interpolate(64, 96, 0, "bilinear"))


    # [137] cat
    rshift137s = [ng.constant([2], dtype=ng.int8), ng.constant([4], dtype=ng.int8)]
    act137 = ng.concat([ng.rshift_round(act136, rshift137s[0]), ng.rshift_round(act135, rshift137s[1]), act0], axis=3)


    # [138] conv
    weight138 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 36), name="refine.0.0.weight")
    weight138.set_value(params["refine.0.0.weight"])

    bias138 = ng.variable(dtype=bias_dtype, shape=(32,), name="refine.0.0.bias")
    bias138.set_value(np.round(params["refine.0.0.bias"] / (float) (1 << 5)).astype(params["refine.0.0.bias"].dtype))

    scale138 = ng.variable(dtype=scale_dtype, shape=(32,), name="refine.0.0.scale")
    scale138.set_value(params["refine.0.0.scale"])

    conv138 = ng.conv2d(act137, weight138, strides=(1, 1, 1, 1), bias=bias138, scale=scale138, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift138 = ng.constant([12], dtype=ng.int8)
    act138 = ng.relu(rshift_round_and_clip(conv138, rshift138, dtype=act_dtype))


    # [139] conv
    weight139 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 32), name="refine.1.0.weight")
    weight139.set_value(params["refine.1.0.weight"])

    bias139 = ng.variable(dtype=bias_dtype, shape=(32,), name="refine.1.0.bias")
    bias139.set_value(np.round(params["refine.1.0.bias"] / (float) (1 << 3)).astype(params["refine.1.0.bias"].dtype))

    scale139 = ng.variable(dtype=scale_dtype, shape=(32,), name="refine.1.0.scale")
    scale139.set_value(params["refine.1.0.scale"])

    conv139 = ng.conv2d(act138, weight139, strides=(1, 1, 1, 1), bias=bias139, scale=scale139, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift139 = ng.constant([13], dtype=ng.int8)
    act139 = ng.relu(rshift_round_and_clip(conv139, rshift139, dtype=act_dtype))


    # [140] conv
    weight140 = ng.variable(dtype=weight_dtype, shape=(1, 3, 3, 32), name="depth_layer_full.0.weight")
    weight140.set_value(params["depth_layer_full.0.weight"])

    bias140 = ng.variable(dtype=bias_dtype, shape=(1,), name="depth_layer_full.0.bias")
    bias140.set_value(np.round(params["depth_layer_full.0.bias"] / (float) (1 << 6)).astype(params["depth_layer_full.0.bias"].dtype))

    conv140 = ng.conv2d(act139, weight140, strides=(1, 1, 1, 1), bias=bias140, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift140 = ng.constant([18], dtype=ng.int8)
    act140 = ng.sigmoid(ng.rshift_round(conv140, rshift140, dtype=act_dtype), lut_addrwidth=9, lut_clip=8.0, range_rate=0.5, dtype=act_dtype)


    return act140
