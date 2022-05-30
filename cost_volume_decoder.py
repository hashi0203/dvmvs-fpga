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
    bias109.set_value(params["decoder_block1.up_convolution.conv.0.bias"])

    scale109 = ng.variable(dtype=scale_dtype, shape=(256,), name="decoder_block1.up_convolution.conv.0.scale")
    scale109.set_value(params["decoder_block1.up_convolution.conv.0.scale"])

    conv109 = ng.multiply(ng.conv2d(act108, weight109, strides=(1, 1, 1, 1), dtype=mid_dtype), scale109)

    lshift109 = ng.constant([9], dtype=ng.int8)
    sum109 = ng.add(conv109, ng.lshift(bias109, lshift109))
    rshift109 = ng.constant([16], dtype=ng.int8)
    act109 = ng.relu(rshift_round_and_clip(sum109, rshift109, dtype=act_dtype))


    # [110] cat
    rshift110 = ng.constant([1], dtype=ng.int8)
    act110 = ng.concat([ng.rshift_round(act109, rshift110), act97], axis=3)


    # [111] conv
    weight111 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 512), name="decoder_block1.convolution1.0.weight")
    weight111.set_value(params["decoder_block1.convolution1.0.weight"])

    bias111 = ng.variable(dtype=bias_dtype, shape=(256,), name="decoder_block1.convolution1.0.bias")
    bias111.set_value(params["decoder_block1.convolution1.0.bias"])

    scale111 = ng.variable(dtype=scale_dtype, shape=(256,), name="decoder_block1.convolution1.0.scale")
    scale111.set_value(params["decoder_block1.convolution1.0.scale"])

    conv111 = ng.multiply(ng.conv2d(act110, weight111, strides=(1, 1, 1, 1), dtype=mid_dtype), scale111)

    lshift111 = ng.constant([5], dtype=ng.int8)
    sum111 = ng.add(conv111, ng.lshift(bias111, lshift111))
    rshift111 = ng.constant([12], dtype=ng.int8)
    act111 = ng.relu(rshift_round_and_clip(sum111, rshift111, dtype=act_dtype))


    # [112] conv
    weight112 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 256), name="decoder_block1.convolution2.0.weight")
    weight112.set_value(params["decoder_block1.convolution2.0.weight"])

    bias112 = ng.variable(dtype=bias_dtype, shape=(256,), name="decoder_block1.convolution2.0.bias")
    bias112.set_value(params["decoder_block1.convolution2.0.bias"])

    scale112 = ng.variable(dtype=scale_dtype, shape=(256,), name="decoder_block1.convolution2.0.scale")
    scale112.set_value(params["decoder_block1.convolution2.0.scale"])

    conv112 = ng.multiply(ng.conv2d(act111, weight112, strides=(1, 1, 1, 1), dtype=mid_dtype), scale112)

    lshift112 = ng.constant([6], dtype=ng.int8)
    sum112 = ng.add(conv112, ng.lshift(bias112, lshift112))
    rshift112 = ng.constant([11], dtype=ng.int8)
    act112 = ng.relu(rshift_round_and_clip(sum112, rshift112, dtype=act_dtype))


    # [113] conv
    weight113 = ng.variable(dtype=weight_dtype, shape=(1, 3, 3, 256), name="depth_layer_one_sixteen.0.weight")
    weight113.set_value(params["depth_layer_one_sixteen.0.weight"])

    bias113 = ng.variable(dtype=bias_dtype, shape=(1,), name="depth_layer_one_sixteen.0.bias")
    bias113.set_value(params["depth_layer_one_sixteen.0.bias"])

    conv113 = ng.conv2d(act112, weight113, strides=(1, 1, 1, 1), dtype=mid_dtype)

    lshift113 = ng.constant([1], dtype=ng.int8)
    sum113 = ng.add(ng.lshift(conv113, lshift113), bias113)
    rshift113 = ng.constant([19], dtype=ng.int8)
    act113 = ng.sigmoid(ng.rshift_round(sum113, rshift113, dtype=act_dtype), lut_addrwidth=9, lut_clip=8.0, range_rate=0.5, dtype=act_dtype)


    # [114] interpolate
    act114 = ng.extern([act112], shape=(1, 8, 12, 256), opcode=0x114, func=interpolate(8, 12, 0, "bilinear"))


    # [115] conv
    weight115 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 256), name="decoder_block2.up_convolution.conv.0.weight")
    weight115.set_value(params["decoder_block2.up_convolution.conv.0.weight"])

    bias115 = ng.variable(dtype=bias_dtype, shape=(128,), name="decoder_block2.up_convolution.conv.0.bias")
    bias115.set_value(params["decoder_block2.up_convolution.conv.0.bias"])

    scale115 = ng.variable(dtype=scale_dtype, shape=(128,), name="decoder_block2.up_convolution.conv.0.scale")
    scale115.set_value(params["decoder_block2.up_convolution.conv.0.scale"])

    conv115 = ng.multiply(ng.conv2d(act114, weight115, strides=(1, 1, 1, 1), dtype=mid_dtype), scale115)

    lshift115 = ng.constant([7], dtype=ng.int8)
    sum115 = ng.add(conv115, ng.lshift(bias115, lshift115))
    rshift115 = ng.constant([13], dtype=ng.int8)
    act115 = ng.relu(rshift_round_and_clip(sum115, rshift115, dtype=act_dtype))


    # [116] interpolate
    act116 = ng.extern([act113], shape=(1, 8, 12, 1), opcode=0x116, func=interpolate(8, 12, 0, "bilinear"))


    # [117] cat
    rshift117 = ng.constant([2], dtype=ng.int8)
    act117 = ng.concat([act115, act92, ng.rshift_round(act116, rshift117)], axis=3)


    # [118] conv
    weight118 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 257), name="decoder_block2.convolution1.0.weight")
    weight118.set_value(params["decoder_block2.convolution1.0.weight"])

    bias118 = ng.variable(dtype=bias_dtype, shape=(128,), name="decoder_block2.convolution1.0.bias")
    bias118.set_value(params["decoder_block2.convolution1.0.bias"])

    scale118 = ng.variable(dtype=scale_dtype, shape=(128,), name="decoder_block2.convolution1.0.scale")
    scale118.set_value(params["decoder_block2.convolution1.0.scale"])

    conv118 = ng.multiply(ng.conv2d(act117, weight118, strides=(1, 1, 1, 1), dtype=mid_dtype), scale118)

    lshift118 = ng.constant([9], dtype=ng.int8)
    sum118 = ng.add(conv118, ng.lshift(bias118, lshift118))
    rshift118 = ng.constant([15], dtype=ng.int8)
    act118 = ng.relu(rshift_round_and_clip(sum118, rshift118, dtype=act_dtype))


    # [119] conv
    weight119 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 128), name="decoder_block2.convolution2.0.weight")
    weight119.set_value(params["decoder_block2.convolution2.0.weight"])

    bias119 = ng.variable(dtype=bias_dtype, shape=(128,), name="decoder_block2.convolution2.0.bias")
    bias119.set_value(params["decoder_block2.convolution2.0.bias"])

    scale119 = ng.variable(dtype=scale_dtype, shape=(128,), name="decoder_block2.convolution2.0.scale")
    scale119.set_value(params["decoder_block2.convolution2.0.scale"])

    conv119 = ng.multiply(ng.conv2d(act118, weight119, strides=(1, 1, 1, 1), dtype=mid_dtype), scale119)

    lshift119 = ng.constant([5], dtype=ng.int8)
    sum119 = ng.add(conv119, ng.lshift(bias119, lshift119))
    rshift119 = ng.constant([11], dtype=ng.int8)
    act119 = ng.relu(rshift_round_and_clip(sum119, rshift119, dtype=act_dtype))


    # [120] conv
    weight120 = ng.variable(dtype=weight_dtype, shape=(1, 3, 3, 128), name="depth_layer_one_eight.0.weight")
    weight120.set_value(params["depth_layer_one_eight.0.weight"])

    bias120 = ng.variable(dtype=bias_dtype, shape=(1,), name="depth_layer_one_eight.0.bias")
    bias120.set_value(params["depth_layer_one_eight.0.bias"])

    conv120 = ng.conv2d(act119, weight120, strides=(1, 1, 1, 1), dtype=mid_dtype)

    lshift120 = ng.constant([1], dtype=ng.int8)
    sum120 = ng.add(ng.lshift(conv120, lshift120), bias120)
    rshift120 = ng.constant([18], dtype=ng.int8)
    act120 = ng.sigmoid(ng.rshift_round(sum120, rshift120, dtype=act_dtype), lut_addrwidth=9, lut_clip=8.0, range_rate=0.5, dtype=act_dtype)


    # [121] interpolate
    act121 = ng.extern([act119], shape=(1, 16, 24, 128), opcode=0x121, func=interpolate(16, 24, 0, "bilinear"))


    # [122] conv
    weight122 = ng.variable(dtype=weight_dtype, shape=(64, 3, 3, 128), name="decoder_block3.up_convolution.conv.0.weight")
    weight122.set_value(params["decoder_block3.up_convolution.conv.0.weight"])

    bias122 = ng.variable(dtype=bias_dtype, shape=(64,), name="decoder_block3.up_convolution.conv.0.bias")
    bias122.set_value(params["decoder_block3.up_convolution.conv.0.bias"])

    scale122 = ng.variable(dtype=scale_dtype, shape=(64,), name="decoder_block3.up_convolution.conv.0.scale")
    scale122.set_value(params["decoder_block3.up_convolution.conv.0.scale"])

    conv122 = ng.multiply(ng.conv2d(act121, weight122, strides=(1, 1, 1, 1), dtype=mid_dtype), scale122)

    lshift122 = ng.constant([6], dtype=ng.int8)
    sum122 = ng.add(conv122, ng.lshift(bias122, lshift122))
    rshift122 = ng.constant([13], dtype=ng.int8)
    act122 = ng.relu(rshift_round_and_clip(sum122, rshift122, dtype=act_dtype))


    # [123] interpolate
    act123 = ng.extern([act120], shape=(1, 16, 24, 1), opcode=0x123, func=interpolate(16, 24, 0, "bilinear"))


    # [124] cat
    rshift124 = ng.constant([3], dtype=ng.int8)
    act124 = ng.concat([act122, act87, ng.rshift_round(act123, rshift124)], axis=3)


    # [125] conv
    weight125 = ng.variable(dtype=weight_dtype, shape=(64, 3, 3, 129), name="decoder_block3.convolution1.0.weight")
    weight125.set_value(params["decoder_block3.convolution1.0.weight"])

    bias125 = ng.variable(dtype=bias_dtype, shape=(64,), name="decoder_block3.convolution1.0.bias")
    bias125.set_value(params["decoder_block3.convolution1.0.bias"])

    scale125 = ng.variable(dtype=scale_dtype, shape=(64,), name="decoder_block3.convolution1.0.scale")
    scale125.set_value(params["decoder_block3.convolution1.0.scale"])

    conv125 = ng.multiply(ng.conv2d(act124, weight125, strides=(1, 1, 1, 1), dtype=mid_dtype), scale125)

    lshift125 = ng.constant([7], dtype=ng.int8)
    sum125 = ng.add(conv125, ng.lshift(bias125, lshift125))
    rshift125 = ng.constant([14], dtype=ng.int8)
    act125 = ng.relu(rshift_round_and_clip(sum125, rshift125, dtype=act_dtype))


    # [126] conv
    weight126 = ng.variable(dtype=weight_dtype, shape=(64, 3, 3, 64), name="decoder_block3.convolution2.0.weight")
    weight126.set_value(params["decoder_block3.convolution2.0.weight"])

    bias126 = ng.variable(dtype=bias_dtype, shape=(64,), name="decoder_block3.convolution2.0.bias")
    bias126.set_value(params["decoder_block3.convolution2.0.bias"])

    scale126 = ng.variable(dtype=scale_dtype, shape=(64,), name="decoder_block3.convolution2.0.scale")
    scale126.set_value(params["decoder_block3.convolution2.0.scale"])

    conv126 = ng.multiply(ng.conv2d(act125, weight126, strides=(1, 1, 1, 1), dtype=mid_dtype), scale126)

    lshift126 = ng.constant([7], dtype=ng.int8)
    sum126 = ng.add(conv126, ng.lshift(bias126, lshift126))
    rshift126 = ng.constant([13], dtype=ng.int8)
    act126 = ng.relu(rshift_round_and_clip(sum126, rshift126, dtype=act_dtype))


    # [127] conv
    weight127 = ng.variable(dtype=weight_dtype, shape=(1, 3, 3, 64), name="depth_layer_quarter.0.weight")
    weight127.set_value(params["depth_layer_quarter.0.weight"])

    bias127 = ng.variable(dtype=bias_dtype, shape=(1,), name="depth_layer_quarter.0.bias")
    bias127.set_value(params["depth_layer_quarter.0.bias"])

    conv127 = ng.conv2d(act126, weight127, strides=(1, 1, 1, 1), dtype=mid_dtype)

    lshift127 = ng.constant([1], dtype=ng.int8)
    sum127 = ng.add(conv127, ng.lshift(bias127, lshift127))
    rshift127 = ng.constant([19], dtype=ng.int8)
    act127 = ng.sigmoid(ng.rshift_round(sum127, rshift127, dtype=act_dtype), lut_addrwidth=9, lut_clip=8.0, range_rate=0.5, dtype=act_dtype)


    # [128] interpolate
    act128 = ng.extern([act126], shape=(1, 32, 48, 64), opcode=0x128, func=interpolate(32, 48, 0, "bilinear"))


    # [129] conv
    weight129 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 64), name="decoder_block4.up_convolution.conv.0.weight")
    weight129.set_value(params["decoder_block4.up_convolution.conv.0.weight"])

    bias129 = ng.variable(dtype=bias_dtype, shape=(32,), name="decoder_block4.up_convolution.conv.0.bias")
    bias129.set_value(params["decoder_block4.up_convolution.conv.0.bias"])

    scale129 = ng.variable(dtype=scale_dtype, shape=(32,), name="decoder_block4.up_convolution.conv.0.scale")
    scale129.set_value(params["decoder_block4.up_convolution.conv.0.scale"])

    conv129 = ng.multiply(ng.conv2d(act128, weight129, strides=(1, 1, 1, 1), dtype=mid_dtype), scale129)

    lshift129 = ng.constant([9], dtype=ng.int8)
    sum129 = ng.add(conv129, ng.lshift(bias129, lshift129))
    rshift129 = ng.constant([15], dtype=ng.int8)
    act129 = ng.relu(rshift_round_and_clip(sum129, rshift129, dtype=act_dtype))


    # [130] interpolate
    act130 = ng.extern([act127], shape=(1, 32, 48, 1), opcode=0x130, func=interpolate(32, 48, 0, "bilinear"))


    # [131] cat
    rshift131 = ng.constant([3], dtype=ng.int8)
    act131 = ng.concat([act129, act82, ng.rshift_round(act130, rshift131)], axis=3)


    # [132] conv
    weight132 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 65), name="decoder_block4.convolution1.0.weight")
    weight132.set_value(params["decoder_block4.convolution1.0.weight"])

    bias132 = ng.variable(dtype=bias_dtype, shape=(32,), name="decoder_block4.convolution1.0.bias")
    bias132.set_value(params["decoder_block4.convolution1.0.bias"])

    scale132 = ng.variable(dtype=scale_dtype, shape=(32,), name="decoder_block4.convolution1.0.scale")
    scale132.set_value(params["decoder_block4.convolution1.0.scale"])

    conv132 = ng.multiply(ng.conv2d(act131, weight132, strides=(1, 1, 1, 1), dtype=mid_dtype), scale132)

    lshift132 = ng.constant([7], dtype=ng.int8)
    sum132 = ng.add(conv132, ng.lshift(bias132, lshift132))
    rshift132 = ng.constant([14], dtype=ng.int8)
    act132 = ng.relu(rshift_round_and_clip(sum132, rshift132, dtype=act_dtype))


    # [133] conv
    weight133 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 32), name="decoder_block4.convolution2.0.weight")
    weight133.set_value(params["decoder_block4.convolution2.0.weight"])

    bias133 = ng.variable(dtype=bias_dtype, shape=(32,), name="decoder_block4.convolution2.0.bias")
    bias133.set_value(params["decoder_block4.convolution2.0.bias"])

    scale133 = ng.variable(dtype=scale_dtype, shape=(32,), name="decoder_block4.convolution2.0.scale")
    scale133.set_value(params["decoder_block4.convolution2.0.scale"])

    conv133 = ng.multiply(ng.conv2d(act132, weight133, strides=(1, 1, 1, 1), dtype=mid_dtype), scale133)

    lshift133 = ng.constant([6], dtype=ng.int8)
    sum133 = ng.add(conv133, ng.lshift(bias133, lshift133))
    rshift133 = ng.constant([13], dtype=ng.int8)
    act133 = ng.relu(rshift_round_and_clip(sum133, rshift133, dtype=act_dtype))


    # [134] conv
    weight134 = ng.variable(dtype=weight_dtype, shape=(1, 3, 3, 32), name="depth_layer_half.0.weight")
    weight134.set_value(params["depth_layer_half.0.weight"])

    bias134 = ng.variable(dtype=bias_dtype, shape=(1,), name="depth_layer_half.0.bias")
    bias134.set_value(params["depth_layer_half.0.bias"])

    conv134 = ng.conv2d(act133, weight134, strides=(1, 1, 1, 1), dtype=mid_dtype)

    lshift134 = ng.constant([1], dtype=ng.int8)
    sum134 = ng.add(ng.lshift(conv134, lshift134), bias134)
    rshift134 = ng.constant([19], dtype=ng.int8)
    act134 = ng.sigmoid(ng.rshift_round(sum134, rshift134, dtype=act_dtype), lut_addrwidth=9, lut_clip=8.0, range_rate=0.5, dtype=act_dtype)


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
    bias138.set_value(params["refine.0.0.bias"])

    scale138 = ng.variable(dtype=scale_dtype, shape=(32,), name="refine.0.0.scale")
    scale138.set_value(params["refine.0.0.scale"])

    conv138 = ng.multiply(ng.conv2d(act137, weight138, strides=(1, 1, 1, 1), dtype=mid_dtype), scale138)

    lshift138 = ng.constant([6], dtype=ng.int8)
    sum138 = ng.add(conv138, ng.lshift(bias138, lshift138))
    rshift138 = ng.constant([12], dtype=ng.int8)
    act138 = ng.relu(rshift_round_and_clip(sum138, rshift138, dtype=act_dtype))


    # [139] conv
    weight139 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 32), name="refine.1.0.weight")
    weight139.set_value(params["refine.1.0.weight"])

    bias139 = ng.variable(dtype=bias_dtype, shape=(32,), name="refine.1.0.bias")
    bias139.set_value(params["refine.1.0.bias"])

    scale139 = ng.variable(dtype=scale_dtype, shape=(32,), name="refine.1.0.scale")
    scale139.set_value(params["refine.1.0.scale"])

    conv139 = ng.multiply(ng.conv2d(act138, weight139, strides=(1, 1, 1, 1), dtype=mid_dtype), scale139)

    lshift139 = ng.constant([6], dtype=ng.int8)
    sum139 = ng.add(conv139, ng.lshift(bias139, lshift139))
    rshift139 = ng.constant([13], dtype=ng.int8)
    act139 = ng.relu(rshift_round_and_clip(sum139, rshift139, dtype=act_dtype))


    # [140] conv
    weight140 = ng.variable(dtype=weight_dtype, shape=(1, 3, 3, 32), name="depth_layer_full.0.weight")
    weight140.set_value(params["depth_layer_full.0.weight"])

    bias140 = ng.variable(dtype=bias_dtype, shape=(1,), name="depth_layer_full.0.bias")
    bias140.set_value(params["depth_layer_full.0.bias"])

    conv140 = ng.conv2d(act139, weight140, strides=(1, 1, 1, 1), dtype=mid_dtype)

    sum140 = ng.add(conv140, bias140)
    rshift140 = ng.constant([18], dtype=ng.int8)
    act140 = ng.sigmoid(ng.rshift_round(sum140, rshift140, dtype=act_dtype), lut_addrwidth=9, lut_clip=8.0, range_rate=0.5, dtype=act_dtype)


    return act140

