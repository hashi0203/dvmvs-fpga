import nngen as ng

def cost_volume_encoder(act78, act74, act70, act66, act80, params,
                        weight_dtype=ng.int8, bias_dtype=ng.int32, scale_dtype=ng.int8, act_dtype=ng.int32):

    # [81] cat
    rshift81 = ng.constant([2], dtype=ng.int8)
    act81 = ng.concat([ng.rshift_round(act78, rshift81), act80], axis=3)


    # [82] conv
    weight82 = ng.variable(dtype=weight_dtype, shape=(32, 5, 5, 96), name="aggregator0.0.weight")
    weight82.set_value(params["aggregator0.0.weight"])

    bias82 = ng.variable(dtype=bias_dtype, shape=(32,), name="aggregator0.0.bias")
    bias82.set_value(params["aggregator0.0.bias"])

    scale82 = ng.variable(dtype=scale_dtype, shape=(32,), name="aggregator0.0.scale")
    scale82.set_value(params["aggregator0.0.scale"])

    conv82 = ng.multiply(ng.conv2d(act81, weight82, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale82)

    lshift82 = ng.constant([5], dtype=ng.int8)
    sum82 = ng.add(conv82, ng.lshift(bias82, lshift82))
    rshift82 = ng.constant([13], dtype=ng.int8)
    act82 = ng.relu(ng.rshift_round(sum82, rshift82))


    # [83] conv
    weight83 = ng.variable(dtype=weight_dtype, shape=(64, 5, 5, 32), name="encoder_block0.down_convolution.down_conv.0.weight")
    weight83.set_value(params["encoder_block0.down_convolution.down_conv.0.weight"])

    bias83 = ng.variable(dtype=bias_dtype, shape=(64,), name="encoder_block0.down_convolution.down_conv.0.bias")
    bias83.set_value(params["encoder_block0.down_convolution.down_conv.0.bias"])

    scale83 = ng.variable(dtype=scale_dtype, shape=(64,), name="encoder_block0.down_convolution.down_conv.0.scale")
    scale83.set_value(params["encoder_block0.down_convolution.down_conv.0.scale"])

    conv83 = ng.multiply(ng.conv2d(act82, weight83, strides=(1, 2, 2, 1), dtype=act_dtype, sum_dtype=ng.int32), scale83)

    lshift83 = ng.constant([8], dtype=ng.int8)
    sum83 = ng.add(conv83, ng.lshift(bias83, lshift83))
    rshift83 = ng.constant([14], dtype=ng.int8)
    act83 = ng.relu(ng.rshift_round(sum83, rshift83))


    # [84] conv
    weight84 = ng.variable(dtype=weight_dtype, shape=(64, 5, 5, 64), name="encoder_block0.standard_convolution.conv1.0.weight")
    weight84.set_value(params["encoder_block0.standard_convolution.conv1.0.weight"])

    bias84 = ng.variable(dtype=bias_dtype, shape=(64,), name="encoder_block0.standard_convolution.conv1.0.bias")
    bias84.set_value(params["encoder_block0.standard_convolution.conv1.0.bias"])

    scale84 = ng.variable(dtype=scale_dtype, shape=(64,), name="encoder_block0.standard_convolution.conv1.0.scale")
    scale84.set_value(params["encoder_block0.standard_convolution.conv1.0.scale"])

    conv84 = ng.multiply(ng.conv2d(act83, weight84, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale84)

    lshift84 = ng.constant([9], dtype=ng.int8)
    sum84 = ng.add(conv84, ng.lshift(bias84, lshift84))
    rshift84 = ng.constant([15], dtype=ng.int8)
    act84 = ng.relu(ng.rshift_round(sum84, rshift84))


    # [85] conv
    weight85 = ng.variable(dtype=weight_dtype, shape=(64, 5, 5, 64), name="encoder_block0.standard_convolution.conv2.0.weight")
    weight85.set_value(params["encoder_block0.standard_convolution.conv2.0.weight"])

    bias85 = ng.variable(dtype=bias_dtype, shape=(64,), name="encoder_block0.standard_convolution.conv2.0.bias")
    bias85.set_value(params["encoder_block0.standard_convolution.conv2.0.bias"])

    scale85 = ng.variable(dtype=scale_dtype, shape=(64,), name="encoder_block0.standard_convolution.conv2.0.scale")
    scale85.set_value(params["encoder_block0.standard_convolution.conv2.0.scale"])

    conv85 = ng.multiply(ng.conv2d(act84, weight85, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale85)

    lshift85 = ng.constant([8], dtype=ng.int8)
    sum85 = ng.add(conv85, ng.lshift(bias85, lshift85))
    rshift85 = ng.constant([15], dtype=ng.int8)
    act85 = ng.relu(ng.rshift_round(sum85, rshift85))


    # [86] cat
    rshift86 = ng.constant([3], dtype=ng.int8)
    act86 = ng.concat([act74, ng.rshift_round(act85, rshift86)], axis=3)


    # [87] conv
    weight87 = ng.variable(dtype=weight_dtype, shape=(64, 3, 3, 96), name="aggregator1.0.weight")
    weight87.set_value(params["aggregator1.0.weight"])

    bias87 = ng.variable(dtype=bias_dtype, shape=(64,), name="aggregator1.0.bias")
    bias87.set_value(params["aggregator1.0.bias"])

    scale87 = ng.variable(dtype=scale_dtype, shape=(64,), name="aggregator1.0.scale")
    scale87.set_value(params["aggregator1.0.scale"])

    conv87 = ng.multiply(ng.conv2d(act86, weight87, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale87)

    lshift87 = ng.constant([6], dtype=ng.int8)
    sum87 = ng.add(conv87, ng.lshift(bias87, lshift87))
    rshift87 = ng.constant([13], dtype=ng.int8)
    act87 = ng.relu(ng.rshift_round(sum87, rshift87))


    # [88] conv
    weight88 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 64), name="encoder_block1.down_convolution.down_conv.0.weight")
    weight88.set_value(params["encoder_block1.down_convolution.down_conv.0.weight"])

    bias88 = ng.variable(dtype=bias_dtype, shape=(128,), name="encoder_block1.down_convolution.down_conv.0.bias")
    bias88.set_value(params["encoder_block1.down_convolution.down_conv.0.bias"])

    scale88 = ng.variable(dtype=scale_dtype, shape=(128,), name="encoder_block1.down_convolution.down_conv.0.scale")
    scale88.set_value(params["encoder_block1.down_convolution.down_conv.0.scale"])

    conv88 = ng.multiply(ng.conv2d(act87, weight88, strides=(1, 2, 2, 1), dtype=act_dtype, sum_dtype=ng.int32), scale88)

    lshift88 = ng.constant([6], dtype=ng.int8)
    sum88 = ng.add(conv88, ng.lshift(bias88, lshift88))
    rshift88 = ng.constant([13], dtype=ng.int8)
    act88 = ng.relu(ng.rshift_round(sum88, rshift88))


    # [89] conv
    weight89 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 128), name="encoder_block1.standard_convolution.conv1.0.weight")
    weight89.set_value(params["encoder_block1.standard_convolution.conv1.0.weight"])

    bias89 = ng.variable(dtype=bias_dtype, shape=(128,), name="encoder_block1.standard_convolution.conv1.0.bias")
    bias89.set_value(params["encoder_block1.standard_convolution.conv1.0.bias"])

    scale89 = ng.variable(dtype=scale_dtype, shape=(128,), name="encoder_block1.standard_convolution.conv1.0.scale")
    scale89.set_value(params["encoder_block1.standard_convolution.conv1.0.scale"])

    conv89 = ng.multiply(ng.conv2d(act88, weight89, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale89)

    lshift89 = ng.constant([7], dtype=ng.int8)
    sum89 = ng.add(conv89, ng.lshift(bias89, lshift89))
    rshift89 = ng.constant([14], dtype=ng.int8)
    act89 = ng.relu(ng.rshift_round(sum89, rshift89))


    # [90] conv
    weight90 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 128), name="encoder_block1.standard_convolution.conv2.0.weight")
    weight90.set_value(params["encoder_block1.standard_convolution.conv2.0.weight"])

    bias90 = ng.variable(dtype=bias_dtype, shape=(128,), name="encoder_block1.standard_convolution.conv2.0.bias")
    bias90.set_value(params["encoder_block1.standard_convolution.conv2.0.bias"])

    scale90 = ng.variable(dtype=scale_dtype, shape=(128,), name="encoder_block1.standard_convolution.conv2.0.scale")
    scale90.set_value(params["encoder_block1.standard_convolution.conv2.0.scale"])

    conv90 = ng.multiply(ng.conv2d(act89, weight90, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale90)

    lshift90 = ng.constant([7], dtype=ng.int8)
    sum90 = ng.add(conv90, ng.lshift(bias90, lshift90))
    rshift90 = ng.constant([15], dtype=ng.int8)
    act90 = ng.relu(ng.rshift_round(sum90, rshift90))


    # [91] cat
    rshift91 = ng.constant([1], dtype=ng.int8)
    act91 = ng.concat([act70, ng.rshift_round(act90, rshift91)], axis=3)


    # [92] conv
    weight92 = ng.variable(dtype=weight_dtype, shape=(128, 3, 3, 160), name="aggregator2.0.weight")
    weight92.set_value(params["aggregator2.0.weight"])

    bias92 = ng.variable(dtype=bias_dtype, shape=(128,), name="aggregator2.0.bias")
    bias92.set_value(params["aggregator2.0.bias"])

    scale92 = ng.variable(dtype=scale_dtype, shape=(128,), name="aggregator2.0.scale")
    scale92.set_value(params["aggregator2.0.scale"])

    conv92 = ng.multiply(ng.conv2d(act91, weight92, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale92)

    lshift92 = ng.constant([7], dtype=ng.int8)
    sum92 = ng.add(conv92, ng.lshift(bias92, lshift92))
    rshift92 = ng.constant([14], dtype=ng.int8)
    act92 = ng.relu(ng.rshift_round(sum92, rshift92))


    # [93] conv
    weight93 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 128), name="encoder_block2.down_convolution.down_conv.0.weight")
    weight93.set_value(params["encoder_block2.down_convolution.down_conv.0.weight"])

    bias93 = ng.variable(dtype=bias_dtype, shape=(256,), name="encoder_block2.down_convolution.down_conv.0.bias")
    bias93.set_value(params["encoder_block2.down_convolution.down_conv.0.bias"])

    scale93 = ng.variable(dtype=scale_dtype, shape=(256,), name="encoder_block2.down_convolution.down_conv.0.scale")
    scale93.set_value(params["encoder_block2.down_convolution.down_conv.0.scale"])

    conv93 = ng.multiply(ng.conv2d(act92, weight93, strides=(1, 2, 2, 1), dtype=act_dtype, sum_dtype=ng.int32), scale93)

    lshift93 = ng.constant([7], dtype=ng.int8)
    sum93 = ng.add(conv93, ng.lshift(bias93, lshift93))
    rshift93 = ng.constant([15], dtype=ng.int8)
    act93 = ng.relu(ng.rshift_round(sum93, rshift93))


    # [94] conv
    weight94 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 256), name="encoder_block2.standard_convolution.conv1.0.weight")
    weight94.set_value(params["encoder_block2.standard_convolution.conv1.0.weight"])

    bias94 = ng.variable(dtype=bias_dtype, shape=(256,), name="encoder_block2.standard_convolution.conv1.0.bias")
    bias94.set_value(params["encoder_block2.standard_convolution.conv1.0.bias"])

    scale94 = ng.variable(dtype=scale_dtype, shape=(256,), name="encoder_block2.standard_convolution.conv1.0.scale")
    scale94.set_value(params["encoder_block2.standard_convolution.conv1.0.scale"])

    conv94 = ng.multiply(ng.conv2d(act93, weight94, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale94)

    lshift94 = ng.constant([7], dtype=ng.int8)
    sum94 = ng.add(conv94, ng.lshift(bias94, lshift94))
    rshift94 = ng.constant([14], dtype=ng.int8)
    act94 = ng.relu(ng.rshift_round(sum94, rshift94))


    # [95] conv
    weight95 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 256), name="encoder_block2.standard_convolution.conv2.0.weight")
    weight95.set_value(params["encoder_block2.standard_convolution.conv2.0.weight"])

    bias95 = ng.variable(dtype=bias_dtype, shape=(256,), name="encoder_block2.standard_convolution.conv2.0.bias")
    bias95.set_value(params["encoder_block2.standard_convolution.conv2.0.bias"])

    scale95 = ng.variable(dtype=scale_dtype, shape=(256,), name="encoder_block2.standard_convolution.conv2.0.scale")
    scale95.set_value(params["encoder_block2.standard_convolution.conv2.0.scale"])

    conv95 = ng.multiply(ng.conv2d(act94, weight95, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale95)

    lshift95 = ng.constant([7], dtype=ng.int8)
    sum95 = ng.add(conv95, ng.lshift(bias95, lshift95))
    rshift95 = ng.constant([15], dtype=ng.int8)
    act95 = ng.relu(ng.rshift_round(sum95, rshift95))


    # [96] cat
    rshift96 = ng.constant([1], dtype=ng.int8)
    act96 = ng.concat([act66, ng.rshift_round(act95, rshift96)], axis=3)


    # [97] conv
    weight97 = ng.variable(dtype=weight_dtype, shape=(256, 3, 3, 288), name="aggregator3.0.weight")
    weight97.set_value(params["aggregator3.0.weight"])

    bias97 = ng.variable(dtype=bias_dtype, shape=(256,), name="aggregator3.0.bias")
    bias97.set_value(params["aggregator3.0.bias"])

    scale97 = ng.variable(dtype=scale_dtype, shape=(256,), name="aggregator3.0.scale")
    scale97.set_value(params["aggregator3.0.scale"])

    conv97 = ng.multiply(ng.conv2d(act96, weight97, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale97)

    lshift97 = ng.constant([5], dtype=ng.int8)
    sum97 = ng.add(conv97, ng.lshift(bias97, lshift97))
    rshift97 = ng.constant([13], dtype=ng.int8)
    act97 = ng.relu(ng.rshift_round(sum97, rshift97))


    # [98] conv
    weight98 = ng.variable(dtype=weight_dtype, shape=(512, 3, 3, 256), name="encoder_block3.down_convolution.down_conv.0.weight")
    weight98.set_value(params["encoder_block3.down_convolution.down_conv.0.weight"])

    bias98 = ng.variable(dtype=bias_dtype, shape=(512,), name="encoder_block3.down_convolution.down_conv.0.bias")
    bias98.set_value(params["encoder_block3.down_convolution.down_conv.0.bias"])

    scale98 = ng.variable(dtype=scale_dtype, shape=(512,), name="encoder_block3.down_convolution.down_conv.0.scale")
    scale98.set_value(params["encoder_block3.down_convolution.down_conv.0.scale"])

    conv98 = ng.multiply(ng.conv2d(act97, weight98, strides=(1, 2, 2, 1), dtype=act_dtype, sum_dtype=ng.int32), scale98)

    lshift98 = ng.constant([8], dtype=ng.int8)
    sum98 = ng.add(conv98, ng.lshift(bias98, lshift98))
    rshift98 = ng.constant([15], dtype=ng.int8)
    act98 = ng.relu(ng.rshift_round(sum98, rshift98))


    # [99] conv
    weight99 = ng.variable(dtype=weight_dtype, shape=(512, 3, 3, 512), name="encoder_block3.standard_convolution.conv1.0.weight")
    weight99.set_value(params["encoder_block3.standard_convolution.conv1.0.weight"])

    bias99 = ng.variable(dtype=bias_dtype, shape=(512,), name="encoder_block3.standard_convolution.conv1.0.bias")
    bias99.set_value(params["encoder_block3.standard_convolution.conv1.0.bias"])

    scale99 = ng.variable(dtype=scale_dtype, shape=(512,), name="encoder_block3.standard_convolution.conv1.0.scale")
    scale99.set_value(params["encoder_block3.standard_convolution.conv1.0.scale"])

    conv99 = ng.multiply(ng.conv2d(act98, weight99, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale99)

    lshift99 = ng.constant([8], dtype=ng.int8)
    sum99 = ng.add(conv99, ng.lshift(bias99, lshift99))
    rshift99 = ng.constant([14], dtype=ng.int8)
    act99 = ng.relu(ng.rshift_round(sum99, rshift99))


    # [100] conv
    weight100 = ng.variable(dtype=weight_dtype, shape=(512, 3, 3, 512), name="encoder_block3.standard_convolution.conv2.0.weight")
    weight100.set_value(params["encoder_block3.standard_convolution.conv2.0.weight"])

    bias100 = ng.variable(dtype=bias_dtype, shape=(512,), name="encoder_block3.standard_convolution.conv2.0.bias")
    bias100.set_value(params["encoder_block3.standard_convolution.conv2.0.bias"])

    scale100 = ng.variable(dtype=scale_dtype, shape=(512,), name="encoder_block3.standard_convolution.conv2.0.scale")
    scale100.set_value(params["encoder_block3.standard_convolution.conv2.0.scale"])

    conv100 = ng.multiply(ng.conv2d(act99, weight100, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale100)

    lshift100 = ng.constant([9], dtype=ng.int8)
    sum100 = ng.add(conv100, ng.lshift(bias100, lshift100))
    rshift100 = ng.constant([16], dtype=ng.int8)
    act100 = ng.relu(ng.rshift_round(sum100, rshift100))


    return act82, act87, act92, act97, act100

