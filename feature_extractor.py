import numpy as np
import nngen as ng

def feature_extractor(act0, params,
                      weight_dtype=ng.int8, bias_dtype=ng.int32, scale_dtype=ng.int8, act_dtype=ng.int32):


    # [1] conv
    weight1 = ng.variable(dtype=weight_dtype, shape=(32, 3, 3, 3), name="layer1.0.weight")
    weight1.set_value(params["layer1.0.weight"])

    bias1 = ng.variable(dtype=bias_dtype, shape=(32,), name="layer1.0.bias")
    bias1.set_value(params["layer1.0.bias"])

    scale1 = ng.variable(dtype=scale_dtype, shape=(32,), name="layer1.0.scale")
    scale1.set_value(params["layer1.0.scale"])

    conv1 = ng.multiply(ng.conv2d(act0, weight1, strides=(1, 2, 2, 1), dtype=act_dtype, sum_dtype=ng.int32), scale1)

    lshift1 = ng.constant([4], dtype=ng.int8)
    sum1 = ng.add(conv1, ng.lshift(bias1, lshift1))
    rshift1 = ng.constant([12], dtype=ng.int8)
    act1 = ng.relu(ng.rshift_round(sum1, rshift1))


    # [2] conv
    weight2 = ng.variable(dtype=weight_dtype, shape=(32, 3, 3, 32), name="layer1.3.weight")
    weight2.set_value(np.array([[[[params["layer1.3.weight"][i][j][k][0] if i == l else 0 for l in range(32)] for k in range(3)] for j in range(3)] for i in range(32)]))

    bias2 = ng.variable(dtype=bias_dtype, shape=(32,), name="layer1.3.bias")
    bias2.set_value(params["layer1.3.bias"])

    scale2 = ng.variable(dtype=scale_dtype, shape=(32,), name="layer1.3.scale")
    scale2.set_value(params["layer1.3.scale"])

    conv2 = ng.multiply(ng.conv2d(act1, weight2, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale2)

    lshift2 = ng.constant([1], dtype=ng.int8)
    sum2 = ng.add(conv2, ng.lshift(bias2, lshift2))
    rshift2 = ng.constant([7], dtype=ng.int8)
    act2 = ng.relu(ng.rshift_round(sum2, rshift2))


    # [3] conv
    weight3 = ng.variable(dtype=weight_dtype, shape=(16, 1, 1, 32), name="layer1.6.weight")
    weight3.set_value(params["layer1.6.weight"])

    bias3 = ng.variable(dtype=bias_dtype, shape=(16,), name="layer1.6.bias")
    bias3.set_value(params["layer1.6.bias"])

    scale3 = ng.variable(dtype=scale_dtype, shape=(16,), name="layer1.6.scale")
    scale3.set_value(params["layer1.6.scale"])

    conv3 = ng.multiply(ng.conv2d(act2, weight3, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale3)

    lshift3 = ng.constant([6], dtype=ng.int8)
    sum3 = ng.add(conv3, ng.lshift(bias3, lshift3))
    rshift3 = ng.constant([13], dtype=ng.int8)
    act3 = ng.rshift_round(sum3, rshift3)


    # [4] conv
    weight4 = ng.variable(dtype=weight_dtype, shape=(48, 1, 1, 16), name="layer2.0.0.layers.0.weight")
    weight4.set_value(params["layer2.0.0.layers.0.weight"])

    bias4 = ng.variable(dtype=bias_dtype, shape=(48,), name="layer2.0.0.layers.0.bias")
    bias4.set_value(params["layer2.0.0.layers.0.bias"])

    scale4 = ng.variable(dtype=scale_dtype, shape=(48,), name="layer2.0.0.layers.0.scale")
    scale4.set_value(params["layer2.0.0.layers.0.scale"])

    conv4 = ng.multiply(ng.conv2d(act3, weight4, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale4)

    lshift4 = ng.constant([3], dtype=ng.int8)
    sum4 = ng.add(conv4, ng.lshift(bias4, lshift4))
    rshift4 = ng.constant([11], dtype=ng.int8)
    act4 = ng.relu(ng.rshift_round(sum4, rshift4))


    # [5] conv
    weight5 = ng.variable(dtype=weight_dtype, shape=(48, 3, 3, 48), name="layer2.0.0.layers.3.weight")
    weight5.set_value(np.array([[[[params["layer2.0.0.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(48)] for k in range(3)] for j in range(3)] for i in range(48)]))

    bias5 = ng.variable(dtype=bias_dtype, shape=(48,), name="layer2.0.0.layers.3.bias")
    bias5.set_value(params["layer2.0.0.layers.3.bias"])

    scale5 = ng.variable(dtype=scale_dtype, shape=(48,), name="layer2.0.0.layers.3.scale")
    scale5.set_value(params["layer2.0.0.layers.3.scale"])

    conv5 = ng.multiply(ng.conv2d(act4, weight5, strides=(1, 2, 2, 1), dtype=act_dtype, sum_dtype=ng.int32), scale5)

    lshift5 = ng.constant([6], dtype=ng.int8)
    sum5 = ng.add(conv5, ng.lshift(bias5, lshift5))
    rshift5 = ng.constant([11], dtype=ng.int8)
    act5 = ng.relu(ng.rshift_round(sum5, rshift5))


    # [6] conv
    weight6 = ng.variable(dtype=weight_dtype, shape=(24, 1, 1, 48), name="layer2.0.0.layers.6.weight")
    weight6.set_value(params["layer2.0.0.layers.6.weight"])

    bias6 = ng.variable(dtype=bias_dtype, shape=(24,), name="layer2.0.0.layers.6.bias")
    bias6.set_value(params["layer2.0.0.layers.6.bias"])

    scale6 = ng.variable(dtype=scale_dtype, shape=(24,), name="layer2.0.0.layers.6.scale")
    scale6.set_value(params["layer2.0.0.layers.6.scale"])

    conv6 = ng.multiply(ng.conv2d(act5, weight6, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale6)

    lshift6 = ng.constant([6], dtype=ng.int8)
    sum6 = ng.add(conv6, ng.lshift(bias6, lshift6))
    rshift6 = ng.constant([13], dtype=ng.int8)
    act6 = ng.rshift_round(sum6, rshift6)


    # [7] conv
    weight7 = ng.variable(dtype=weight_dtype, shape=(72, 1, 1, 24), name="layer2.0.1.layers.0.weight")
    weight7.set_value(params["layer2.0.1.layers.0.weight"])

    bias7 = ng.variable(dtype=bias_dtype, shape=(72,), name="layer2.0.1.layers.0.bias")
    bias7.set_value(params["layer2.0.1.layers.0.bias"])

    scale7 = ng.variable(dtype=scale_dtype, shape=(72,), name="layer2.0.1.layers.0.scale")
    scale7.set_value(params["layer2.0.1.layers.0.scale"])

    conv7 = ng.multiply(ng.conv2d(act6, weight7, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale7)

    lshift7 = ng.constant([4], dtype=ng.int8)
    sum7 = ng.add(conv7, ng.lshift(bias7, lshift7))
    rshift7 = ng.constant([11], dtype=ng.int8)
    act7 = ng.relu(ng.rshift_round(sum7, rshift7))


    # [8] conv
    weight8 = ng.variable(dtype=weight_dtype, shape=(72, 3, 3, 72), name="layer2.0.1.layers.3.weight")
    weight8.set_value(np.array([[[[params["layer2.0.1.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(72)] for k in range(3)] for j in range(3)] for i in range(72)]))

    bias8 = ng.variable(dtype=bias_dtype, shape=(72,), name="layer2.0.1.layers.3.bias")
    bias8.set_value(params["layer2.0.1.layers.3.bias"])

    scale8 = ng.variable(dtype=scale_dtype, shape=(72,), name="layer2.0.1.layers.3.scale")
    scale8.set_value(params["layer2.0.1.layers.3.scale"])

    conv8 = ng.multiply(ng.conv2d(act7, weight8, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale8)

    lshift8 = ng.constant([5], dtype=ng.int8)
    sum8 = ng.add(conv8, ng.lshift(bias8, lshift8))
    rshift8 = ng.constant([11], dtype=ng.int8)
    act8 = ng.relu(ng.rshift_round(sum8, rshift8))


    # [9] conv
    weight9 = ng.variable(dtype=weight_dtype, shape=(24, 1, 1, 72), name="layer2.0.1.layers.6.weight")
    weight9.set_value(params["layer2.0.1.layers.6.weight"])

    bias9 = ng.variable(dtype=bias_dtype, shape=(24,), name="layer2.0.1.layers.6.bias")
    bias9.set_value(params["layer2.0.1.layers.6.bias"])

    scale9 = ng.variable(dtype=scale_dtype, shape=(24,), name="layer2.0.1.layers.6.scale")
    scale9.set_value(params["layer2.0.1.layers.6.scale"])

    conv9 = ng.multiply(ng.conv2d(act8, weight9, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale9)

    lshift9 = ng.constant([5], dtype=ng.int8)
    sum9 = ng.add(conv9, ng.lshift(bias9, lshift9))
    rshift9 = ng.constant([13], dtype=ng.int8)
    act9 = ng.rshift_round(sum9, rshift9)


    # [10] add
    act10 = ng.add(act9, act6)


    # [11] conv
    weight11 = ng.variable(dtype=weight_dtype, shape=(72, 1, 1, 24), name="layer2.0.2.layers.0.weight")
    weight11.set_value(params["layer2.0.2.layers.0.weight"])

    bias11 = ng.variable(dtype=bias_dtype, shape=(72,), name="layer2.0.2.layers.0.bias")
    bias11.set_value(params["layer2.0.2.layers.0.bias"])

    scale11 = ng.variable(dtype=scale_dtype, shape=(72,), name="layer2.0.2.layers.0.scale")
    scale11.set_value(params["layer2.0.2.layers.0.scale"])

    conv11 = ng.multiply(ng.conv2d(act10, weight11, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale11)

    lshift11 = ng.constant([4], dtype=ng.int8)
    sum11 = ng.add(conv11, ng.lshift(bias11, lshift11))
    rshift11 = ng.constant([12], dtype=ng.int8)
    act11 = ng.relu(ng.rshift_round(sum11, rshift11))


    # [12] conv
    weight12 = ng.variable(dtype=weight_dtype, shape=(72, 3, 3, 72), name="layer2.0.2.layers.3.weight")
    weight12.set_value(np.array([[[[params["layer2.0.2.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(72)] for k in range(3)] for j in range(3)] for i in range(72)]))

    bias12 = ng.variable(dtype=bias_dtype, shape=(72,), name="layer2.0.2.layers.3.bias")
    bias12.set_value(params["layer2.0.2.layers.3.bias"])

    scale12 = ng.variable(dtype=scale_dtype, shape=(72,), name="layer2.0.2.layers.3.scale")
    scale12.set_value(params["layer2.0.2.layers.3.scale"])

    conv12 = ng.multiply(ng.conv2d(act11, weight12, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale12)

    lshift12 = ng.constant([2], dtype=ng.int8)
    sum12 = ng.add(conv12, ng.lshift(bias12, lshift12))
    rshift12 = ng.constant([8], dtype=ng.int8)
    act12 = ng.relu(ng.rshift_round(sum12, rshift12))


    # [13] conv
    weight13 = ng.variable(dtype=weight_dtype, shape=(24, 1, 1, 72), name="layer2.0.2.layers.6.weight")
    weight13.set_value(params["layer2.0.2.layers.6.weight"])

    bias13 = ng.variable(dtype=bias_dtype, shape=(24,), name="layer2.0.2.layers.6.bias")
    bias13.set_value(params["layer2.0.2.layers.6.bias"])

    scale13 = ng.variable(dtype=scale_dtype, shape=(24,), name="layer2.0.2.layers.6.scale")
    scale13.set_value(params["layer2.0.2.layers.6.scale"])

    conv13 = ng.multiply(ng.conv2d(act12, weight13, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale13)

    lshift13 = ng.constant([7], dtype=ng.int8)
    sum13 = ng.add(conv13, ng.lshift(bias13, lshift13))
    rshift13 = ng.constant([14], dtype=ng.int8)
    act13 = ng.rshift_round(sum13, rshift13)


    # [14] add
    act14 = ng.add(act13, act10)


    # [15] conv
    weight15 = ng.variable(dtype=weight_dtype, shape=(72, 1, 1, 24), name="layer3.0.0.layers.0.weight")
    weight15.set_value(params["layer3.0.0.layers.0.weight"])

    bias15 = ng.variable(dtype=bias_dtype, shape=(72,), name="layer3.0.0.layers.0.bias")
    bias15.set_value(params["layer3.0.0.layers.0.bias"])

    scale15 = ng.variable(dtype=scale_dtype, shape=(72,), name="layer3.0.0.layers.0.scale")
    scale15.set_value(params["layer3.0.0.layers.0.scale"])

    conv15 = ng.multiply(ng.conv2d(act14, weight15, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale15)

    lshift15 = ng.constant([4], dtype=ng.int8)
    sum15 = ng.add(conv15, ng.lshift(bias15, lshift15))
    rshift15 = ng.constant([13], dtype=ng.int8)
    act15 = ng.relu(ng.rshift_round(sum15, rshift15))


    # [16] conv
    weight16 = ng.variable(dtype=weight_dtype, shape=(72, 5, 5, 72), name="layer3.0.0.layers.3.weight")
    weight16.set_value(np.array([[[[params["layer3.0.0.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(72)] for k in range(5)] for j in range(5)] for i in range(72)]))

    bias16 = ng.variable(dtype=bias_dtype, shape=(72,), name="layer3.0.0.layers.3.bias")
    bias16.set_value(params["layer3.0.0.layers.3.bias"])

    scale16 = ng.variable(dtype=scale_dtype, shape=(72,), name="layer3.0.0.layers.3.scale")
    scale16.set_value(params["layer3.0.0.layers.3.scale"])

    conv16 = ng.multiply(ng.conv2d(act15, weight16, strides=(1, 2, 2, 1), dtype=act_dtype, sum_dtype=ng.int32), scale16)

    lshift16 = ng.constant([5], dtype=ng.int8)
    sum16 = ng.add(conv16, ng.lshift(bias16, lshift16))
    rshift16 = ng.constant([11], dtype=ng.int8)
    act16 = ng.relu(ng.rshift_round(sum16, rshift16))


    # [17] conv
    weight17 = ng.variable(dtype=weight_dtype, shape=(40, 1, 1, 72), name="layer3.0.0.layers.6.weight")
    weight17.set_value(params["layer3.0.0.layers.6.weight"])

    bias17 = ng.variable(dtype=bias_dtype, shape=(40,), name="layer3.0.0.layers.6.bias")
    bias17.set_value(params["layer3.0.0.layers.6.bias"])

    scale17 = ng.variable(dtype=scale_dtype, shape=(40,), name="layer3.0.0.layers.6.scale")
    scale17.set_value(params["layer3.0.0.layers.6.scale"])

    conv17 = ng.multiply(ng.conv2d(act16, weight17, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale17)

    lshift17 = ng.constant([7], dtype=ng.int8)
    sum17 = ng.add(conv17, ng.lshift(bias17, lshift17))
    rshift17 = ng.constant([13], dtype=ng.int8)
    act17 = ng.rshift_round(sum17, rshift17)


    # [18] conv
    weight18 = ng.variable(dtype=weight_dtype, shape=(120, 1, 1, 40), name="layer3.0.1.layers.0.weight")
    weight18.set_value(params["layer3.0.1.layers.0.weight"])

    bias18 = ng.variable(dtype=bias_dtype, shape=(120,), name="layer3.0.1.layers.0.bias")
    bias18.set_value(params["layer3.0.1.layers.0.bias"])

    scale18 = ng.variable(dtype=scale_dtype, shape=(120,), name="layer3.0.1.layers.0.scale")
    scale18.set_value(params["layer3.0.1.layers.0.scale"])

    conv18 = ng.multiply(ng.conv2d(act17, weight18, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale18)

    lshift18 = ng.constant([6], dtype=ng.int8)
    sum18 = ng.add(conv18, ng.lshift(bias18, lshift18))
    rshift18 = ng.constant([13], dtype=ng.int8)
    act18 = ng.relu(ng.rshift_round(sum18, rshift18))


    # [19] conv
    weight19 = ng.variable(dtype=weight_dtype, shape=(120, 5, 5, 120), name="layer3.0.1.layers.3.weight")
    weight19.set_value(np.array([[[[params["layer3.0.1.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(120)] for k in range(5)] for j in range(5)] for i in range(120)]))

    bias19 = ng.variable(dtype=bias_dtype, shape=(120,), name="layer3.0.1.layers.3.bias")
    bias19.set_value(params["layer3.0.1.layers.3.bias"])

    scale19 = ng.variable(dtype=scale_dtype, shape=(120,), name="layer3.0.1.layers.3.scale")
    scale19.set_value(params["layer3.0.1.layers.3.scale"])

    conv19 = ng.multiply(ng.conv2d(act18, weight19, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale19)

    lshift19 = ng.constant([5], dtype=ng.int8)
    sum19 = ng.add(conv19, ng.lshift(bias19, lshift19))
    rshift19 = ng.constant([9], dtype=ng.int8)
    act19 = ng.relu(ng.rshift_round(sum19, rshift19))


    # [20] conv
    weight20 = ng.variable(dtype=weight_dtype, shape=(40, 1, 1, 120), name="layer3.0.1.layers.6.weight")
    weight20.set_value(params["layer3.0.1.layers.6.weight"])

    bias20 = ng.variable(dtype=bias_dtype, shape=(40,), name="layer3.0.1.layers.6.bias")
    bias20.set_value(params["layer3.0.1.layers.6.bias"])

    scale20 = ng.variable(dtype=scale_dtype, shape=(40,), name="layer3.0.1.layers.6.scale")
    scale20.set_value(params["layer3.0.1.layers.6.scale"])

    conv20 = ng.multiply(ng.conv2d(act19, weight20, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale20)

    lshift20 = ng.constant([6], dtype=ng.int8)
    sum20 = ng.add(conv20, ng.lshift(bias20, lshift20))
    rshift20 = ng.constant([13], dtype=ng.int8)
    act20 = ng.rshift_round(sum20, rshift20)


    # [21] add
    rshift21 = ng.constant([1], dtype=ng.int8)
    act21 = ng.rshift_round(ng.add(act20, act17), rshift21)


    # [22] conv
    weight22 = ng.variable(dtype=weight_dtype, shape=(120, 1, 1, 40), name="layer3.0.2.layers.0.weight")
    weight22.set_value(params["layer3.0.2.layers.0.weight"])

    bias22 = ng.variable(dtype=bias_dtype, shape=(120,), name="layer3.0.2.layers.0.bias")
    bias22.set_value(params["layer3.0.2.layers.0.bias"])

    scale22 = ng.variable(dtype=scale_dtype, shape=(120,), name="layer3.0.2.layers.0.scale")
    scale22.set_value(params["layer3.0.2.layers.0.scale"])

    conv22 = ng.multiply(ng.conv2d(act21, weight22, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale22)

    lshift22 = ng.constant([4], dtype=ng.int8)
    sum22 = ng.add(conv22, ng.lshift(bias22, lshift22))
    rshift22 = ng.constant([12], dtype=ng.int8)
    act22 = ng.relu(ng.rshift_round(sum22, rshift22))


    # [23] conv
    weight23 = ng.variable(dtype=weight_dtype, shape=(120, 5, 5, 120), name="layer3.0.2.layers.3.weight")
    weight23.set_value(np.array([[[[params["layer3.0.2.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(120)] for k in range(5)] for j in range(5)] for i in range(120)]))

    bias23 = ng.variable(dtype=bias_dtype, shape=(120,), name="layer3.0.2.layers.3.bias")
    bias23.set_value(params["layer3.0.2.layers.3.bias"])

    scale23 = ng.variable(dtype=scale_dtype, shape=(120,), name="layer3.0.2.layers.3.scale")
    scale23.set_value(params["layer3.0.2.layers.3.scale"])

    conv23 = ng.multiply(ng.conv2d(act22, weight23, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale23)

    lshift23 = ng.constant([3], dtype=ng.int8)
    sum23 = ng.add(conv23, ng.lshift(bias23, lshift23))
    rshift23 = ng.constant([9], dtype=ng.int8)
    act23 = ng.relu(ng.rshift_round(sum23, rshift23))


    # [24] conv
    weight24 = ng.variable(dtype=weight_dtype, shape=(40, 1, 1, 120), name="layer3.0.2.layers.6.weight")
    weight24.set_value(params["layer3.0.2.layers.6.weight"])

    bias24 = ng.variable(dtype=bias_dtype, shape=(40,), name="layer3.0.2.layers.6.bias")
    bias24.set_value(params["layer3.0.2.layers.6.bias"])

    scale24 = ng.variable(dtype=scale_dtype, shape=(40,), name="layer3.0.2.layers.6.scale")
    scale24.set_value(params["layer3.0.2.layers.6.scale"])

    conv24 = ng.multiply(ng.conv2d(act23, weight24, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale24)

    lshift24 = ng.constant([7], dtype=ng.int8)
    sum24 = ng.add(conv24, ng.lshift(bias24, lshift24))
    rshift24 = ng.constant([13], dtype=ng.int8)
    act24 = ng.rshift_round(sum24, rshift24)


    # [25] add
    lshift25 = ng.constant([1], dtype=ng.int8)
    rshift25 = ng.constant([1], dtype=ng.int8)
    act25 = ng.rshift_round(ng.add(act24, ng.lshift(act21, lshift25)), rshift25)


    # [26] conv
    weight26 = ng.variable(dtype=weight_dtype, shape=(240, 1, 1, 40), name="layer4.0.0.layers.0.weight")
    weight26.set_value(params["layer4.0.0.layers.0.weight"])

    bias26 = ng.variable(dtype=bias_dtype, shape=(240,), name="layer4.0.0.layers.0.bias")
    bias26.set_value(params["layer4.0.0.layers.0.bias"])

    scale26 = ng.variable(dtype=scale_dtype, shape=(240,), name="layer4.0.0.layers.0.scale")
    scale26.set_value(params["layer4.0.0.layers.0.scale"])

    conv26 = ng.multiply(ng.conv2d(act25, weight26, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale26)

    lshift26 = ng.constant([4], dtype=ng.int8)
    sum26 = ng.add(conv26, ng.lshift(bias26, lshift26))
    rshift26 = ng.constant([12], dtype=ng.int8)
    act26 = ng.relu(ng.rshift_round(sum26, rshift26))


    # [27] conv
    weight27 = ng.variable(dtype=weight_dtype, shape=(240, 5, 5, 240), name="layer4.0.0.layers.3.weight")
    weight27.set_value(np.array([[[[params["layer4.0.0.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(240)] for k in range(5)] for j in range(5)] for i in range(240)]))

    bias27 = ng.variable(dtype=bias_dtype, shape=(240,), name="layer4.0.0.layers.3.bias")
    bias27.set_value(params["layer4.0.0.layers.3.bias"])

    scale27 = ng.variable(dtype=scale_dtype, shape=(240,), name="layer4.0.0.layers.3.scale")
    scale27.set_value(params["layer4.0.0.layers.3.scale"])

    conv27 = ng.multiply(ng.conv2d(act26, weight27, strides=(1, 2, 2, 1), dtype=act_dtype, sum_dtype=ng.int32), scale27)

    lshift27 = ng.constant([5], dtype=ng.int8)
    sum27 = ng.add(conv27, ng.lshift(bias27, lshift27))
    rshift27 = ng.constant([12], dtype=ng.int8)
    act27 = ng.relu(ng.rshift_round(sum27, rshift27))


    # [28] conv
    weight28 = ng.variable(dtype=weight_dtype, shape=(80, 1, 1, 240), name="layer4.0.0.layers.6.weight")
    weight28.set_value(params["layer4.0.0.layers.6.weight"])

    bias28 = ng.variable(dtype=bias_dtype, shape=(80,), name="layer4.0.0.layers.6.bias")
    bias28.set_value(params["layer4.0.0.layers.6.bias"])

    scale28 = ng.variable(dtype=scale_dtype, shape=(80,), name="layer4.0.0.layers.6.scale")
    scale28.set_value(params["layer4.0.0.layers.6.scale"])

    conv28 = ng.multiply(ng.conv2d(act27, weight28, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale28)

    lshift28 = ng.constant([6], dtype=ng.int8)
    sum28 = ng.add(conv28, ng.lshift(bias28, lshift28))
    rshift28 = ng.constant([13], dtype=ng.int8)
    act28 = ng.rshift_round(sum28, rshift28)


    # [29] conv
    weight29 = ng.variable(dtype=weight_dtype, shape=(480, 1, 1, 80), name="layer4.0.1.layers.0.weight")
    weight29.set_value(params["layer4.0.1.layers.0.weight"])

    bias29 = ng.variable(dtype=bias_dtype, shape=(480,), name="layer4.0.1.layers.0.bias")
    bias29.set_value(params["layer4.0.1.layers.0.bias"])

    scale29 = ng.variable(dtype=scale_dtype, shape=(480,), name="layer4.0.1.layers.0.scale")
    scale29.set_value(params["layer4.0.1.layers.0.scale"])

    conv29 = ng.multiply(ng.conv2d(act28, weight29, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale29)

    lshift29 = ng.constant([6], dtype=ng.int8)
    sum29 = ng.add(conv29, ng.lshift(bias29, lshift29))
    rshift29 = ng.constant([13], dtype=ng.int8)
    act29 = ng.relu(ng.rshift_round(sum29, rshift29))


    # [30] conv
    weight30 = ng.variable(dtype=weight_dtype, shape=(480, 5, 5, 480), name="layer4.0.1.layers.3.weight")
    weight30.set_value(np.array([[[[params["layer4.0.1.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(480)] for k in range(5)] for j in range(5)] for i in range(480)]))

    bias30 = ng.variable(dtype=bias_dtype, shape=(480,), name="layer4.0.1.layers.3.bias")
    bias30.set_value(params["layer4.0.1.layers.3.bias"])

    scale30 = ng.variable(dtype=scale_dtype, shape=(480,), name="layer4.0.1.layers.3.scale")
    scale30.set_value(params["layer4.0.1.layers.3.scale"])

    conv30 = ng.multiply(ng.conv2d(act29, weight30, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale30)

    lshift30 = ng.constant([2], dtype=ng.int8)
    sum30 = ng.add(conv30, ng.lshift(bias30, lshift30))
    rshift30 = ng.constant([7], dtype=ng.int8)
    act30 = ng.relu(ng.rshift_round(sum30, rshift30))


    # [31] conv
    weight31 = ng.variable(dtype=weight_dtype, shape=(80, 1, 1, 480), name="layer4.0.1.layers.6.weight")
    weight31.set_value(params["layer4.0.1.layers.6.weight"])

    bias31 = ng.variable(dtype=bias_dtype, shape=(80,), name="layer4.0.1.layers.6.bias")
    bias31.set_value(params["layer4.0.1.layers.6.bias"])

    scale31 = ng.variable(dtype=scale_dtype, shape=(80,), name="layer4.0.1.layers.6.scale")
    scale31.set_value(params["layer4.0.1.layers.6.scale"])

    conv31 = ng.multiply(ng.conv2d(act30, weight31, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale31)

    lshift31 = ng.constant([9], dtype=ng.int8)
    sum31 = ng.add(conv31, ng.lshift(bias31, lshift31))
    rshift31 = ng.constant([14], dtype=ng.int8)
    act31 = ng.rshift_round(sum31, rshift31)


    # [32] add
    lshift32 = ng.constant([1], dtype=ng.int8)
    rshift32 = ng.constant([1], dtype=ng.int8)
    act32 = ng.rshift_round(ng.add(act31, ng.lshift(act28, lshift32)), rshift32)


    # [33] conv
    weight33 = ng.variable(dtype=weight_dtype, shape=(480, 1, 1, 80), name="layer4.0.2.layers.0.weight")
    weight33.set_value(params["layer4.0.2.layers.0.weight"])

    bias33 = ng.variable(dtype=bias_dtype, shape=(480,), name="layer4.0.2.layers.0.bias")
    bias33.set_value(params["layer4.0.2.layers.0.bias"])

    scale33 = ng.variable(dtype=scale_dtype, shape=(480,), name="layer4.0.2.layers.0.scale")
    scale33.set_value(params["layer4.0.2.layers.0.scale"])

    conv33 = ng.multiply(ng.conv2d(act32, weight33, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale33)

    lshift33 = ng.constant([5], dtype=ng.int8)
    sum33 = ng.add(conv33, ng.lshift(bias33, lshift33))
    rshift33 = ng.constant([12], dtype=ng.int8)
    act33 = ng.relu(ng.rshift_round(sum33, rshift33))


    # [34] conv
    weight34 = ng.variable(dtype=weight_dtype, shape=(480, 5, 5, 480), name="layer4.0.2.layers.3.weight")
    weight34.set_value(np.array([[[[params["layer4.0.2.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(480)] for k in range(5)] for j in range(5)] for i in range(480)]))

    bias34 = ng.variable(dtype=bias_dtype, shape=(480,), name="layer4.0.2.layers.3.bias")
    bias34.set_value(params["layer4.0.2.layers.3.bias"])

    scale34 = ng.variable(dtype=scale_dtype, shape=(480,), name="layer4.0.2.layers.3.scale")
    scale34.set_value(params["layer4.0.2.layers.3.scale"])

    conv34 = ng.multiply(ng.conv2d(act33, weight34, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale34)

    lshift34 = ng.constant([2], dtype=ng.int8)
    sum34 = ng.add(conv34, ng.lshift(bias34, lshift34))
    rshift34 = ng.constant([8], dtype=ng.int8)
    act34 = ng.relu(ng.rshift_round(sum34, rshift34))


    # [35] conv
    weight35 = ng.variable(dtype=weight_dtype, shape=(80, 1, 1, 480), name="layer4.0.2.layers.6.weight")
    weight35.set_value(params["layer4.0.2.layers.6.weight"])

    bias35 = ng.variable(dtype=bias_dtype, shape=(80,), name="layer4.0.2.layers.6.bias")
    bias35.set_value(params["layer4.0.2.layers.6.bias"])

    scale35 = ng.variable(dtype=scale_dtype, shape=(80,), name="layer4.0.2.layers.6.scale")
    scale35.set_value(params["layer4.0.2.layers.6.scale"])

    conv35 = ng.multiply(ng.conv2d(act34, weight35, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale35)

    lshift35 = ng.constant([6], dtype=ng.int8)
    sum35 = ng.add(conv35, ng.lshift(bias35, lshift35))
    rshift35 = ng.constant([13], dtype=ng.int8)
    act35 = ng.rshift_round(sum35, rshift35)


    # [36] add
    lshift36 = ng.constant([1], dtype=ng.int8)
    rshift36 = ng.constant([1], dtype=ng.int8)
    act36 = ng.rshift_round(ng.add(act35, ng.lshift(act32, lshift36)), rshift36)


    # [37] conv
    weight37 = ng.variable(dtype=weight_dtype, shape=(480, 1, 1, 80), name="layer4.1.0.layers.0.weight")
    weight37.set_value(params["layer4.1.0.layers.0.weight"])

    bias37 = ng.variable(dtype=bias_dtype, shape=(480,), name="layer4.1.0.layers.0.bias")
    bias37.set_value(params["layer4.1.0.layers.0.bias"])

    scale37 = ng.variable(dtype=scale_dtype, shape=(480,), name="layer4.1.0.layers.0.scale")
    scale37.set_value(params["layer4.1.0.layers.0.scale"])

    conv37 = ng.multiply(ng.conv2d(act36, weight37, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale37)

    lshift37 = ng.constant([6], dtype=ng.int8)
    sum37 = ng.add(conv37, ng.lshift(bias37, lshift37))
    rshift37 = ng.constant([12], dtype=ng.int8)
    act37 = ng.relu(ng.rshift_round(sum37, rshift37))


    # [38] conv
    weight38 = ng.variable(dtype=weight_dtype, shape=(480, 3, 3, 480), name="layer4.1.0.layers.3.weight")
    weight38.set_value(np.array([[[[params["layer4.1.0.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(480)] for k in range(3)] for j in range(3)] for i in range(480)]))

    bias38 = ng.variable(dtype=bias_dtype, shape=(480,), name="layer4.1.0.layers.3.bias")
    bias38.set_value(params["layer4.1.0.layers.3.bias"])

    scale38 = ng.variable(dtype=scale_dtype, shape=(480,), name="layer4.1.0.layers.3.scale")
    scale38.set_value(params["layer4.1.0.layers.3.scale"])

    conv38 = ng.multiply(ng.conv2d(act37, weight38, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale38)

    lshift38 = ng.constant([3], dtype=ng.int8)
    sum38 = ng.add(conv38, ng.lshift(bias38, lshift38))
    rshift38 = ng.constant([9], dtype=ng.int8)
    act38 = ng.relu(ng.rshift_round(sum38, rshift38))


    # [39] conv
    weight39 = ng.variable(dtype=weight_dtype, shape=(96, 1, 1, 480), name="layer4.1.0.layers.6.weight")
    weight39.set_value(params["layer4.1.0.layers.6.weight"])

    bias39 = ng.variable(dtype=bias_dtype, shape=(96,), name="layer4.1.0.layers.6.bias")
    bias39.set_value(params["layer4.1.0.layers.6.bias"])

    scale39 = ng.variable(dtype=scale_dtype, shape=(96,), name="layer4.1.0.layers.6.scale")
    scale39.set_value(params["layer4.1.0.layers.6.scale"])

    conv39 = ng.multiply(ng.conv2d(act38, weight39, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale39)

    lshift39 = ng.constant([7], dtype=ng.int8)
    sum39 = ng.add(conv39, ng.lshift(bias39, lshift39))
    rshift39 = ng.constant([14], dtype=ng.int8)
    act39 = ng.rshift_round(sum39, rshift39)


    # [40] conv
    weight40 = ng.variable(dtype=weight_dtype, shape=(576, 1, 1, 96), name="layer4.1.1.layers.0.weight")
    weight40.set_value(params["layer4.1.1.layers.0.weight"])

    bias40 = ng.variable(dtype=bias_dtype, shape=(576,), name="layer4.1.1.layers.0.bias")
    bias40.set_value(params["layer4.1.1.layers.0.bias"])

    scale40 = ng.variable(dtype=scale_dtype, shape=(576,), name="layer4.1.1.layers.0.scale")
    scale40.set_value(params["layer4.1.1.layers.0.scale"])

    conv40 = ng.multiply(ng.conv2d(act39, weight40, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale40)

    lshift40 = ng.constant([6], dtype=ng.int8)
    sum40 = ng.add(conv40, ng.lshift(bias40, lshift40))
    rshift40 = ng.constant([13], dtype=ng.int8)
    act40 = ng.relu(ng.rshift_round(sum40, rshift40))


    # [41] conv
    weight41 = ng.variable(dtype=weight_dtype, shape=(576, 3, 3, 576), name="layer4.1.1.layers.3.weight")
    weight41.set_value(np.array([[[[params["layer4.1.1.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(576)] for k in range(3)] for j in range(3)] for i in range(576)]))

    bias41 = ng.variable(dtype=bias_dtype, shape=(576,), name="layer4.1.1.layers.3.bias")
    bias41.set_value(params["layer4.1.1.layers.3.bias"])

    scale41 = ng.variable(dtype=scale_dtype, shape=(576,), name="layer4.1.1.layers.3.scale")
    scale41.set_value(params["layer4.1.1.layers.3.scale"])

    conv41 = ng.multiply(ng.conv2d(act40, weight41, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale41)

    lshift41 = ng.constant([2], dtype=ng.int8)
    sum41 = ng.add(conv41, ng.lshift(bias41, lshift41))
    rshift41 = ng.constant([7], dtype=ng.int8)
    act41 = ng.relu(ng.rshift_round(sum41, rshift41))


    # [42] conv
    weight42 = ng.variable(dtype=weight_dtype, shape=(96, 1, 1, 576), name="layer4.1.1.layers.6.weight")
    weight42.set_value(params["layer4.1.1.layers.6.weight"])

    bias42 = ng.variable(dtype=bias_dtype, shape=(96,), name="layer4.1.1.layers.6.bias")
    bias42.set_value(params["layer4.1.1.layers.6.bias"])

    scale42 = ng.variable(dtype=scale_dtype, shape=(96,), name="layer4.1.1.layers.6.scale")
    scale42.set_value(params["layer4.1.1.layers.6.scale"])

    conv42 = ng.multiply(ng.conv2d(act41, weight42, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale42)

    lshift42 = ng.constant([8], dtype=ng.int8)
    sum42 = ng.add(conv42, ng.lshift(bias42, lshift42))
    rshift42 = ng.constant([15], dtype=ng.int8)
    act42 = ng.rshift_round(sum42, rshift42)


    # [43] add
    lshift43 = ng.constant([1], dtype=ng.int8)
    rshift43 = ng.constant([1], dtype=ng.int8)
    act43 = ng.rshift_round(ng.add(act42, ng.lshift(act39, lshift43)), rshift43)


    # [44] conv
    weight44 = ng.variable(dtype=weight_dtype, shape=(576, 1, 1, 96), name="layer5.0.0.layers.0.weight")
    weight44.set_value(params["layer5.0.0.layers.0.weight"])

    bias44 = ng.variable(dtype=bias_dtype, shape=(576,), name="layer5.0.0.layers.0.bias")
    bias44.set_value(params["layer5.0.0.layers.0.bias"])

    scale44 = ng.variable(dtype=scale_dtype, shape=(576,), name="layer5.0.0.layers.0.scale")
    scale44.set_value(params["layer5.0.0.layers.0.scale"])

    conv44 = ng.multiply(ng.conv2d(act43, weight44, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale44)

    lshift44 = ng.constant([5], dtype=ng.int8)
    sum44 = ng.add(conv44, ng.lshift(bias44, lshift44))
    rshift44 = ng.constant([12], dtype=ng.int8)
    act44 = ng.relu(ng.rshift_round(sum44, rshift44))


    # [45] conv
    weight45 = ng.variable(dtype=weight_dtype, shape=(576, 5, 5, 576), name="layer5.0.0.layers.3.weight")
    weight45.set_value(np.array([[[[params["layer5.0.0.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(576)] for k in range(5)] for j in range(5)] for i in range(576)]))

    bias45 = ng.variable(dtype=bias_dtype, shape=(576,), name="layer5.0.0.layers.3.bias")
    bias45.set_value(params["layer5.0.0.layers.3.bias"])

    scale45 = ng.variable(dtype=scale_dtype, shape=(576,), name="layer5.0.0.layers.3.scale")
    scale45.set_value(params["layer5.0.0.layers.3.scale"])

    conv45 = ng.multiply(ng.conv2d(act44, weight45, strides=(1, 2, 2, 1), dtype=act_dtype, sum_dtype=ng.int32), scale45)

    lshift45 = ng.constant([3], dtype=ng.int8)
    sum45 = ng.add(conv45, ng.lshift(bias45, lshift45))
    rshift45 = ng.constant([9], dtype=ng.int8)
    act45 = ng.relu(ng.rshift_round(sum45, rshift45))


    # [46] conv
    weight46 = ng.variable(dtype=weight_dtype, shape=(192, 1, 1, 576), name="layer5.0.0.layers.6.weight")
    weight46.set_value(params["layer5.0.0.layers.6.weight"])

    bias46 = ng.variable(dtype=bias_dtype, shape=(192,), name="layer5.0.0.layers.6.bias")
    bias46.set_value(params["layer5.0.0.layers.6.bias"])

    scale46 = ng.variable(dtype=scale_dtype, shape=(192,), name="layer5.0.0.layers.6.scale")
    scale46.set_value(params["layer5.0.0.layers.6.scale"])

    conv46 = ng.multiply(ng.conv2d(act45, weight46, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale46)

    lshift46 = ng.constant([8], dtype=ng.int8)
    sum46 = ng.add(conv46, ng.lshift(bias46, lshift46))
    rshift46 = ng.constant([15], dtype=ng.int8)
    act46 = ng.rshift_round(sum46, rshift46)


    # [47] conv
    weight47 = ng.variable(dtype=weight_dtype, shape=(1152, 1, 1, 192), name="layer5.0.1.layers.0.weight")
    weight47.set_value(params["layer5.0.1.layers.0.weight"])

    bias47 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.0.1.layers.0.bias")
    bias47.set_value(params["layer5.0.1.layers.0.bias"])

    scale47 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.0.1.layers.0.scale")
    scale47.set_value(params["layer5.0.1.layers.0.scale"])

    conv47 = ng.multiply(ng.conv2d(act46, weight47, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale47)

    lshift47 = ng.constant([6], dtype=ng.int8)
    sum47 = ng.add(conv47, ng.lshift(bias47, lshift47))
    rshift47 = ng.constant([13], dtype=ng.int8)
    act47 = ng.relu(ng.rshift_round(sum47, rshift47))


    # [48] conv
    weight48 = ng.variable(dtype=weight_dtype, shape=(1152, 5, 5, 1152), name="layer5.0.1.layers.3.weight")
    weight48.set_value(np.array([[[[params["layer5.0.1.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(1152)] for k in range(5)] for j in range(5)] for i in range(1152)]))

    bias48 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.0.1.layers.3.bias")
    bias48.set_value(params["layer5.0.1.layers.3.bias"])

    scale48 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.0.1.layers.3.scale")
    scale48.set_value(params["layer5.0.1.layers.3.scale"])

    conv48 = ng.multiply(ng.conv2d(act47, weight48, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale48)

    lshift48 = ng.constant([1], dtype=ng.int8)
    sum48 = ng.add(conv48, ng.lshift(bias48, lshift48))
    rshift48 = ng.constant([7], dtype=ng.int8)
    act48 = ng.relu(ng.rshift_round(sum48, rshift48))


    # [49] conv
    weight49 = ng.variable(dtype=weight_dtype, shape=(192, 1, 1, 1152), name="layer5.0.1.layers.6.weight")
    weight49.set_value(params["layer5.0.1.layers.6.weight"])

    bias49 = ng.variable(dtype=bias_dtype, shape=(192,), name="layer5.0.1.layers.6.bias")
    bias49.set_value(params["layer5.0.1.layers.6.bias"])

    scale49 = ng.variable(dtype=scale_dtype, shape=(192,), name="layer5.0.1.layers.6.scale")
    scale49.set_value(params["layer5.0.1.layers.6.scale"])

    conv49 = ng.multiply(ng.conv2d(act48, weight49, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale49)

    lshift49 = ng.constant([8], dtype=ng.int8)
    sum49 = ng.add(conv49, ng.lshift(bias49, lshift49))
    rshift49 = ng.constant([15], dtype=ng.int8)
    act49 = ng.rshift_round(sum49, rshift49)


    # [50] add
    lshift50 = ng.constant([1], dtype=ng.int8)
    rshift50 = ng.constant([1], dtype=ng.int8)
    act50 = ng.rshift_round(ng.add(act49, ng.lshift(act46, lshift50)), rshift50)


    # [51] conv
    weight51 = ng.variable(dtype=weight_dtype, shape=(1152, 1, 1, 192), name="layer5.0.2.layers.0.weight")
    weight51.set_value(params["layer5.0.2.layers.0.weight"])

    bias51 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.0.2.layers.0.bias")
    bias51.set_value(params["layer5.0.2.layers.0.bias"])

    scale51 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.0.2.layers.0.scale")
    scale51.set_value(params["layer5.0.2.layers.0.scale"])

    conv51 = ng.multiply(ng.conv2d(act50, weight51, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale51)

    lshift51 = ng.constant([6], dtype=ng.int8)
    sum51 = ng.add(conv51, ng.lshift(bias51, lshift51))
    rshift51 = ng.constant([13], dtype=ng.int8)
    act51 = ng.relu(ng.rshift_round(sum51, rshift51))


    # [52] conv
    weight52 = ng.variable(dtype=weight_dtype, shape=(1152, 5, 5, 1152), name="layer5.0.2.layers.3.weight")
    weight52.set_value(np.array([[[[params["layer5.0.2.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(1152)] for k in range(5)] for j in range(5)] for i in range(1152)]))

    bias52 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.0.2.layers.3.bias")
    bias52.set_value(params["layer5.0.2.layers.3.bias"])

    scale52 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.0.2.layers.3.scale")
    scale52.set_value(params["layer5.0.2.layers.3.scale"])

    conv52 = ng.multiply(ng.conv2d(act51, weight52, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale52)

    lshift52 = ng.constant([2], dtype=ng.int8)
    sum52 = ng.add(conv52, ng.lshift(bias52, lshift52))
    rshift52 = ng.constant([7], dtype=ng.int8)
    act52 = ng.relu(ng.rshift_round(sum52, rshift52))


    # [53] conv
    weight53 = ng.variable(dtype=weight_dtype, shape=(192, 1, 1, 1152), name="layer5.0.2.layers.6.weight")
    weight53.set_value(params["layer5.0.2.layers.6.weight"])

    bias53 = ng.variable(dtype=bias_dtype, shape=(192,), name="layer5.0.2.layers.6.bias")
    bias53.set_value(params["layer5.0.2.layers.6.bias"])

    scale53 = ng.variable(dtype=scale_dtype, shape=(192,), name="layer5.0.2.layers.6.scale")
    scale53.set_value(params["layer5.0.2.layers.6.scale"])

    conv53 = ng.multiply(ng.conv2d(act52, weight53, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale53)

    lshift53 = ng.constant([8], dtype=ng.int8)
    sum53 = ng.add(conv53, ng.lshift(bias53, lshift53))
    rshift53 = ng.constant([15], dtype=ng.int8)
    act53 = ng.rshift_round(sum53, rshift53)


    # [54] add
    lshift54 = ng.constant([1], dtype=ng.int8)
    rshift54 = ng.constant([1], dtype=ng.int8)
    act54 = ng.rshift_round(ng.add(act53, ng.lshift(act50, lshift54)), rshift54)


    # [55] conv
    weight55 = ng.variable(dtype=weight_dtype, shape=(1152, 1, 1, 192), name="layer5.0.3.layers.0.weight")
    weight55.set_value(params["layer5.0.3.layers.0.weight"])

    bias55 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.0.3.layers.0.bias")
    bias55.set_value(params["layer5.0.3.layers.0.bias"])

    scale55 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.0.3.layers.0.scale")
    scale55.set_value(params["layer5.0.3.layers.0.scale"])

    conv55 = ng.multiply(ng.conv2d(act54, weight55, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale55)

    lshift55 = ng.constant([7], dtype=ng.int8)
    sum55 = ng.add(conv55, ng.lshift(bias55, lshift55))
    rshift55 = ng.constant([14], dtype=ng.int8)
    act55 = ng.relu(ng.rshift_round(sum55, rshift55))


    # [56] conv
    weight56 = ng.variable(dtype=weight_dtype, shape=(1152, 5, 5, 1152), name="layer5.0.3.layers.3.weight")
    weight56.set_value(np.array([[[[params["layer5.0.3.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(1152)] for k in range(5)] for j in range(5)] for i in range(1152)]))

    bias56 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.0.3.layers.3.bias")
    bias56.set_value(params["layer5.0.3.layers.3.bias"])

    scale56 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.0.3.layers.3.scale")
    scale56.set_value(params["layer5.0.3.layers.3.scale"])

    conv56 = ng.multiply(ng.conv2d(act55, weight56, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale56)

    lshift56 = ng.constant([2], dtype=ng.int8)
    sum56 = ng.add(conv56, ng.lshift(bias56, lshift56))
    rshift56 = ng.constant([7], dtype=ng.int8)
    act56 = ng.relu(ng.rshift_round(sum56, rshift56))


    # [57] conv
    weight57 = ng.variable(dtype=weight_dtype, shape=(192, 1, 1, 1152), name="layer5.0.3.layers.6.weight")
    weight57.set_value(params["layer5.0.3.layers.6.weight"])

    bias57 = ng.variable(dtype=bias_dtype, shape=(192,), name="layer5.0.3.layers.6.bias")
    bias57.set_value(params["layer5.0.3.layers.6.bias"])

    scale57 = ng.variable(dtype=scale_dtype, shape=(192,), name="layer5.0.3.layers.6.scale")
    scale57.set_value(params["layer5.0.3.layers.6.scale"])

    conv57 = ng.multiply(ng.conv2d(act56, weight57, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale57)

    lshift57 = ng.constant([9], dtype=ng.int8)
    sum57 = ng.add(conv57, ng.lshift(bias57, lshift57))
    rshift57 = ng.constant([15], dtype=ng.int8)
    act57 = ng.rshift_round(sum57, rshift57)


    # [58] add
    lshift58 = ng.constant([1], dtype=ng.int8)
    rshift58 = ng.constant([1], dtype=ng.int8)
    act58 = ng.rshift_round(ng.add(act57, ng.lshift(act54, lshift58)), rshift58)


    # [59] conv
    weight59 = ng.variable(dtype=weight_dtype, shape=(1152, 1, 1, 192), name="layer5.1.0.layers.0.weight")
    weight59.set_value(params["layer5.1.0.layers.0.weight"])

    bias59 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.1.0.layers.0.bias")
    bias59.set_value(params["layer5.1.0.layers.0.bias"])

    scale59 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.1.0.layers.0.scale")
    scale59.set_value(params["layer5.1.0.layers.0.scale"])

    conv59 = ng.multiply(ng.conv2d(act58, weight59, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale59)

    lshift59 = ng.constant([7], dtype=ng.int8)
    sum59 = ng.add(conv59, ng.lshift(bias59, lshift59))
    rshift59 = ng.constant([14], dtype=ng.int8)
    act59 = ng.relu(ng.rshift_round(sum59, rshift59))


    # [60] conv
    weight60 = ng.variable(dtype=weight_dtype, shape=(1152, 3, 3, 1152), name="layer5.1.0.layers.3.weight")
    weight60.set_value(np.array([[[[params["layer5.1.0.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(1152)] for k in range(3)] for j in range(3)] for i in range(1152)]))

    bias60 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.1.0.layers.3.bias")
    bias60.set_value(params["layer5.1.0.layers.3.bias"])

    scale60 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.1.0.layers.3.scale")
    scale60.set_value(params["layer5.1.0.layers.3.scale"])

    conv60 = ng.multiply(ng.conv2d(act59, weight60, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale60)

    lshift60 = ng.constant([3], dtype=ng.int8)
    sum60 = ng.add(conv60, ng.lshift(bias60, lshift60))
    rshift60 = ng.constant([9], dtype=ng.int8)
    act60 = ng.relu(ng.rshift_round(sum60, rshift60))


    # [61] conv
    weight61 = ng.variable(dtype=weight_dtype, shape=(320, 1, 1, 1152), name="layer5.1.0.layers.6.weight")
    weight61.set_value(params["layer5.1.0.layers.6.weight"])

    bias61 = ng.variable(dtype=bias_dtype, shape=(320,), name="layer5.1.0.layers.6.bias")
    bias61.set_value(params["layer5.1.0.layers.6.bias"])

    scale61 = ng.variable(dtype=scale_dtype, shape=(320,), name="layer5.1.0.layers.6.scale")
    scale61.set_value(params["layer5.1.0.layers.6.scale"])

    conv61 = ng.multiply(ng.conv2d(act60, weight61, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32), scale61)

    lshift61 = ng.constant([9], dtype=ng.int8)
    sum61 = ng.add(conv61, ng.lshift(bias61, lshift61))
    rshift61 = ng.constant([15], dtype=ng.int8)
    act61 = ng.rshift_round(sum61, rshift61)


    return act3, act14, act25, act43, act61