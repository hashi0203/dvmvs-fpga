import numpy as np
import nngen as ng
from utils import rshift_round_and_clip

def feature_extractor(act0, params,
                      weight_dtype=ng.int8, bias_dtype=ng.int32, scale_dtype=ng.int8, act_dtype=ng.int16, mid_dtype=ng.int32):

    # [1] conv
    weight1 = ng.variable(dtype=weight_dtype, shape=(32, 3, 3, 3), name="layer1.0.weight")
    weight1.set_value(params["layer1.0.weight"])

    bias1 = ng.variable(dtype=bias_dtype, shape=(32,), name="layer1.0.bias")
    bias1.set_value(np.round(params["layer1.0.bias"] / (float) (1 << 6)).astype(params["layer1.0.bias"].dtype))

    scale1 = ng.variable(dtype=scale_dtype, shape=(32,), name="layer1.0.scale")
    scale1.set_value(params["layer1.0.scale"])

    rshift1 = ng.constant([12], dtype=ng.int8)
    act1 = ng.conv2d(act0, weight1, strides=(1, 2, 2, 1), bias=bias1, scale=scale1, rshift_out=rshift1, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [2] conv
    weight2 = ng.variable(dtype=weight_dtype, shape=(32, 3, 3, 32), name="layer1.3.weight")
    weight2.set_value(np.array([[[[params["layer1.3.weight"][i][j][k][0] if i == l else 0 for l in range(32)] for k in range(3)] for j in range(3)] for i in range(32)]))

    bias2 = ng.variable(dtype=bias_dtype, shape=(32,), name="layer1.3.bias")
    bias2.set_value(np.round(params["layer1.3.bias"] / (float) (1 << 9)).astype(params["layer1.3.bias"].dtype))

    scale2 = ng.variable(dtype=scale_dtype, shape=(32,), name="layer1.3.scale")
    scale2.set_value(params["layer1.3.scale"])

    rshift2 = ng.constant([7], dtype=ng.int8)
    act2 = ng.conv2d(act1, weight2, strides=(1, 1, 1, 1), bias=bias2, scale=scale2, rshift_out=rshift2, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [3] conv
    weight3 = ng.variable(dtype=weight_dtype, shape=(16, 1, 1, 32), name="layer1.6.weight")
    weight3.set_value(params["layer1.6.weight"])

    bias3 = ng.variable(dtype=bias_dtype, shape=(16,), name="layer1.6.bias")
    bias3.set_value(np.round(params["layer1.6.bias"] / (float) (1 << 6)).astype(params["layer1.6.bias"].dtype))

    scale3 = ng.variable(dtype=scale_dtype, shape=(16,), name="layer1.6.scale")
    scale3.set_value(params["layer1.6.scale"])

    rshift3 = ng.constant([13], dtype=ng.int8)
    act3 = ng.conv2d(act2, weight3, strides=(1, 1, 1, 1), bias=bias3, scale=scale3, rshift_out=rshift3, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [4] conv
    weight4 = ng.variable(dtype=weight_dtype, shape=(48, 1, 1, 16), name="layer2.0.0.layers.0.weight")
    weight4.set_value(params["layer2.0.0.layers.0.weight"])

    bias4 = ng.variable(dtype=bias_dtype, shape=(48,), name="layer2.0.0.layers.0.bias")
    bias4.set_value(np.round(params["layer2.0.0.layers.0.bias"] / (float) (1 << 6)).astype(params["layer2.0.0.layers.0.bias"].dtype))

    scale4 = ng.variable(dtype=scale_dtype, shape=(48,), name="layer2.0.0.layers.0.scale")
    scale4.set_value(params["layer2.0.0.layers.0.scale"])

    rshift4 = ng.constant([11], dtype=ng.int8)
    act4 = ng.conv2d(act3, weight4, strides=(1, 1, 1, 1), bias=bias4, scale=scale4, rshift_out=rshift4, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [5] conv
    weight5 = ng.variable(dtype=weight_dtype, shape=(48, 3, 3, 48), name="layer2.0.0.layers.3.weight")
    weight5.set_value(np.array([[[[params["layer2.0.0.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(48)] for k in range(3)] for j in range(3)] for i in range(48)]))

    bias5 = ng.variable(dtype=bias_dtype, shape=(48,), name="layer2.0.0.layers.3.bias")
    bias5.set_value(np.round(params["layer2.0.0.layers.3.bias"] / (float) (1 << 5)).astype(params["layer2.0.0.layers.3.bias"].dtype))

    scale5 = ng.variable(dtype=scale_dtype, shape=(48,), name="layer2.0.0.layers.3.scale")
    scale5.set_value(params["layer2.0.0.layers.3.scale"])

    rshift5 = ng.constant([11], dtype=ng.int8)
    act5 = ng.conv2d(act4, weight5, strides=(1, 2, 2, 1), bias=bias5, scale=scale5, rshift_out=rshift5, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [6] conv
    weight6 = ng.variable(dtype=weight_dtype, shape=(24, 1, 1, 48), name="layer2.0.0.layers.6.weight")
    weight6.set_value(params["layer2.0.0.layers.6.weight"])

    bias6 = ng.variable(dtype=bias_dtype, shape=(24,), name="layer2.0.0.layers.6.bias")
    bias6.set_value(np.round(params["layer2.0.0.layers.6.bias"] / (float) (1 << 6)).astype(params["layer2.0.0.layers.6.bias"].dtype))

    scale6 = ng.variable(dtype=scale_dtype, shape=(24,), name="layer2.0.0.layers.6.scale")
    scale6.set_value(params["layer2.0.0.layers.6.scale"])

    rshift6 = ng.constant([13], dtype=ng.int8)
    act6 = ng.conv2d(act5, weight6, strides=(1, 1, 1, 1), bias=bias6, scale=scale6, rshift_out=rshift6, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [7] conv
    weight7 = ng.variable(dtype=weight_dtype, shape=(72, 1, 1, 24), name="layer2.0.1.layers.0.weight")
    weight7.set_value(params["layer2.0.1.layers.0.weight"])

    bias7 = ng.variable(dtype=bias_dtype, shape=(72,), name="layer2.0.1.layers.0.bias")
    bias7.set_value(np.round(params["layer2.0.1.layers.0.bias"] / (float) (1 << 6)).astype(params["layer2.0.1.layers.0.bias"].dtype))

    scale7 = ng.variable(dtype=scale_dtype, shape=(72,), name="layer2.0.1.layers.0.scale")
    scale7.set_value(params["layer2.0.1.layers.0.scale"])

    rshift7 = ng.constant([11], dtype=ng.int8)
    act7 = ng.conv2d(act6, weight7, strides=(1, 1, 1, 1), bias=bias7, scale=scale7, rshift_out=rshift7, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [8] conv
    weight8 = ng.variable(dtype=weight_dtype, shape=(72, 3, 3, 72), name="layer2.0.1.layers.3.weight")
    weight8.set_value(np.array([[[[params["layer2.0.1.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(72)] for k in range(3)] for j in range(3)] for i in range(72)]))

    bias8 = ng.variable(dtype=bias_dtype, shape=(72,), name="layer2.0.1.layers.3.bias")
    bias8.set_value(np.round(params["layer2.0.1.layers.3.bias"] / (float) (1 << 6)).astype(params["layer2.0.1.layers.3.bias"].dtype))

    scale8 = ng.variable(dtype=scale_dtype, shape=(72,), name="layer2.0.1.layers.3.scale")
    scale8.set_value(params["layer2.0.1.layers.3.scale"])

    rshift8 = ng.constant([11], dtype=ng.int8)
    act8 = ng.conv2d(act7, weight8, strides=(1, 1, 1, 1), bias=bias8, scale=scale8, rshift_out=rshift8, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [9] conv
    weight9 = ng.variable(dtype=weight_dtype, shape=(24, 1, 1, 72), name="layer2.0.1.layers.6.weight")
    weight9.set_value(params["layer2.0.1.layers.6.weight"])

    bias9 = ng.variable(dtype=bias_dtype, shape=(24,), name="layer2.0.1.layers.6.bias")
    bias9.set_value(np.round(params["layer2.0.1.layers.6.bias"] / (float) (1 << 6)).astype(params["layer2.0.1.layers.6.bias"].dtype))

    scale9 = ng.variable(dtype=scale_dtype, shape=(24,), name="layer2.0.1.layers.6.scale")
    scale9.set_value(params["layer2.0.1.layers.6.scale"])

    rshift9 = ng.constant([13], dtype=ng.int8)
    act9 = ng.conv2d(act8, weight9, strides=(1, 1, 1, 1), bias=bias9, scale=scale9, rshift_out=rshift9, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [10] add
    act10 = ng.add(act9, act6)


    # [11] conv
    weight11 = ng.variable(dtype=weight_dtype, shape=(72, 1, 1, 24), name="layer2.0.2.layers.0.weight")
    weight11.set_value(params["layer2.0.2.layers.0.weight"])

    bias11 = ng.variable(dtype=bias_dtype, shape=(72,), name="layer2.0.2.layers.0.bias")
    bias11.set_value(np.round(params["layer2.0.2.layers.0.bias"] / (float) (1 << 6)).astype(params["layer2.0.2.layers.0.bias"].dtype))

    scale11 = ng.variable(dtype=scale_dtype, shape=(72,), name="layer2.0.2.layers.0.scale")
    scale11.set_value(params["layer2.0.2.layers.0.scale"])

    rshift11 = ng.constant([12], dtype=ng.int8)
    act11 = ng.conv2d(act10, weight11, strides=(1, 1, 1, 1), bias=bias11, scale=scale11, rshift_out=rshift11, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [12] conv
    weight12 = ng.variable(dtype=weight_dtype, shape=(72, 3, 3, 72), name="layer2.0.2.layers.3.weight")
    weight12.set_value(np.array([[[[params["layer2.0.2.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(72)] for k in range(3)] for j in range(3)] for i in range(72)]))

    bias12 = ng.variable(dtype=bias_dtype, shape=(72,), name="layer2.0.2.layers.3.bias")
    bias12.set_value(np.round(params["layer2.0.2.layers.3.bias"] / (float) (1 << 7)).astype(params["layer2.0.2.layers.3.bias"].dtype))

    scale12 = ng.variable(dtype=scale_dtype, shape=(72,), name="layer2.0.2.layers.3.scale")
    scale12.set_value(params["layer2.0.2.layers.3.scale"])

    rshift12 = ng.constant([8], dtype=ng.int8)
    act12 = ng.conv2d(act11, weight12, strides=(1, 1, 1, 1), bias=bias12, scale=scale12, rshift_out=rshift12, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [13] conv
    weight13 = ng.variable(dtype=weight_dtype, shape=(24, 1, 1, 72), name="layer2.0.2.layers.6.weight")
    weight13.set_value(params["layer2.0.2.layers.6.weight"])

    bias13 = ng.variable(dtype=bias_dtype, shape=(24,), name="layer2.0.2.layers.6.bias")
    bias13.set_value(np.round(params["layer2.0.2.layers.6.bias"] / (float) (1 << 5)).astype(params["layer2.0.2.layers.6.bias"].dtype))

    scale13 = ng.variable(dtype=scale_dtype, shape=(24,), name="layer2.0.2.layers.6.scale")
    scale13.set_value(params["layer2.0.2.layers.6.scale"])

    rshift13 = ng.constant([14], dtype=ng.int8)
    act13 = ng.conv2d(act12, weight13, strides=(1, 1, 1, 1), bias=bias13, scale=scale13, rshift_out=rshift13, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [14] add
    act14 = ng.add(act13, act10)


    # [15] conv
    weight15 = ng.variable(dtype=weight_dtype, shape=(72, 1, 1, 24), name="layer3.0.0.layers.0.weight")
    weight15.set_value(params["layer3.0.0.layers.0.weight"])

    bias15 = ng.variable(dtype=bias_dtype, shape=(72,), name="layer3.0.0.layers.0.bias")
    bias15.set_value(np.round(params["layer3.0.0.layers.0.bias"] / (float) (1 << 7)).astype(params["layer3.0.0.layers.0.bias"].dtype))

    scale15 = ng.variable(dtype=scale_dtype, shape=(72,), name="layer3.0.0.layers.0.scale")
    scale15.set_value(params["layer3.0.0.layers.0.scale"])

    rshift15 = ng.constant([13], dtype=ng.int8)
    act15 = ng.conv2d(act14, weight15, strides=(1, 1, 1, 1), bias=bias15, scale=scale15, rshift_out=rshift15, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [16] conv
    weight16 = ng.variable(dtype=weight_dtype, shape=(72, 5, 5, 72), name="layer3.0.0.layers.3.weight")
    weight16.set_value(np.array([[[[params["layer3.0.0.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(72)] for k in range(5)] for j in range(5)] for i in range(72)]))

    bias16 = ng.variable(dtype=bias_dtype, shape=(72,), name="layer3.0.0.layers.3.bias")
    bias16.set_value(np.round(params["layer3.0.0.layers.3.bias"] / (float) (1 << 5)).astype(params["layer3.0.0.layers.3.bias"].dtype))

    scale16 = ng.variable(dtype=scale_dtype, shape=(72,), name="layer3.0.0.layers.3.scale")
    scale16.set_value(params["layer3.0.0.layers.3.scale"])

    rshift16 = ng.constant([11], dtype=ng.int8)
    act16 = ng.conv2d(act15, weight16, strides=(1, 2, 2, 1), bias=bias16, scale=scale16, rshift_out=rshift16, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [17] conv
    weight17 = ng.variable(dtype=weight_dtype, shape=(40, 1, 1, 72), name="layer3.0.0.layers.6.weight")
    weight17.set_value(params["layer3.0.0.layers.6.weight"])

    bias17 = ng.variable(dtype=bias_dtype, shape=(40,), name="layer3.0.0.layers.6.bias")
    bias17.set_value(np.round(params["layer3.0.0.layers.6.bias"] / (float) (1 << 5)).astype(params["layer3.0.0.layers.6.bias"].dtype))

    scale17 = ng.variable(dtype=scale_dtype, shape=(40,), name="layer3.0.0.layers.6.scale")
    scale17.set_value(params["layer3.0.0.layers.6.scale"])

    rshift17 = ng.constant([13], dtype=ng.int8)
    act17 = ng.conv2d(act16, weight17, strides=(1, 1, 1, 1), bias=bias17, scale=scale17, rshift_out=rshift17, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [18] conv
    weight18 = ng.variable(dtype=weight_dtype, shape=(120, 1, 1, 40), name="layer3.0.1.layers.0.weight")
    weight18.set_value(params["layer3.0.1.layers.0.weight"])

    bias18 = ng.variable(dtype=bias_dtype, shape=(120,), name="layer3.0.1.layers.0.bias")
    bias18.set_value(np.round(params["layer3.0.1.layers.0.bias"] / (float) (1 << 4)).astype(params["layer3.0.1.layers.0.bias"].dtype))

    scale18 = ng.variable(dtype=scale_dtype, shape=(120,), name="layer3.0.1.layers.0.scale")
    scale18.set_value(params["layer3.0.1.layers.0.scale"])

    rshift18 = ng.constant([13], dtype=ng.int8)
    act18 = ng.conv2d(act17, weight18, strides=(1, 1, 1, 1), bias=bias18, scale=scale18, rshift_out=rshift18, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [19] conv
    weight19 = ng.variable(dtype=weight_dtype, shape=(120, 5, 5, 120), name="layer3.0.1.layers.3.weight")
    weight19.set_value(np.array([[[[params["layer3.0.1.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(120)] for k in range(5)] for j in range(5)] for i in range(120)]))

    bias19 = ng.variable(dtype=bias_dtype, shape=(120,), name="layer3.0.1.layers.3.bias")
    bias19.set_value(np.round(params["layer3.0.1.layers.3.bias"] / (float) (1 << 5)).astype(params["layer3.0.1.layers.3.bias"].dtype))

    scale19 = ng.variable(dtype=scale_dtype, shape=(120,), name="layer3.0.1.layers.3.scale")
    scale19.set_value(params["layer3.0.1.layers.3.scale"])

    rshift19 = ng.constant([9], dtype=ng.int8)
    act19 = ng.conv2d(act18, weight19, strides=(1, 1, 1, 1), bias=bias19, scale=scale19, rshift_out=rshift19, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [20] conv
    weight20 = ng.variable(dtype=weight_dtype, shape=(40, 1, 1, 120), name="layer3.0.1.layers.6.weight")
    weight20.set_value(params["layer3.0.1.layers.6.weight"])

    bias20 = ng.variable(dtype=bias_dtype, shape=(40,), name="layer3.0.1.layers.6.bias")
    bias20.set_value(np.round(params["layer3.0.1.layers.6.bias"] / (float) (1 << 6)).astype(params["layer3.0.1.layers.6.bias"].dtype))

    scale20 = ng.variable(dtype=scale_dtype, shape=(40,), name="layer3.0.1.layers.6.scale")
    scale20.set_value(params["layer3.0.1.layers.6.scale"])

    rshift20 = ng.constant([13], dtype=ng.int8)
    act20 = ng.conv2d(act19, weight20, strides=(1, 1, 1, 1), bias=bias20, scale=scale20, rshift_out=rshift20, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [21] add
    rshift21 = ng.constant([1], dtype=ng.int8)
    act21 = rshift_round_and_clip(ng.add(act20, act17, dtype=mid_dtype), rshift21, dtype=act_dtype)


    # [22] conv
    weight22 = ng.variable(dtype=weight_dtype, shape=(120, 1, 1, 40), name="layer3.0.2.layers.0.weight")
    weight22.set_value(params["layer3.0.2.layers.0.weight"])

    bias22 = ng.variable(dtype=bias_dtype, shape=(120,), name="layer3.0.2.layers.0.bias")
    bias22.set_value(np.round(params["layer3.0.2.layers.0.bias"] / (float) (1 << 4)).astype(params["layer3.0.2.layers.0.bias"].dtype))

    scale22 = ng.variable(dtype=scale_dtype, shape=(120,), name="layer3.0.2.layers.0.scale")
    scale22.set_value(params["layer3.0.2.layers.0.scale"])

    rshift22 = ng.constant([12], dtype=ng.int8)
    act22 = ng.conv2d(act21, weight22, strides=(1, 1, 1, 1), bias=bias22, scale=scale22, rshift_out=rshift22, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [23] conv
    weight23 = ng.variable(dtype=weight_dtype, shape=(120, 5, 5, 120), name="layer3.0.2.layers.3.weight")
    weight23.set_value(np.array([[[[params["layer3.0.2.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(120)] for k in range(5)] for j in range(5)] for i in range(120)]))

    bias23 = ng.variable(dtype=bias_dtype, shape=(120,), name="layer3.0.2.layers.3.bias")
    bias23.set_value(np.round(params["layer3.0.2.layers.3.bias"] / (float) (1 << 6)).astype(params["layer3.0.2.layers.3.bias"].dtype))

    scale23 = ng.variable(dtype=scale_dtype, shape=(120,), name="layer3.0.2.layers.3.scale")
    scale23.set_value(params["layer3.0.2.layers.3.scale"])

    rshift23 = ng.constant([9], dtype=ng.int8)
    act23 = ng.conv2d(act22, weight23, strides=(1, 1, 1, 1), bias=bias23, scale=scale23, rshift_out=rshift23, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [24] conv
    weight24 = ng.variable(dtype=weight_dtype, shape=(40, 1, 1, 120), name="layer3.0.2.layers.6.weight")
    weight24.set_value(params["layer3.0.2.layers.6.weight"])

    bias24 = ng.variable(dtype=bias_dtype, shape=(40,), name="layer3.0.2.layers.6.bias")
    bias24.set_value(np.round(params["layer3.0.2.layers.6.bias"] / (float) (1 << 5)).astype(params["layer3.0.2.layers.6.bias"].dtype))

    scale24 = ng.variable(dtype=scale_dtype, shape=(40,), name="layer3.0.2.layers.6.scale")
    scale24.set_value(params["layer3.0.2.layers.6.scale"])

    rshift24 = ng.constant([13], dtype=ng.int8)
    act24 = ng.conv2d(act23, weight24, strides=(1, 1, 1, 1), bias=bias24, scale=scale24, rshift_out=rshift24, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [25] add
    lshift25 = ng.constant([1], dtype=ng.int8)
    rshift25 = ng.constant([1], dtype=ng.int8)
    act25 = rshift_round_and_clip(ng.add(act24, ng.lshift(act21, lshift25, dtype=mid_dtype)), rshift25, dtype=act_dtype)


    # [26] conv
    weight26 = ng.variable(dtype=weight_dtype, shape=(240, 1, 1, 40), name="layer4.0.0.layers.0.weight")
    weight26.set_value(params["layer4.0.0.layers.0.weight"])

    bias26 = ng.variable(dtype=bias_dtype, shape=(240,), name="layer4.0.0.layers.0.bias")
    bias26.set_value(np.round(params["layer4.0.0.layers.0.bias"] / (float) (1 << 6)).astype(params["layer4.0.0.layers.0.bias"].dtype))

    scale26 = ng.variable(dtype=scale_dtype, shape=(240,), name="layer4.0.0.layers.0.scale")
    scale26.set_value(params["layer4.0.0.layers.0.scale"])

    rshift26 = ng.constant([12], dtype=ng.int8)
    act26 = ng.conv2d(act25, weight26, strides=(1, 1, 1, 1), bias=bias26, scale=scale26, rshift_out=rshift26, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [27] conv
    weight27 = ng.variable(dtype=weight_dtype, shape=(240, 5, 5, 240), name="layer4.0.0.layers.3.weight")
    weight27.set_value(np.array([[[[params["layer4.0.0.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(240)] for k in range(5)] for j in range(5)] for i in range(240)]))

    bias27 = ng.variable(dtype=bias_dtype, shape=(240,), name="layer4.0.0.layers.3.bias")
    bias27.set_value(np.round(params["layer4.0.0.layers.3.bias"] / (float) (1 << 4)).astype(params["layer4.0.0.layers.3.bias"].dtype))

    scale27 = ng.variable(dtype=scale_dtype, shape=(240,), name="layer4.0.0.layers.3.scale")
    scale27.set_value(params["layer4.0.0.layers.3.scale"])

    rshift27 = ng.constant([12], dtype=ng.int8)
    act27 = ng.conv2d(act26, weight27, strides=(1, 2, 2, 1), bias=bias27, scale=scale27, rshift_out=rshift27, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [28] conv
    weight28 = ng.variable(dtype=weight_dtype, shape=(80, 1, 1, 240), name="layer4.0.0.layers.6.weight")
    weight28.set_value(params["layer4.0.0.layers.6.weight"])

    bias28 = ng.variable(dtype=bias_dtype, shape=(80,), name="layer4.0.0.layers.6.bias")
    bias28.set_value(np.round(params["layer4.0.0.layers.6.bias"] / (float) (1 << 6)).astype(params["layer4.0.0.layers.6.bias"].dtype))

    scale28 = ng.variable(dtype=scale_dtype, shape=(80,), name="layer4.0.0.layers.6.scale")
    scale28.set_value(params["layer4.0.0.layers.6.scale"])

    rshift28 = ng.constant([13], dtype=ng.int8)
    act28 = ng.conv2d(act27, weight28, strides=(1, 1, 1, 1), bias=bias28, scale=scale28, rshift_out=rshift28, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [29] conv
    weight29 = ng.variable(dtype=weight_dtype, shape=(480, 1, 1, 80), name="layer4.0.1.layers.0.weight")
    weight29.set_value(params["layer4.0.1.layers.0.weight"])

    bias29 = ng.variable(dtype=bias_dtype, shape=(480,), name="layer4.0.1.layers.0.bias")
    bias29.set_value(np.round(params["layer4.0.1.layers.0.bias"] / (float) (1 << 1)).astype(params["layer4.0.1.layers.0.bias"].dtype))

    scale29 = ng.variable(dtype=scale_dtype, shape=(480,), name="layer4.0.1.layers.0.scale")
    scale29.set_value(params["layer4.0.1.layers.0.scale"])

    rshift29 = ng.constant([13], dtype=ng.int8)
    act29 = ng.conv2d(act28, weight29, strides=(1, 1, 1, 1), bias=bias29, scale=scale29, rshift_out=rshift29, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [30] conv
    weight30 = ng.variable(dtype=weight_dtype, shape=(480, 5, 5, 480), name="layer4.0.1.layers.3.weight")
    weight30.set_value(np.array([[[[params["layer4.0.1.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(480)] for k in range(5)] for j in range(5)] for i in range(480)]))

    bias30 = ng.variable(dtype=bias_dtype, shape=(480,), name="layer4.0.1.layers.3.bias")
    bias30.set_value(np.round(params["layer4.0.1.layers.3.bias"] / (float) (1 << 6)).astype(params["layer4.0.1.layers.3.bias"].dtype))

    scale30 = ng.variable(dtype=scale_dtype, shape=(480,), name="layer4.0.1.layers.3.scale")
    scale30.set_value(params["layer4.0.1.layers.3.scale"])

    rshift30 = ng.constant([7], dtype=ng.int8)
    act30 = ng.conv2d(act29, weight30, strides=(1, 1, 1, 1), bias=bias30, scale=scale30, rshift_out=rshift30, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [31] conv
    weight31 = ng.variable(dtype=weight_dtype, shape=(80, 1, 1, 480), name="layer4.0.1.layers.6.weight")
    weight31.set_value(params["layer4.0.1.layers.6.weight"])

    bias31 = ng.variable(dtype=bias_dtype, shape=(80,), name="layer4.0.1.layers.6.bias")
    bias31.set_value(np.round(params["layer4.0.1.layers.6.bias"] / (float) (1 << 3)).astype(params["layer4.0.1.layers.6.bias"].dtype))

    scale31 = ng.variable(dtype=scale_dtype, shape=(80,), name="layer4.0.1.layers.6.scale")
    scale31.set_value(params["layer4.0.1.layers.6.scale"])

    rshift31 = ng.constant([14], dtype=ng.int8)
    act31 = ng.conv2d(act30, weight31, strides=(1, 1, 1, 1), bias=bias31, scale=scale31, rshift_out=rshift31, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [32] add
    lshift32 = ng.constant([1], dtype=ng.int8)
    rshift32 = ng.constant([1], dtype=ng.int8)
    act32 = rshift_round_and_clip(ng.add(act31, ng.lshift(act28, lshift32, dtype=mid_dtype)), rshift32, dtype=act_dtype)


    # [33] conv
    weight33 = ng.variable(dtype=weight_dtype, shape=(480, 1, 1, 80), name="layer4.0.2.layers.0.weight")
    weight33.set_value(params["layer4.0.2.layers.0.weight"])

    bias33 = ng.variable(dtype=bias_dtype, shape=(480,), name="layer4.0.2.layers.0.bias")
    bias33.set_value(np.round(params["layer4.0.2.layers.0.bias"] / (float) (1 << 4)).astype(params["layer4.0.2.layers.0.bias"].dtype))

    scale33 = ng.variable(dtype=scale_dtype, shape=(480,), name="layer4.0.2.layers.0.scale")
    scale33.set_value(params["layer4.0.2.layers.0.scale"])

    rshift33 = ng.constant([12], dtype=ng.int8)
    act33 = ng.conv2d(act32, weight33, strides=(1, 1, 1, 1), bias=bias33, scale=scale33, rshift_out=rshift33, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [34] conv
    weight34 = ng.variable(dtype=weight_dtype, shape=(480, 5, 5, 480), name="layer4.0.2.layers.3.weight")
    weight34.set_value(np.array([[[[params["layer4.0.2.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(480)] for k in range(5)] for j in range(5)] for i in range(480)]))

    bias34 = ng.variable(dtype=bias_dtype, shape=(480,), name="layer4.0.2.layers.3.bias")
    bias34.set_value(np.round(params["layer4.0.2.layers.3.bias"] / (float) (1 << 6)).astype(params["layer4.0.2.layers.3.bias"].dtype))

    scale34 = ng.variable(dtype=scale_dtype, shape=(480,), name="layer4.0.2.layers.3.scale")
    scale34.set_value(params["layer4.0.2.layers.3.scale"])

    rshift34 = ng.constant([8], dtype=ng.int8)
    act34 = ng.conv2d(act33, weight34, strides=(1, 1, 1, 1), bias=bias34, scale=scale34, rshift_out=rshift34, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [35] conv
    weight35 = ng.variable(dtype=weight_dtype, shape=(80, 1, 1, 480), name="layer4.0.2.layers.6.weight")
    weight35.set_value(params["layer4.0.2.layers.6.weight"])

    bias35 = ng.variable(dtype=bias_dtype, shape=(80,), name="layer4.0.2.layers.6.bias")
    bias35.set_value(np.round(params["layer4.0.2.layers.6.bias"] / (float) (1 << 6)).astype(params["layer4.0.2.layers.6.bias"].dtype))

    scale35 = ng.variable(dtype=scale_dtype, shape=(80,), name="layer4.0.2.layers.6.scale")
    scale35.set_value(params["layer4.0.2.layers.6.scale"])

    rshift35 = ng.constant([13], dtype=ng.int8)
    act35 = ng.conv2d(act34, weight35, strides=(1, 1, 1, 1), bias=bias35, scale=scale35, rshift_out=rshift35, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [36] add
    lshift36 = ng.constant([1], dtype=ng.int8)
    rshift36 = ng.constant([1], dtype=ng.int8)
    act36 = rshift_round_and_clip(ng.add(act35, ng.lshift(act32, lshift36, dtype=mid_dtype)), rshift36, dtype=act_dtype)


    # [37] conv
    weight37 = ng.variable(dtype=weight_dtype, shape=(480, 1, 1, 80), name="layer4.1.0.layers.0.weight")
    weight37.set_value(params["layer4.1.0.layers.0.weight"])

    bias37 = ng.variable(dtype=bias_dtype, shape=(480,), name="layer4.1.0.layers.0.bias")
    bias37.set_value(np.round(params["layer4.1.0.layers.0.bias"] / (float) (1 << 5)).astype(params["layer4.1.0.layers.0.bias"].dtype))

    scale37 = ng.variable(dtype=scale_dtype, shape=(480,), name="layer4.1.0.layers.0.scale")
    scale37.set_value(params["layer4.1.0.layers.0.scale"])

    rshift37 = ng.constant([12], dtype=ng.int8)
    act37 = ng.conv2d(act36, weight37, strides=(1, 1, 1, 1), bias=bias37, scale=scale37, rshift_out=rshift37, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [38] conv
    weight38 = ng.variable(dtype=weight_dtype, shape=(480, 3, 3, 480), name="layer4.1.0.layers.3.weight")
    weight38.set_value(np.array([[[[params["layer4.1.0.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(480)] for k in range(3)] for j in range(3)] for i in range(480)]))

    bias38 = ng.variable(dtype=bias_dtype, shape=(480,), name="layer4.1.0.layers.3.bias")
    bias38.set_value(np.round(params["layer4.1.0.layers.3.bias"] / (float) (1 << 5)).astype(params["layer4.1.0.layers.3.bias"].dtype))

    scale38 = ng.variable(dtype=scale_dtype, shape=(480,), name="layer4.1.0.layers.3.scale")
    scale38.set_value(params["layer4.1.0.layers.3.scale"])

    rshift38 = ng.constant([9], dtype=ng.int8)
    act38 = ng.conv2d(act37, weight38, strides=(1, 1, 1, 1), bias=bias38, scale=scale38, rshift_out=rshift38, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [39] conv
    weight39 = ng.variable(dtype=weight_dtype, shape=(96, 1, 1, 480), name="layer4.1.0.layers.6.weight")
    weight39.set_value(params["layer4.1.0.layers.6.weight"])

    bias39 = ng.variable(dtype=bias_dtype, shape=(96,), name="layer4.1.0.layers.6.bias")
    bias39.set_value(np.round(params["layer4.1.0.layers.6.bias"] / (float) (1 << 5)).astype(params["layer4.1.0.layers.6.bias"].dtype))

    scale39 = ng.variable(dtype=scale_dtype, shape=(96,), name="layer4.1.0.layers.6.scale")
    scale39.set_value(params["layer4.1.0.layers.6.scale"])

    rshift39 = ng.constant([14], dtype=ng.int8)
    act39 = ng.conv2d(act38, weight39, strides=(1, 1, 1, 1), bias=bias39, scale=scale39, rshift_out=rshift39, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [40] conv
    weight40 = ng.variable(dtype=weight_dtype, shape=(576, 1, 1, 96), name="layer4.1.1.layers.0.weight")
    weight40.set_value(params["layer4.1.1.layers.0.weight"])

    bias40 = ng.variable(dtype=bias_dtype, shape=(576,), name="layer4.1.1.layers.0.bias")
    bias40.set_value(np.round(params["layer4.1.1.layers.0.bias"] / (float) (1 << 2)).astype(params["layer4.1.1.layers.0.bias"].dtype))

    scale40 = ng.variable(dtype=scale_dtype, shape=(576,), name="layer4.1.1.layers.0.scale")
    scale40.set_value(params["layer4.1.1.layers.0.scale"])

    rshift40 = ng.constant([13], dtype=ng.int8)
    act40 = ng.conv2d(act39, weight40, strides=(1, 1, 1, 1), bias=bias40, scale=scale40, rshift_out=rshift40, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [41] conv
    weight41 = ng.variable(dtype=weight_dtype, shape=(576, 3, 3, 576), name="layer4.1.1.layers.3.weight")
    weight41.set_value(np.array([[[[params["layer4.1.1.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(576)] for k in range(3)] for j in range(3)] for i in range(576)]))

    bias41 = ng.variable(dtype=bias_dtype, shape=(576,), name="layer4.1.1.layers.3.bias")
    bias41.set_value(np.round(params["layer4.1.1.layers.3.bias"] / (float) (1 << 7)).astype(params["layer4.1.1.layers.3.bias"].dtype))

    scale41 = ng.variable(dtype=scale_dtype, shape=(576,), name="layer4.1.1.layers.3.scale")
    scale41.set_value(params["layer4.1.1.layers.3.scale"])

    rshift41 = ng.constant([7], dtype=ng.int8)
    act41 = ng.conv2d(act40, weight41, strides=(1, 1, 1, 1), bias=bias41, scale=scale41, rshift_out=rshift41, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [42] conv
    weight42 = ng.variable(dtype=weight_dtype, shape=(96, 1, 1, 576), name="layer4.1.1.layers.6.weight")
    weight42.set_value(params["layer4.1.1.layers.6.weight"])

    bias42 = ng.variable(dtype=bias_dtype, shape=(96,), name="layer4.1.1.layers.6.bias")
    bias42.set_value(np.round(params["layer4.1.1.layers.6.bias"] / (float) (1 << 4)).astype(params["layer4.1.1.layers.6.bias"].dtype))

    scale42 = ng.variable(dtype=scale_dtype, shape=(96,), name="layer4.1.1.layers.6.scale")
    scale42.set_value(params["layer4.1.1.layers.6.scale"])

    rshift42 = ng.constant([15], dtype=ng.int8)
    act42 = ng.conv2d(act41, weight42, strides=(1, 1, 1, 1), bias=bias42, scale=scale42, rshift_out=rshift42, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [43] add
    lshift43 = ng.constant([1], dtype=ng.int8)
    rshift43 = ng.constant([1], dtype=ng.int8)
    act43 = rshift_round_and_clip(ng.add(act42, ng.lshift(act39, lshift43, dtype=mid_dtype)), rshift43, dtype=act_dtype)


    # [44] conv
    weight44 = ng.variable(dtype=weight_dtype, shape=(576, 1, 1, 96), name="layer5.0.0.layers.0.weight")
    weight44.set_value(params["layer5.0.0.layers.0.weight"])

    bias44 = ng.variable(dtype=bias_dtype, shape=(576,), name="layer5.0.0.layers.0.bias")
    bias44.set_value(np.round(params["layer5.0.0.layers.0.bias"] / (float) (1 << 2)).astype(params["layer5.0.0.layers.0.bias"].dtype))

    scale44 = ng.variable(dtype=scale_dtype, shape=(576,), name="layer5.0.0.layers.0.scale")
    scale44.set_value(params["layer5.0.0.layers.0.scale"])

    rshift44 = ng.constant([12], dtype=ng.int8)
    act44 = ng.conv2d(act43, weight44, strides=(1, 1, 1, 1), bias=bias44, scale=scale44, rshift_out=rshift44, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [45] conv
    weight45 = ng.variable(dtype=weight_dtype, shape=(576, 5, 5, 576), name="layer5.0.0.layers.3.weight")
    weight45.set_value(np.array([[[[params["layer5.0.0.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(576)] for k in range(5)] for j in range(5)] for i in range(576)]))

    bias45 = ng.variable(dtype=bias_dtype, shape=(576,), name="layer5.0.0.layers.3.bias")
    bias45.set_value(np.round(params["layer5.0.0.layers.3.bias"] / (float) (1 << 4)).astype(params["layer5.0.0.layers.3.bias"].dtype))

    scale45 = ng.variable(dtype=scale_dtype, shape=(576,), name="layer5.0.0.layers.3.scale")
    scale45.set_value(params["layer5.0.0.layers.3.scale"])

    rshift45 = ng.constant([9], dtype=ng.int8)
    act45 = ng.conv2d(act44, weight45, strides=(1, 2, 2, 1), bias=bias45, scale=scale45, rshift_out=rshift45, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [46] conv
    weight46 = ng.variable(dtype=weight_dtype, shape=(192, 1, 1, 576), name="layer5.0.0.layers.6.weight")
    weight46.set_value(params["layer5.0.0.layers.6.weight"])

    bias46 = ng.variable(dtype=bias_dtype, shape=(192,), name="layer5.0.0.layers.6.bias")
    bias46.set_value(np.round(params["layer5.0.0.layers.6.bias"] / (float) (1 << 4)).astype(params["layer5.0.0.layers.6.bias"].dtype))

    scale46 = ng.variable(dtype=scale_dtype, shape=(192,), name="layer5.0.0.layers.6.scale")
    scale46.set_value(params["layer5.0.0.layers.6.scale"])

    rshift46 = ng.constant([15], dtype=ng.int8)
    act46 = ng.conv2d(act45, weight46, strides=(1, 1, 1, 1), bias=bias46, scale=scale46, rshift_out=rshift46, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [47] conv
    weight47 = ng.variable(dtype=weight_dtype, shape=(1152, 1, 1, 192), name="layer5.0.1.layers.0.weight")
    weight47.set_value(params["layer5.0.1.layers.0.weight"])

    bias47 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.0.1.layers.0.bias")
    bias47.set_value(params["layer5.0.1.layers.0.bias"])

    scale47 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.0.1.layers.0.scale")
    scale47.set_value(params["layer5.0.1.layers.0.scale"])

    rshift47 = ng.constant([13], dtype=ng.int8)
    act47 = ng.conv2d(act46, weight47, strides=(1, 1, 1, 1), bias=bias47, scale=scale47, rshift_out=rshift47, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [48] conv
    weight48 = ng.variable(dtype=weight_dtype, shape=(1152, 5, 5, 1152), name="layer5.0.1.layers.3.weight")
    weight48.set_value(np.array([[[[params["layer5.0.1.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(1152)] for k in range(5)] for j in range(5)] for i in range(1152)]))

    bias48 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.0.1.layers.3.bias")
    bias48.set_value(np.round(params["layer5.0.1.layers.3.bias"] / (float) (1 << 7)).astype(params["layer5.0.1.layers.3.bias"].dtype))

    scale48 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.0.1.layers.3.scale")
    scale48.set_value(params["layer5.0.1.layers.3.scale"])

    rshift48 = ng.constant([7], dtype=ng.int8)
    act48 = ng.conv2d(act47, weight48, strides=(1, 1, 1, 1), bias=bias48, scale=scale48, rshift_out=rshift48, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [49] conv
    weight49 = ng.variable(dtype=weight_dtype, shape=(192, 1, 1, 1152), name="layer5.0.1.layers.6.weight")
    weight49.set_value(params["layer5.0.1.layers.6.weight"])

    bias49 = ng.variable(dtype=bias_dtype, shape=(192,), name="layer5.0.1.layers.6.bias")
    bias49.set_value(np.round(params["layer5.0.1.layers.6.bias"] / (float) (1 << 4)).astype(params["layer5.0.1.layers.6.bias"].dtype))

    scale49 = ng.variable(dtype=scale_dtype, shape=(192,), name="layer5.0.1.layers.6.scale")
    scale49.set_value(params["layer5.0.1.layers.6.scale"])

    rshift49 = ng.constant([15], dtype=ng.int8)
    act49 = ng.conv2d(act48, weight49, strides=(1, 1, 1, 1), bias=bias49, scale=scale49, rshift_out=rshift49, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [50] add
    lshift50 = ng.constant([1], dtype=ng.int8)
    rshift50 = ng.constant([1], dtype=ng.int8)
    act50 = rshift_round_and_clip(ng.add(act49, ng.lshift(act46, lshift50, dtype=mid_dtype)), rshift50, dtype=act_dtype)


    # [51] conv
    weight51 = ng.variable(dtype=weight_dtype, shape=(1152, 1, 1, 192), name="layer5.0.2.layers.0.weight")
    weight51.set_value(params["layer5.0.2.layers.0.weight"])

    bias51 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.0.2.layers.0.bias")
    bias51.set_value(np.round(params["layer5.0.2.layers.0.bias"] / (float) (1 << 2)).astype(params["layer5.0.2.layers.0.bias"].dtype))

    scale51 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.0.2.layers.0.scale")
    scale51.set_value(params["layer5.0.2.layers.0.scale"])

    rshift51 = ng.constant([13], dtype=ng.int8)
    act51 = ng.conv2d(act50, weight51, strides=(1, 1, 1, 1), bias=bias51, scale=scale51, rshift_out=rshift51, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [52] conv
    weight52 = ng.variable(dtype=weight_dtype, shape=(1152, 5, 5, 1152), name="layer5.0.2.layers.3.weight")
    weight52.set_value(np.array([[[[params["layer5.0.2.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(1152)] for k in range(5)] for j in range(5)] for i in range(1152)]))

    bias52 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.0.2.layers.3.bias")
    bias52.set_value(np.round(params["layer5.0.2.layers.3.bias"] / (float) (1 << 7)).astype(params["layer5.0.2.layers.3.bias"].dtype))

    scale52 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.0.2.layers.3.scale")
    scale52.set_value(params["layer5.0.2.layers.3.scale"])

    rshift52 = ng.constant([7], dtype=ng.int8)
    act52 = ng.conv2d(act51, weight52, strides=(1, 1, 1, 1), bias=bias52, scale=scale52, rshift_out=rshift52, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [53] conv
    weight53 = ng.variable(dtype=weight_dtype, shape=(192, 1, 1, 1152), name="layer5.0.2.layers.6.weight")
    weight53.set_value(params["layer5.0.2.layers.6.weight"])

    bias53 = ng.variable(dtype=bias_dtype, shape=(192,), name="layer5.0.2.layers.6.bias")
    bias53.set_value(np.round(params["layer5.0.2.layers.6.bias"] / (float) (1 << 5)).astype(params["layer5.0.2.layers.6.bias"].dtype))

    scale53 = ng.variable(dtype=scale_dtype, shape=(192,), name="layer5.0.2.layers.6.scale")
    scale53.set_value(params["layer5.0.2.layers.6.scale"])

    rshift53 = ng.constant([15], dtype=ng.int8)
    act53 = ng.conv2d(act52, weight53, strides=(1, 1, 1, 1), bias=bias53, scale=scale53, rshift_out=rshift53, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [54] add
    lshift54 = ng.constant([1], dtype=ng.int8)
    rshift54 = ng.constant([1], dtype=ng.int8)
    act54 = rshift_round_and_clip(ng.add(act53, ng.lshift(act50, lshift54, dtype=mid_dtype)), rshift54, dtype=act_dtype)


    # [55] conv
    weight55 = ng.variable(dtype=weight_dtype, shape=(1152, 1, 1, 192), name="layer5.0.3.layers.0.weight")
    weight55.set_value(params["layer5.0.3.layers.0.weight"])

    bias55 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.0.3.layers.0.bias")
    bias55.set_value(np.round(params["layer5.0.3.layers.0.bias"] / (float) (1 << 1)).astype(params["layer5.0.3.layers.0.bias"].dtype))

    scale55 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.0.3.layers.0.scale")
    scale55.set_value(params["layer5.0.3.layers.0.scale"])

    rshift55 = ng.constant([14], dtype=ng.int8)
    act55 = ng.conv2d(act54, weight55, strides=(1, 1, 1, 1), bias=bias55, scale=scale55, rshift_out=rshift55, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [56] conv
    weight56 = ng.variable(dtype=weight_dtype, shape=(1152, 5, 5, 1152), name="layer5.0.3.layers.3.weight")
    weight56.set_value(np.array([[[[params["layer5.0.3.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(1152)] for k in range(5)] for j in range(5)] for i in range(1152)]))

    bias56 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.0.3.layers.3.bias")
    bias56.set_value(np.round(params["layer5.0.3.layers.3.bias"] / (float) (1 << 6)).astype(params["layer5.0.3.layers.3.bias"].dtype))

    scale56 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.0.3.layers.3.scale")
    scale56.set_value(params["layer5.0.3.layers.3.scale"])

    rshift56 = ng.constant([7], dtype=ng.int8)
    act56 = ng.conv2d(act55, weight56, strides=(1, 1, 1, 1), bias=bias56, scale=scale56, rshift_out=rshift56, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [57] conv
    weight57 = ng.variable(dtype=weight_dtype, shape=(192, 1, 1, 1152), name="layer5.0.3.layers.6.weight")
    weight57.set_value(params["layer5.0.3.layers.6.weight"])

    bias57 = ng.variable(dtype=bias_dtype, shape=(192,), name="layer5.0.3.layers.6.bias")
    bias57.set_value(np.round(params["layer5.0.3.layers.6.bias"] / (float) (1 << 4)).astype(params["layer5.0.3.layers.6.bias"].dtype))

    scale57 = ng.variable(dtype=scale_dtype, shape=(192,), name="layer5.0.3.layers.6.scale")
    scale57.set_value(params["layer5.0.3.layers.6.scale"])

    rshift57 = ng.constant([15], dtype=ng.int8)
    act57 = ng.conv2d(act56, weight57, strides=(1, 1, 1, 1), bias=bias57, scale=scale57, rshift_out=rshift57, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [58] add
    lshift58 = ng.constant([1], dtype=ng.int8)
    rshift58 = ng.constant([1], dtype=ng.int8)
    act58 = rshift_round_and_clip(ng.add(act57, ng.lshift(act54, lshift58, dtype=mid_dtype)), rshift58, dtype=act_dtype)


    # [59] conv
    weight59 = ng.variable(dtype=weight_dtype, shape=(1152, 1, 1, 192), name="layer5.1.0.layers.0.weight")
    weight59.set_value(params["layer5.1.0.layers.0.weight"])

    bias59 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.1.0.layers.0.bias")
    bias59.set_value(params["layer5.1.0.layers.0.bias"])

    scale59 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.1.0.layers.0.scale")
    scale59.set_value(params["layer5.1.0.layers.0.scale"])

    rshift59 = ng.constant([14], dtype=ng.int8)
    act59 = ng.conv2d(act58, weight59, strides=(1, 1, 1, 1), bias=bias59, scale=scale59, rshift_out=rshift59, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [60] conv
    weight60 = ng.variable(dtype=weight_dtype, shape=(1152, 3, 3, 1152), name="layer5.1.0.layers.3.weight")
    weight60.set_value(np.array([[[[params["layer5.1.0.layers.3.weight"][i][j][k][0] if i == l else 0 for l in range(1152)] for k in range(3)] for j in range(3)] for i in range(1152)]))

    bias60 = ng.variable(dtype=bias_dtype, shape=(1152,), name="layer5.1.0.layers.3.bias")
    bias60.set_value(np.round(params["layer5.1.0.layers.3.bias"] / (float) (1 << 4)).astype(params["layer5.1.0.layers.3.bias"].dtype))

    scale60 = ng.variable(dtype=scale_dtype, shape=(1152,), name="layer5.1.0.layers.3.scale")
    scale60.set_value(params["layer5.1.0.layers.3.scale"])

    rshift60 = ng.constant([9], dtype=ng.int8)
    act60 = ng.conv2d(act59, weight60, strides=(1, 1, 1, 1), bias=bias60, scale=scale60, rshift_out=rshift60, act_func=ng.relu, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    # [61] conv
    weight61 = ng.variable(dtype=weight_dtype, shape=(320, 1, 1, 1152), name="layer5.1.0.layers.6.weight")
    weight61.set_value(params["layer5.1.0.layers.6.weight"])

    bias61 = ng.variable(dtype=bias_dtype, shape=(320,), name="layer5.1.0.layers.6.bias")
    bias61.set_value(np.round(params["layer5.1.0.layers.6.bias"] / (float) (1 << 4)).astype(params["layer5.1.0.layers.6.bias"].dtype))

    scale61 = ng.variable(dtype=scale_dtype, shape=(320,), name="layer5.1.0.layers.6.scale")
    scale61.set_value(params["layer5.1.0.layers.6.scale"])

    rshift61 = ng.constant([15], dtype=ng.int8)
    act61 = ng.conv2d(act60, weight61, strides=(1, 1, 1, 1), bias=bias61, scale=scale61, rshift_out=rshift61, asymmetric_clip=True, dtype=act_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)


    return act3, act14, act25, act43, act61
