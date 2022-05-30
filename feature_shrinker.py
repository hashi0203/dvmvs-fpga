import numpy as np
import nngen as ng
from utils import rshift_round_and_clip, interpolate

def feature_shrinker(act3, act14, act25, act43, act61, params,
                     weight_dtype=ng.int8, bias_dtype=ng.int32, act_dtype=ng.int16, mid_dtype=ng.int32):

    # [62] conv
    weight62 = ng.variable(dtype=weight_dtype, shape=(32, 1, 1, 320), name="fpn.inner_blocks.4.weight")
    weight62.set_value(params["fpn.inner_blocks.4.weight"])

    bias62 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.inner_blocks.4.bias")
    bias62.set_value(np.round(params["fpn.inner_blocks.4.bias"] / (float) (1 << 7)).astype(params["fpn.inner_blocks.4.bias"].dtype))

    conv62 = ng.conv2d(act61, weight62, strides=(1, 1, 1, 1), bias=bias62, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift62 = ng.constant([9], dtype=ng.int8)
    act62 = rshift_round_and_clip(conv62, rshift62, dtype=act_dtype)


    # [63] interpolate
    act63 = ng.extern([act62], shape=(1, 4, 6, 32), opcode=0x63, func=interpolate(4, 6, 0, "nearest"))


    # [64] conv
    weight64 = ng.variable(dtype=weight_dtype, shape=(32, 1, 1, 96), name="fpn.inner_blocks.3.weight")
    weight64.set_value(params["fpn.inner_blocks.3.weight"])

    bias64 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.inner_blocks.3.bias")
    bias64.set_value(np.round(params["fpn.inner_blocks.3.bias"] / (float) (1 << 9)).astype(params["fpn.inner_blocks.3.bias"].dtype))

    conv64 = ng.conv2d(act43, weight64, strides=(1, 1, 1, 1), bias=bias64, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift64 = ng.constant([8], dtype=ng.int8)
    act64 = rshift_round_and_clip(conv64, rshift64, dtype=act_dtype)


    # [65] add
    lshift65 = ng.constant([1], dtype=ng.int8)
    rshift65 = ng.constant([1], dtype=ng.int8)
    act65 = rshift_round_and_clip(ng.add(ng.lshift(act64, lshift65, dtype=mid_dtype), act63), rshift65, dtype=act_dtype)


    # [66] conv
    weight66 = ng.variable(dtype=weight_dtype, shape=(32, 3, 3, 32), name="fpn.layer_blocks.3.weight")
    weight66.set_value(params["fpn.layer_blocks.3.weight"])

    bias66 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.layer_blocks.3.bias")
    bias66.set_value(np.round(params["fpn.layer_blocks.3.bias"] / (float) (1 << 9)).astype(params["fpn.layer_blocks.3.bias"].dtype))

    conv66 = ng.conv2d(act65, weight66, strides=(1, 1, 1, 1), bias=bias66, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift66 = ng.constant([9], dtype=ng.int8)
    act66 = rshift_round_and_clip(conv66, rshift66, dtype=act_dtype)


    # [67] interpolate
    act67 = ng.extern([act65], shape=(1, 8, 12, 32), opcode=0x67, func=interpolate(8, 12, 0, "nearest"))


    # [68] conv
    weight68 = ng.variable(dtype=weight_dtype, shape=(32, 1, 1, 40), name="fpn.inner_blocks.2.weight")
    weight68.set_value(params["fpn.inner_blocks.2.weight"])

    bias68 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.inner_blocks.2.bias")
    bias68.set_value(np.round(params["fpn.inner_blocks.2.bias"] / (float) (1 << 10)).astype(params["fpn.inner_blocks.2.bias"].dtype))

    conv68 = ng.conv2d(act25, weight68, strides=(1, 1, 1, 1), bias=bias68, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift68 = ng.constant([8], dtype=ng.int8)
    act68 = rshift_round_and_clip(conv68, rshift68, dtype=act_dtype)


    # [69] add
    lshift69 = ng.constant([1], dtype=ng.int8)
    rshift69 = ng.constant([1], dtype=ng.int8)
    act69 = rshift_round_and_clip(ng.add(ng.lshift(act68, lshift69, dtype=mid_dtype), act67), rshift69, dtype=act_dtype)


    # [70] conv
    weight70 = ng.variable(dtype=weight_dtype, shape=(32, 3, 3, 32), name="fpn.layer_blocks.2.weight")
    weight70.set_value(params["fpn.layer_blocks.2.weight"])

    bias70 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.layer_blocks.2.bias")
    bias70.set_value(np.round(params["fpn.layer_blocks.2.bias"] / (float) (1 << 9)).astype(params["fpn.layer_blocks.2.bias"].dtype))

    conv70 = ng.conv2d(act69, weight70, strides=(1, 1, 1, 1), bias=bias70, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift70 = ng.constant([8], dtype=ng.int8)
    act70 = rshift_round_and_clip(conv70, rshift70, dtype=act_dtype)


    # [71] interpolate
    act71 = ng.extern([act69], shape=(1, 16, 24, 32), opcode=0x71, func=interpolate(16, 24, 0, "nearest"))


    # [72] conv
    weight72 = ng.variable(dtype=weight_dtype, shape=(32, 1, 1, 24), name="fpn.inner_blocks.1.weight")
    weight72.set_value(params["fpn.inner_blocks.1.weight"])

    bias72 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.inner_blocks.1.bias")
    bias72.set_value(np.round(params["fpn.inner_blocks.1.bias"] / (float) (1 << 11)).astype(params["fpn.inner_blocks.1.bias"].dtype))

    conv72 = ng.conv2d(act14, weight72, strides=(1, 1, 1, 1), bias=bias72, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift72 = ng.constant([8], dtype=ng.int8)
    act72 = rshift_round_and_clip(conv72, rshift72, dtype=act_dtype)


    # [73] add
    lshift73 = ng.constant([1], dtype=ng.int8)
    rshift73 = ng.constant([1], dtype=ng.int8)
    act73 = rshift_round_and_clip(ng.add(ng.lshift(act72, lshift73, dtype=mid_dtype), act71), rshift73, dtype=act_dtype)


    # [74] conv
    weight74 = ng.variable(dtype=weight_dtype, shape=(32, 3, 3, 32), name="fpn.layer_blocks.1.weight")
    weight74.set_value(params["fpn.layer_blocks.1.weight"])

    bias74 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.layer_blocks.1.bias")
    bias74.set_value(np.round(params["fpn.layer_blocks.1.bias"] / (float) (1 << 9)).astype(params["fpn.layer_blocks.1.bias"].dtype))

    conv74 = ng.conv2d(act73, weight74, strides=(1, 1, 1, 1), bias=bias74, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift74 = ng.constant([9], dtype=ng.int8)
    act74 = rshift_round_and_clip(conv74, rshift74, dtype=act_dtype)


    # [75] interpolate
    act75 = ng.extern([act73], shape=(1, 32, 48, 32), opcode=0x75, func=interpolate(32, 48, 0, "nearest"))


    # [76] conv
    weight76 = ng.variable(dtype=weight_dtype, shape=(32, 1, 1, 16), name="fpn.inner_blocks.0.weight")
    weight76.set_value(params["fpn.inner_blocks.0.weight"])

    bias76 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.inner_blocks.0.bias")
    bias76.set_value(np.round(params["fpn.inner_blocks.0.bias"] / (float) (1 << 11)).astype(params["fpn.inner_blocks.0.bias"].dtype))

    conv76 = ng.conv2d(act3, weight76, strides=(1, 1, 1, 1), bias=bias76, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift76 = ng.constant([7], dtype=ng.int8)
    act76 = rshift_round_and_clip(conv76, rshift76, dtype=act_dtype)


    # [77] add
    lshift77 = ng.constant([1], dtype=ng.int8)
    rshift77 = ng.constant([1], dtype=ng.int8)
    act77 = rshift_round_and_clip(ng.add(act76, ng.lshift(act75, lshift77, dtype=mid_dtype)), rshift77, dtype=act_dtype)


    # [78] conv
    weight78 = ng.variable(dtype=weight_dtype, shape=(32, 3, 3, 32), name="fpn.layer_blocks.0.weight")
    weight78.set_value(params["fpn.layer_blocks.0.weight"])

    bias78 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.layer_blocks.0.bias")
    bias78.set_value(np.round(params["fpn.layer_blocks.0.bias"] / (float) (1 << 11)).astype(params["fpn.layer_blocks.0.bias"].dtype))

    conv78 = ng.conv2d(act77, weight78, strides=(1, 1, 1, 1), bias=bias78, asymmetric_clip=True, dtype=mid_dtype, mul_dtype=mid_dtype, sum_dtype=mid_dtype)

    rshift78 = ng.constant([9], dtype=ng.int8)
    act78 = rshift_round_and_clip(conv78, rshift78, dtype=act_dtype)


    return act78, act74, act70, act66
