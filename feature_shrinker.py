import nngen as ng
from interpolate import interpolate

def feature_shrinker(act3, act14, act25, act43, act61, params,
                     weight_dtype=ng.int8, bias_dtype=ng.int32, scale_dtype=ng.int8, act_dtype=ng.int32):

    # [62] conv
    weight62 = ng.variable(dtype=weight_dtype, shape=(32, 1, 1, 320), name="fpn.inner_blocks.4.weight")
    weight62.set_value(params["fpn.inner_blocks.4.weight"])

    bias62 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.inner_blocks.4.bias")
    bias62.set_value(params["fpn.inner_blocks.4.bias"])

    conv62 = ng.conv2d(act61, weight62, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32)

    lshift62 = ng.constant([1], dtype=ng.int8)
    sum62 = ng.add(ng.lshift(conv62, lshift62), bias62)
    rshift62 = ng.constant([10], dtype=ng.int8)
    act62 = ng.rshift_round(sum62, rshift62)


    # [63] interpolate
    act63 = ng.extern([act62], opcode=0x63, func=interpolate(4, 6, 0, "nearest"))
    act63.shape = (1, 4, 6, 32)


    # [64] conv
    weight64 = ng.variable(dtype=weight_dtype, shape=(32, 1, 1, 96), name="fpn.inner_blocks.3.weight")
    weight64.set_value(params["fpn.inner_blocks.3.weight"])

    bias64 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.inner_blocks.3.bias")
    bias64.set_value(params["fpn.inner_blocks.3.bias"])

    conv64 = ng.conv2d(act43, weight64, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32)

    lshift64 = ng.constant([3], dtype=ng.int8)
    sum64 = ng.add(ng.lshift(conv64, lshift64), bias64)
    rshift64 = ng.constant([11], dtype=ng.int8)
    act64 = ng.rshift_round(sum64, rshift64)


    # [65] add
    lshift65 = ng.constant([1], dtype=ng.int8)
    rshift65 = ng.constant([1], dtype=ng.int8)
    act65 = ng.rshift_round(ng.add(ng.lshift(act64, lshift65), act63), rshift65)


    # [66] conv
    weight66 = ng.variable(dtype=weight_dtype, shape=(32, 3, 3, 32), name="fpn.layer_blocks.3.weight")
    weight66.set_value(params["fpn.layer_blocks.3.weight"])

    bias66 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.layer_blocks.3.bias")
    bias66.set_value(params["fpn.layer_blocks.3.bias"])

    conv66 = ng.conv2d(act65, weight66, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32)

    lshift66 = ng.constant([3], dtype=ng.int8)
    sum66 = ng.add(ng.lshift(conv66, lshift66), bias66)
    rshift66 = ng.constant([12], dtype=ng.int8)
    act66 = ng.rshift_round(sum66, rshift66)


    # [67] interpolate
    act67 = ng.extern([act65], opcode=0x67, func=interpolate(8, 12, 0, "nearest"))
    act67.shape = (1, 8, 12, 32)


    # [68] conv
    weight68 = ng.variable(dtype=weight_dtype, shape=(32, 1, 1, 40), name="fpn.inner_blocks.2.weight")
    weight68.set_value(params["fpn.inner_blocks.2.weight"])

    bias68 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.inner_blocks.2.bias")
    bias68.set_value(params["fpn.inner_blocks.2.bias"])

    conv68 = ng.conv2d(act25, weight68, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32)

    lshift68 = ng.constant([4], dtype=ng.int8)
    sum68 = ng.add(ng.lshift(conv68, lshift68), bias68)
    rshift68 = ng.constant([12], dtype=ng.int8)
    act68 = ng.rshift_round(sum68, rshift68)


    # [69] add
    lshift69 = ng.constant([1], dtype=ng.int8)
    rshift69 = ng.constant([1], dtype=ng.int8)
    act69 = ng.rshift_round(ng.add(ng.lshift(act68, lshift69), act67), rshift69)


    # [70] conv
    weight70 = ng.variable(dtype=weight_dtype, shape=(32, 3, 3, 32), name="fpn.layer_blocks.2.weight")
    weight70.set_value(params["fpn.layer_blocks.2.weight"])

    bias70 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.layer_blocks.2.bias")
    bias70.set_value(params["fpn.layer_blocks.2.bias"])

    conv70 = ng.conv2d(act69, weight70, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32)

    lshift70 = ng.constant([3], dtype=ng.int8)
    sum70 = ng.add(ng.lshift(conv70, lshift70), bias70)
    rshift70 = ng.constant([11], dtype=ng.int8)
    act70 = ng.rshift_round(sum70, rshift70)


    # [71] interpolate
    act71 = ng.extern([act69], opcode=0x71, func=interpolate(16, 24, 0, "nearest"))
    act71.shape = (1, 16, 24, 32)


    # [72] conv
    weight72 = ng.variable(dtype=weight_dtype, shape=(32, 1, 1, 24), name="fpn.inner_blocks.1.weight")
    weight72.set_value(params["fpn.inner_blocks.1.weight"])

    bias72 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.inner_blocks.1.bias")
    bias72.set_value(params["fpn.inner_blocks.1.bias"])

    conv72 = ng.conv2d(act14, weight72, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32)

    lshift72 = ng.constant([5], dtype=ng.int8)
    sum72 = ng.add(ng.lshift(conv72, lshift72), bias72)
    rshift72 = ng.constant([13], dtype=ng.int8)
    act72 = ng.rshift_round(sum72, rshift72)


    # [73] add
    lshift73 = ng.constant([1], dtype=ng.int8)
    rshift73 = ng.constant([1], dtype=ng.int8)
    act73 = ng.rshift_round(ng.add(ng.lshift(act72, lshift73), act71), rshift73)


    # [74] conv
    weight74 = ng.variable(dtype=weight_dtype, shape=(32, 3, 3, 32), name="fpn.layer_blocks.1.weight")
    weight74.set_value(params["fpn.layer_blocks.1.weight"])

    bias74 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.layer_blocks.1.bias")
    bias74.set_value(params["fpn.layer_blocks.1.bias"])

    conv74 = ng.conv2d(act73, weight74, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32)

    lshift74 = ng.constant([3], dtype=ng.int8)
    sum74 = ng.add(ng.lshift(conv74, lshift74), bias74)
    rshift74 = ng.constant([12], dtype=ng.int8)
    act74 = ng.rshift_round(sum74, rshift74)


    # [75] interpolate
    act75 = ng.extern([act73], opcode=0x75, func=interpolate(32, 48, 0, "nearest"))
    act75.shape = (1, 32, 48, 32)


    # [76] conv
    weight76 = ng.variable(dtype=weight_dtype, shape=(32, 1, 1, 16), name="fpn.inner_blocks.0.weight")
    weight76.set_value(params["fpn.inner_blocks.0.weight"])

    bias76 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.inner_blocks.0.bias")
    bias76.set_value(params["fpn.inner_blocks.0.bias"])

    conv76 = ng.conv2d(act3, weight76, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32)

    lshift76 = ng.constant([5], dtype=ng.int8)
    sum76 = ng.add(ng.lshift(conv76, lshift76), bias76)
    rshift76 = ng.constant([12], dtype=ng.int8)
    act76 = ng.rshift_round(sum76, rshift76)


    # [77] add
    lshift77 = ng.constant([1], dtype=ng.int8)
    rshift77 = ng.constant([1], dtype=ng.int8)
    act77 = ng.rshift_round(ng.add(act76, ng.lshift(act75, lshift77)), rshift77)


    # [78] conv
    weight78 = ng.variable(dtype=weight_dtype, shape=(32, 3, 3, 32), name="fpn.layer_blocks.0.weight")
    weight78.set_value(params["fpn.layer_blocks.0.weight"])

    bias78 = ng.variable(dtype=bias_dtype, shape=(32,), name="fpn.layer_blocks.0.bias")
    bias78.set_value(params["fpn.layer_blocks.0.bias"])

    conv78 = ng.conv2d(act77, weight78, strides=(1, 1, 1, 1), dtype=act_dtype, sum_dtype=ng.int32)

    lshift78 = ng.constant([5], dtype=ng.int8)
    sum78 = ng.add(ng.lshift(conv78, lshift78), bias78)
    rshift78 = ng.constant([14], dtype=ng.int8)
    act78 = ng.rshift_round(sum78, rshift78)


    return act78, act74, act70, act66

