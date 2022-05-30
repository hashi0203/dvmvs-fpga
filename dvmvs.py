from __future__ import absolute_import
from __future__ import print_function

import sys
import os
from xml.sax.handler import feature_external_ges

import numpy as np
import nngen as ng

from feature_extractor import feature_extractor
from feature_shrinker import feature_shrinker
from cost_volume_fusion import cost_volume_fusion
from cost_volume_encoder import cost_volume_encoder
from convlstm import LSTMFusion
from cost_volume_decoder import cost_volume_decoder
from utils import lstm_state_calculator

weight_dtype = ng.int8
bias_dtype = ng.int32
scale_dtype = ng.int8
act_dtype = ng.int16
batchsize = 1
max_n_measurement_frames = 2

base_dir = os.path.dirname(os.path.abspath(__file__))
params = np.load(os.path.join(base_dir, "params/params.npz"))
inputs = np.load(os.path.join(base_dir, "params/inputs.npz"))
outputs = np.load(os.path.join(base_dir, "params/outputs.npz"))
# mids = np.load(os.path.join(base_dir, "params/mids.npz"))

# input
print("preparing inputs...")
input_layer = ng.placeholder(dtype=act_dtype, shape=(batchsize, 64, 96, 3), name='input_layer')
measurement_features = [ng.placeholder(dtype=act_dtype, shape=(batchsize, 32, 48, 32), name='measurement_feature%d' % m)
                        for m in range(max_n_measurement_frames)]
n_measurement_frames = ng.placeholder(dtype=ng.uint8, shape=(1,), name='n_measurement_frames')
frame_number = ng.placeholder(dtype=ng.uint8, shape=(1,), name='frame_number')
hidden_state = ng.placeholder(dtype=act_dtype, shape=(batchsize, 2, 3, 512), name='hidden_state')
cell_state = ng.placeholder(dtype=act_dtype, shape=(batchsize, 2, 3, 512), name='cell_state')

# feature_list = ["half", "quarter", "one_eight", "one_sixteen"]
# reference_features = [ng.placeholder(dtype=act_dtype, shape=(batchsize, 32 >> i, 48 >> i, 32), name='feature_%s' % feature_list[i]) for i in range(4)]

print("preparing feature extractor...")
layers = feature_extractor(input_layer, params)

print("preparing feature shrinker...")
reference_features = feature_shrinker(*layers, params)

print("preparing cost volume fusion...")
cost_volume = cost_volume_fusion(frame_number, reference_features[0], n_measurement_frames, measurement_features,
                                 inputs["half_K"], inputs["pose1s"], inputs["pose2ss"])

print("preparing cost volume encoder...")
skips = cost_volume_encoder(*reference_features, cost_volume, params)

print("preparing LSTM fusion...")
lstm_states = LSTMFusion(skips[-1], hidden_state, cell_state, params)

print("preparing cost volume decoder...")
depth_full = cost_volume_decoder(input_layer, *skips[:-1], lstm_states[0], params)


def prepare_input_value(value, lshift):
    value *= 1 << lshift
    value = np.clip(value, -1 * 2 ** (act_dtype.width - 1) - 1, 2 ** (act_dtype.width - 1))
    return np.round(value.astype(np.float64)).astype(np.int32)


lstm_state = None
previous_depth = None
previous_pose = None
calc = lstm_state_calculator(inputs, 14-1, 12)
for n in range(len(inputs["input"])):
    input_layer_value = prepare_input_value(inputs["input"][n].transpose(0, 2, 3, 1), 12)
    measurement_features_value = prepare_input_value(inputs["measurement_features"][n].transpose(0, 1, 3, 4, 2), 9)
    n_measurement_frames_value = np.array([inputs["n_measurement_frames"][n]]).astype(np.uint8)
    frame_number_value = np.array([n]).astype(np.uint8)

    lstm_state = inputs["hidden_state"][n], inputs["cell_state"][n]
    lstm_state = calc(lstm_state, previous_depth, previous_pose, inputs["pose1s"][n])
    hidden_state_value = prepare_input_value(lstm_state[0].transpose(0, 2, 3, 1), 14-1)
    cell_state_value = prepare_input_value(lstm_state[1].transpose(0, 2, 3, 1), 12)

    ng_inputs = {}
    ng_inputs["input_layer"] = input_layer_value
    for m in range(max_n_measurement_frames):
        ng_inputs["measurement_feature%d" % m] = measurement_features_value[m]
    ng_inputs["n_measurement_frames"] = n_measurement_frames_value
    ng_inputs["frame_number"] = frame_number_value
    ng_inputs["hidden_state"] = hidden_state_value
    ng_inputs["cell_state"] = cell_state_value

    # feature_shifts = [9, 10, 11, 11]
    # features_value = [prepare_input_value(outputs["feature_%s" % feature_list[i]].transpose(0, 2, 3, 1), feature_shifts[i]) for i in range(4)]
    # for i in range(4):
    #     ng_inputs["feature_%s" % feature_list[i]] = features_value[i]


    print("evaluating...")
    eval_outs = ng.eval(layers + reference_features[::-1] + (cost_volume,) + skips + lstm_states[::-1] + (depth_full,), **ng_inputs)
    # eval_outs = ng.eval((cost_volume,) + skips + lstm_states[::-1] + (depth_full,), **ng_inputs)

    files = ["layer1", "layer2", "layer3", "layer4", "layer5",
             "feature_one_sixteen", "feature_one_eight", "feature_quarter", "feature_half",
             "cost_volume",
             "skip0", "skip1", "skip2", "skip3", "bottom",
             "cell_state", "hidden_state",
             "depth_org"]
    shifts = [11, 11, 11, 12, 13,
              11, 11, 10, 9,
              7,
              13, 13, 13, 12, 13,
              12, 14,
              14]
    # files = files[9:]
    # shifts = shifts[9:]
    for i in range(len(eval_outs)):
        if i != len(files) - 1:
            continue
        ground_truth = outputs[files[i]][n]
        print(files[i], ground_truth.shape)
        output_layer_value = eval_outs[i].transpose(0, 3, 1, 2) / (1 << shifts[i])
        print(np.mean(output_layer_value.reshape(-1)), np.std(output_layer_value.reshape(-1)))
        print(np.mean(ground_truth.reshape(-1)), np.std(ground_truth.reshape(-1)))
        print(np.corrcoef(output_layer_value.reshape(-1), ground_truth.reshape(-1))[0, 1])
        print("--------------------------")
    print()

    min_depth = 0.25
    max_depth = 20.0
    inverse_depth_base = 1 / max_depth
    inverse_depth_multiplier = 1 / min_depth - 1 / max_depth

    depth_org = (eval_outs[-1].transpose(0, 3, 1, 2) / (1 << shifts[-1])).astype(np.float32)
    inverse_depth_full = inverse_depth_multiplier * depth_org + inverse_depth_base
    previous_depth = 1.0 / inverse_depth_full
    previous_pose = inputs["pose1s"][n].copy()
