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

act_dtype = ng.int32
weight_dtype = ng.int8
bias_dtype = ng.int32
scale_dtype = ng.int8
batchsize = 1

base_dir = os.path.dirname(os.path.abspath(__file__))
params = np.load(os.path.join(base_dir, "params/params.npz"))
inputs = np.load(os.path.join(base_dir, "params/inputs.npz"))
outputs = np.load(os.path.join(base_dir, "params/outputs.npz"))
# mids = np.load(os.path.join(base_dir, "params/mids.npz"))


def prepare_input_value(value, lshift):
    value *= 1 << lshift
    value = np.clip(value, -1 * 2 ** (act_dtype.width - 1) - 1, 2 ** (act_dtype.width - 1))
    return np.round(value.astype(np.float64)).astype(np.int32)


input_layer_value = prepare_input_value(inputs["input"].transpose(0, 2, 3, 1), 12)
measurement_features_value = prepare_input_value(inputs["measurement_features"].transpose(0, 1, 3, 4, 2), 9)
hidden_state_value = prepare_input_value(inputs["hidden_state"].transpose(0, 2, 3, 1), 14-1)
cell_state_value = prepare_input_value(inputs["cell_state"].transpose(0, 2, 3, 1), 12)

n_measurement_frames = measurement_features_value.shape[0]


# input
print("preparing inputs...")
input_layer = ng.placeholder(dtype=act_dtype, shape=(batchsize, 64, 96, 3), name='input_layer')
measurement_features = [ng.placeholder(dtype=act_dtype, shape=(batchsize, 32, 48, 32), name='measurement_feature%d' % m)
                        for m in range(n_measurement_frames)]
hidden_state = ng.placeholder(dtype=act_dtype, shape=(batchsize, 2, 3, 512), name='hidden_state')
cell_state = ng.placeholder(dtype=act_dtype, shape=(batchsize, 2, 3, 512), name='cell_state')

ng_inputs = {}
ng_inputs["input_layer"] = input_layer_value
for m in range(n_measurement_frames):
    ng_inputs["measurement_feature%d" % m] = measurement_features_value[m]
ng_inputs["hidden_state"] = hidden_state_value
ng_inputs["cell_state"] = cell_state_value


feature_list = ["half", "quarter", "one_eight", "one_sixteen"]
feature_shifts = [9, 10, 11, 11]
features_value = [prepare_input_value(outputs["feature_%s" % feature_list[i]].transpose(0, 2, 3, 1), feature_shifts[i]) for i in range(4)]
reference_features = [ng.placeholder(dtype=act_dtype, shape=features_value[i].shape, name='feature_%s' % feature_list[i]) for i in range(4)]
for i in range(4):
    ng_inputs["feature_%s" % feature_list[i]] = features_value[i]


# print("preparing feature extractor...")
# layers = feature_extractor(input_layer, params)

# print("preparing feature shrinker...")
# reference_features = feature_shrinker(*layers, params)

print("preparing cost volume fusion...")
cost_volume = cost_volume_fusion(reference_features[0], measurement_features, inputs["warpings"], n_measurement_frames)

print("preparing cost volume encoder...")
skips = cost_volume_encoder(*reference_features, cost_volume, params)

print("preparing LSTM fusion...")
lstm_states = LSTMFusion(skips[-1], hidden_state, cell_state, params)

print("evaluating...")
# eval_outs = ng.eval(layers + reference_features[::-1] + (cost_volume,) + skips + lstm_states[::-1], **ng_inputs)
eval_outs = ng.eval((cost_volume,) + skips + lstm_states[::-1], **ng_inputs)


files = ["layer1", "layer2", "layer3", "layer4", "layer5",
         "feature_one_sixteen", "feature_one_eight", "feature_quarter", "feature_half",
         "cost_volume",
         "skip0", "skip1", "skip2", "skip3", "bottom",
         "cell_state", "hidden_state"]
shifts = [11, 11, 11, 12, 13,
          11, 11, 10, 9,
          7,
          13, 13, 13, 12, 13,
          12, 14]
files = files[9:]
shifts = shifts[9:]
for i in range(len(eval_outs)):
    print(files[i], outputs[files[i]].shape)
    output_layer_value = eval_outs[i].transpose(0, 3, 1, 2) / (1 << shifts[i])
    print(np.mean(output_layer_value.reshape(-1)), np.std(output_layer_value.reshape(-1)))
    print(np.mean(outputs[files[i]].reshape(-1)), np.std(outputs[files[i]].reshape(-1)))
    print(np.corrcoef(output_layer_value.reshape(-1), outputs[files[i]].reshape(-1))[0, 1])
    print("--------------------------")
