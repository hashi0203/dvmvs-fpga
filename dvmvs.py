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

print(inputs["measurement_features"][0][0][0])

input_scale_factor = 1 << 12
input_layer_value = inputs["input"].transpose(0, 2, 3, 1)
input_layer_value = input_layer_value * input_scale_factor
input_layer_value = np.clip(input_layer_value,
                            -1 * 2 ** (act_dtype.width - 1) - 1, 2 ** (act_dtype.width - 1))
input_layer_value = np.round(input_layer_value.astype(np.float64)).astype(np.int32)

features_scale_factor = 1 << 9
measurement_features_value = inputs["measurement_features"].transpose(0, 1, 3, 4, 2)
measurement_features_value = measurement_features_value * features_scale_factor
measurement_features_value = np.clip(measurement_features_value,
                                     -1 * 2 ** (act_dtype.width - 1) - 1, 2 ** (act_dtype.width - 1))
measurement_features_value = np.round(measurement_features_value.astype(np.float64)).astype(np.int32)

n_measurement_frames = measurement_features_value.shape[0]

# # input
print("preparing inputs...")
input_layer = ng.placeholder(dtype=act_dtype,
                             shape=(batchsize, 64, 96, 3),  # N, H, W, C
                             name='input_layer')
measurement_features = [ng.placeholder(dtype=act_dtype, shape=(batchsize, 32, 48, 32), name='measurement_feature%d' % m)
                        for m in range(n_measurement_frames)]

ng_inputs = {}
ng_inputs["input_layer"] = input_layer_value
for m in range(n_measurement_frames):
    ng_inputs["measurement_feature%d" % m] = measurement_features_value[m]


print("preparing feature extractor...")
layers = feature_extractor(input_layer, params)

print("preparing feature shrinker...")
reference_features = feature_shrinker(*layers, params)

print("preparing cost volume fusion...")
cost_volume = cost_volume_fusion(reference_features[0], measurement_features, inputs["warpings"], n_measurement_frames=1)

print("evaluating...")
eval_outs = ng.eval(layers + reference_features[::-1] + (cost_volume,), **ng_inputs)


files = ["layer1", "layer2", "layer3", "layer4", "layer5",
         "feature_one_sixteen", "feature_one_eight", "feature_quarter", "feature_half",
         "cost_volume"]
shifts = [11, 11, 11, 12, 13,
          11, 11, 10, 9,
          7]
for i in range(len(eval_outs)):
    print(files[i], outputs[files[i]].shape)
    output_layer_value = eval_outs[i].transpose(0, 3, 1, 2) / (1 << shifts[i])
    print(np.mean(output_layer_value.reshape(-1)), np.mean(outputs[files[i]].reshape(-1)))
    print(np.corrcoef(output_layer_value.reshape(-1), outputs[files[i]].reshape(-1))[0, 1])
    print("--------------------------")
