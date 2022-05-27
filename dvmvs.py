from __future__ import absolute_import
from __future__ import print_function

import sys
import os
from xml.sax.handler import feature_external_ges

import numpy as np
import nngen as ng
from feature_extractor import feature_extractor
from feature_shrinker import feature_shrinker

act_dtype = ng.int32
weight_dtype = ng.int8
bias_dtype = ng.int32
scale_dtype = ng.int8
batchsize = 1

# # input
input_layer = ng.placeholder(dtype=act_dtype,
                             shape=(batchsize, 64, 96, 3),  # N, H, W, C
                             name='input_layer')

base_dir = os.path.dirname(os.path.abspath(__file__))
params = np.load(os.path.join(base_dir, "params/params.npz"))
inputs = np.load(os.path.join(base_dir, "params/inputs.npz"))
outputs = np.load(os.path.join(base_dir, "params/outputs.npz"))
# mids = np.load(os.path.join(base_dir, "params/mids.npz"))

input_layer_value = inputs["input"].transpose(0, 2, 3, 1)
act_scale_factor = int(2 ** 12)
input_layer_value = input_layer_value * act_scale_factor
input_layer_value = np.clip(input_layer_value,
                            -1 * 2 ** (act_dtype.width - 1) - 1, 2 ** (act_dtype.width - 1))
input_layer_value = np.round(input_layer_value.astype(np.float64)).astype(np.int32)

print("preparing feature extractor...")
layers = feature_extractor(input_layer, params)

print("preparing feature shrinker...")
reference_features = feature_shrinker(*layers, params)

print("evaluating...")
eval_outs = ng.eval(layers + reference_features, input_layer=input_layer_value)


files = ["layer1", "layer2", "layer3", "layer4", "layer5",
         "feature_half", "feature_quarter", "feature_one_eight", "feature_one_sixteen"]
for i in range(len(eval_outs)):
    output_layer_value = eval_outs[i].transpose(0, 3, 1, 2)
    print(np.corrcoef(output_layer_value.reshape(-1), outputs[files[i]].reshape(-1))[0, 1])
