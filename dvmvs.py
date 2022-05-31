from __future__ import absolute_import
from __future__ import print_function

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
from tester import Tester


def prepare_placeholders(batchsize, max_n_measurement_frames, act_dtype):
    print("preparing placeholders...")

    input_layer = ng.placeholder(dtype=act_dtype, shape=(batchsize, 64, 96, 3), name='input_layer')
    measurement_features = [ng.placeholder(dtype=act_dtype, shape=(batchsize, 32, 48, 32), name='measurement_feature%d' % m)
                            for m in range(max_n_measurement_frames)]
    n_measurement_frames = ng.placeholder(dtype=ng.uint8, shape=(1,), name='n_measurement_frames')
    frame_number = ng.placeholder(dtype=ng.uint8, shape=(1,), name='frame_number')
    hidden_state = ng.placeholder(dtype=act_dtype, shape=(batchsize, 2, 3, 512), name='hidden_state')
    cell_state = ng.placeholder(dtype=act_dtype, shape=(batchsize, 2, 3, 512), name='cell_state')

    # feature_list = ["half", "quarter", "one_eight", "one_sixteen"]
    # reference_features = [ng.placeholder(dtype=act_dtype, shape=(batchsize, 32 >> i, 48 >> i, 32), name='feature_%s' % feature_list[i]) for i in range(4)]

    return input_layer, measurement_features, n_measurement_frames, frame_number, hidden_state, cell_state


def prepare_nets(input_layer, measurement_features, n_measurement_frames, frame_number, hidden_state, cell_state):
    print("preparing feature extractor...")
    layers = feature_extractor(input_layer, params)

    print("preparing feature shrinker...")
    reference_features = feature_shrinker(*layers, params)

    print("preparing cost volume fusion...")
    cost_volume = cost_volume_fusion(frame_number, reference_features[0], n_measurement_frames, measurement_features,
                                     inputs["half_K"], inputs["current_pose"], inputs["measurement_poses"])

    print("preparing cost volume encoder...")
    skips = cost_volume_encoder(*reference_features, cost_volume, params)

    print("preparing LSTM fusion...")
    lstm_states = LSTMFusion(skips[-1], hidden_state, cell_state, params)

    print("preparing cost volume decoder...")
    depth_full = cost_volume_decoder(input_layer, *skips[:-1], lstm_states[0], params)

    return layers, reference_features, cost_volume, skips, lstm_states, depth_full


if __name__ == '__main__':
    # weight_dtype = ng.int8
    # bias_dtype = ng.int32
    # scale_dtype = ng.int8
    act_dtype = ng.int16
    batchsize = 1
    max_n_measurement_frames = 2

    base_dir = os.path.dirname(os.path.abspath(__file__))
    params = np.load(os.path.join(base_dir, "params/params.npz"))
    inputs = np.load(os.path.join(base_dir, "params/inputs.npz"))
    outputs = np.load(os.path.join(base_dir, "params/outputs.npz"))

    placeholders = prepare_placeholders(batchsize, max_n_measurement_frames, act_dtype)
    nets = prepare_nets(*placeholders)

    test_module = Tester(inputs, outputs, max_n_measurement_frames, *nets, act_dtype)
    # test_module.test_all(verbose=False)
    test_module.test_one(verbose=True)