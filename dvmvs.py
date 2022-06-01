from __future__ import absolute_import
from __future__ import print_function

import time
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

from verify import Verifier
from simulate import Simulator


def prepare_placeholders(batchsize, max_n_measurement_frames, act_dtype):
    print("preparing placeholders...")

    reference_image = ng.placeholder(dtype=act_dtype, shape=(batchsize, 64, 96, 3), name='reference_image')
    measurement_features = [ng.placeholder(dtype=act_dtype, shape=(batchsize, 32, 48, 32), name='measurement_feature%d' % m)
                            for m in range(max_n_measurement_frames)]
    n_measurement_frames = ng.placeholder(dtype=ng.uint8, shape=(1,), name='n_measurement_frames')
    frame_number = ng.placeholder(dtype=ng.uint8, shape=(1,), name='frame_number')
    hidden_state = ng.placeholder(dtype=act_dtype, shape=(batchsize, 2, 3, 512), name='hidden_state')
    cell_state = ng.placeholder(dtype=act_dtype, shape=(batchsize, 2, 3, 512), name='cell_state')

    # feature_list = ["half", "quarter", "one_eight", "one_sixteen"]
    # reference_features = [ng.placeholder(dtype=act_dtype, shape=(batchsize, 32 >> i, 48 >> i, 32), name='feature_%s' % feature_list[i]) for i in range(4)]

    return reference_image, measurement_features, n_measurement_frames, frame_number, hidden_state, cell_state


def prepare_nets(reference_image, measurement_features, n_measurement_frames, frame_number, hidden_state, cell_state):
    print("preparing feature extractor...")
    start_time = time.process_time()
    layers = feature_extractor(reference_image, params)
    print("\t%f [s]" % (time.process_time() - start_time))

    print("preparing feature shrinker...")
    start_time = time.process_time()
    reference_features = feature_shrinker(*layers, params)
    print("\t%f [s]" % (time.process_time() - start_time))

    print("preparing cost volume fusion...")
    start_time = time.process_time()
    cost_volume = cost_volume_fusion(frame_number, reference_features[0], n_measurement_frames, measurement_features,
                                     inputs["half_K"], inputs["current_pose"], inputs["measurement_poses"])
    print("\t%f [s]" % (time.process_time() - start_time))

    print("preparing cost volume encoder...")
    start_time = time.process_time()
    skips = cost_volume_encoder(*reference_features, cost_volume, params)
    print("\t%f [s]" % (time.process_time() - start_time))

    print("preparing LSTM fusion...")
    start_time = time.process_time()
    lstm_states = LSTMFusion(skips[-1], hidden_state, cell_state, params)
    print("\t%f [s]" % (time.process_time() - start_time))

    print("preparing cost volume decoder...")
    start_time = time.process_time()
    depth_full = cost_volume_decoder(reference_image, *skips[:-1], lstm_states[0], params)
    print("\t%f [s]" % (time.process_time() - start_time))

    return layers, reference_features, cost_volume, skips, lstm_states, depth_full


if __name__ == '__main__':
    # weight_dtype = ng.int8
    # bias_dtype = ng.int32
    # scale_dtype = ng.int8
    act_dtype = ng.int16
    batchsize = 1
    max_n_measurement_frames = 2
    project_name = "dvmvs"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    params = np.load(os.path.join(base_dir, "params/params.npz"))
    inputs = np.load(os.path.join(base_dir, "params/inputs.npz"))
    outputs = np.load(os.path.join(base_dir, "params/outputs.npz"))

    input_layers = prepare_placeholders(batchsize, max_n_measurement_frames, act_dtype)
    nets = prepare_nets(*input_layers)
    output_layer = nets[-1]

    verifier = Verifier(inputs, outputs, max_n_measurement_frames, act_dtype)
    input_layer_values, output_layer_value = verifier.verify_all(*nets, verbose=False)
    input_layer_values, output_layer_value = verifier.verify_one(*nets, verbose=True)


    # Convert the NNgen dataflow to a hardware description (Verilog HDL and IP-XACT)
    print("converting NNgen dataflow to hardware description...")
    silent = True
    axi_datawidth = 32

    # to IP-XACT (the method returns Veriloggen object, as well as to_veriloggen)
    start_time = time.process_time()
    targ = ng.to_ipxact([output_layer], project_name, silent=silent,
                        config={'maxi_datawidth': axi_datawidth})
    print("\t%f [s]" % (time.process_time() - start_time))
    print('# IP-XACT was generated. Check the current directory.')


    # Save the quantized weights
    print("saving weights...")
    # convert weight values to a memory image:
    # on a real FPGA platform, this image will be used as a part of the model definition.
    start_time = time.process_time()
    param_filename = os.path.join(base_dir, 'params/%s_nngen' % project_name)
    chunk_size = 64
    param_data = ng.export_ndarray([output_layer], chunk_size)
    np.savez_compressed(param_filename, param_data)
    print('# weights was saved at %s' % param_filename)
    print("\t%f [s]" % (time.process_time() - start_time))


    # simulator = Simulator(project_name, targ, param_data, axi_datawidth, chunk_size, act_dtype)
    # simulator.simulate(input_layers, input_layer_values, output_layer, output_layer_value)
