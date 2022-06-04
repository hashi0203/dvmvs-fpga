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
    frame_number = ng.placeholder(dtype=act_dtype, shape=(1,), name='frame_number')
    n_measurement_frames = ng.placeholder(dtype=act_dtype, shape=(1,), name='n_measurement_frames')
    measurement_features = [ng.placeholder(dtype=act_dtype, shape=(batchsize, 32, 48, 32), name='measurement_feature%d' % m)
                            for m in range(max_n_measurement_frames)]
    hidden_state = ng.placeholder(dtype=act_dtype, shape=(batchsize, 2, 3, 512), name='hidden_state')
    cell_state = ng.placeholder(dtype=act_dtype, shape=(batchsize, 2, 3, 512), name='cell_state')

    # feature_list = ["half", "quarter", "one_eight", "one_sixteen"]
    # reference_features = [ng.placeholder(dtype=act_dtype, shape=(batchsize, 32 >> i, 48 >> i, 32), name='feature_%s' % feature_list[i]) for i in range(4)]

    return reference_image, frame_number, n_measurement_frames, measurement_features, hidden_state, cell_state


def prepare_nets(reference_image, frame_number, n_measurement_frames, measurement_features, hidden_state, cell_state,
                 pars, dtypes):

    externs = []

    print("preparing feature extractor...")
    layers = feature_extractor(reference_image, params, **pars, **dtypes)

    print("preparing feature shrinker...")
    reference_features, extern = feature_shrinker(*layers, params, **pars, **dtypes)
    externs.extend(extern)

    print("preparing cost volume fusion...")
    cost_volume, extern = cost_volume_fusion(frame_number, reference_features[0], n_measurement_frames, measurement_features,
                                             inputs["half_K"], inputs["current_pose"], inputs["measurement_poses"], dtypes["act_dtype"])
    externs.extend(extern)

    print("preparing cost volume encoder...")
    skips = cost_volume_encoder(*reference_features, *cost_volume, params, **pars, **dtypes)

    print("preparing LSTM fusion...")
    lstm_states, extern = LSTMFusion(skips[-1], hidden_state, cell_state, params, **pars, **dtypes)
    externs.extend(extern)

    print("preparing cost volume decoder...")
    depth_full, extern = cost_volume_decoder(reference_image, *skips[:-1], lstm_states[0], params, **pars, **dtypes)
    externs.extend(extern)

    return (layers, reference_features, cost_volume, skips, lstm_states, depth_full), externs


if __name__ == '__main__':
    batchsize = 1
    max_n_measurement_frames = 2
    project_name = "dvmvs"

    par_ich = 1
    par_och = 1
    par = par_och
    pars = {"par_ich": par_ich, "par_och": par_och, "par": par}

    weight_dtype = ng.int8
    bias_dtype = ng.int32
    scale_dtype = ng.int8
    act_dtype = ng.int16
    mid_dtype = ng.int32
    dtypes = {"weight_dtype": weight_dtype, "bias_dtype": bias_dtype,
              "scale_dtype": scale_dtype, "act_dtype": act_dtype, "mid_dtype": mid_dtype}

    base_dir = os.path.dirname(os.path.abspath(__file__))
    params = np.load(os.path.join(base_dir, "params/params.npz"))
    inputs = np.load(os.path.join(base_dir, "params/inputs.npz"))
    outputs = np.load(os.path.join(base_dir, "params/outputs.npz"))

    start_time = time.process_time()
    input_layers = prepare_placeholders(batchsize, max_n_measurement_frames, act_dtype)
    nets, externs = prepare_nets(*input_layers, pars, dtypes)
    output_layer = nets[-1][0]
    print("\t%f [s]" % (time.process_time() - start_time))


    skip_verify = False
    input_filename = os.path.join(base_dir, 'params_nngen/inputs.npz')
    output_filename = os.path.join(base_dir, 'params_nngen/outputs.npz')
    if skip_verify:
        print("loading input and output values...")
        input_file = np.load(input_filename)
        input_layer_values = {}
        for f in input_file.files:
            input_layer_values[f] = input_file[f]
        output_file = np.load(output_filename)
        output_layer_value = output_file[output_file.files[0]]
    else:
        print("verifying...")
        verifier = Verifier(inputs, outputs, max_n_measurement_frames, act_dtype)
        # input_layer_values, output_layer_value, output_layers = verifier.verify_all(*nets, verbose=False)
        input_layer_values, output_layer_values, output_layers = verifier.verify_one(*nets, verbose=True)
        np.savez_compressed(input_filename, **input_layer_values)
        np.savez_compressed(output_filename, **output_layer_values)


    skip_to_ipxact = False
    axi_datawidth = 128
    if skip_to_ipxact:
        print("skiping to_ipxact...")
    else:
        # Convert the NNgen dataflow to a hardware description (Verilog HDL and IP-XACT)
        print("converting NNgen dataflow to hardware description...")
        # to IP-XACT (the method returns Veriloggen object, as well as to_veriloggen)
        start_time = time.process_time()
        targ = ng.to_ipxact(output_layers, project_name, silent=False,
                            config={'maxi_datawidth': axi_datawidth})
        print("\t%f [s]" % (time.process_time() - start_time))
        print('# IP-XACT was generated. Check the current directory.')


    skip_export = True
    param_filename = os.path.join(base_dir, 'params_nngen/params.npz')
    chunk_size = 64 # should not be changed
    if skip_export:
        print("loading params...")
        param_file = np.load(param_filename)
        param_data = param_file[param_file.files[0]]
    else:
        # Save the quantized weights
        print("saving params...")
        # convert weight values to a memory image:
        # on a real FPGA platform, this image will be used as a part of the model definition.
        start_time = time.process_time()
        param_data = ng.export_ndarray([output_layer], chunk_size)
        np.savez_compressed(param_filename, param_data)
        print('# weights was saved at %s' % param_filename)
        print("\t%f [s]" % (time.process_time() - start_time))


    flat_input_layers = []
    for layer in input_layers:
        if isinstance(layer, list):
            flat_input_layers.extend(layer)
        else:
            flat_input_layers.append(layer)

    measurement_names = ["measurement_feature%d" % m for m in range(max_n_measurement_frames)]
    input_names = ["reference_image", "frame_number", "n_measurement_frames"] + measurement_names + ["hidden_state", "cell_state"]
    flat_input_layer_values = []
    for name in input_names:
        if isinstance(input_layer_values[name], list):
            flat_input_layer_values.extend(input_layer_values[name])
        else:
            flat_input_layer_values.append(input_layer_values[name])

    for name, layer in zip(input_names, flat_input_layers):
        print("%20s: %6d," % (name, layer.addr), layer.aligned_shape)

    for extern in externs:
        print(extern[-1])
        print("\toutput: addr %d, shape" % extern[0].addr, extern[0].shape, ", aligned_shape", extern[0].aligned_shape)
        for i, e in enumerate(extern[1]):
            print("\tinput%d: addr %d, shape" % (i, e.addr), e.shape, ", aligned_shape", e.aligned_shape)

    print("simulating verilog code...")
    start_time = time.process_time()
    simulator = Simulator(project_name, targ, param_data, axi_datawidth, chunk_size, par_ich, par_och, act_dtype)
    simulator.simulate(flat_input_layers, flat_input_layer_values, output_layer, output_layer_value)
    print("\t%f [s]" % (time.process_time() - start_time))
