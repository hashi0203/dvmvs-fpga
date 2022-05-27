from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import time
from xml.sax.handler import feature_external_ges

import numpy as np
import nngen as ng
from model import FeatureExtractor


# --------------------
# (1) Represent a DNN model as a dataflow by NNgen operators
# --------------------

# data types
act_dtype = ng.int64
# weight_dtype = ng.int8
# bias_dtype = ng.int32
# scale_dtype = ng.int8
batchsize = 1

# # input
input_layer = ng.placeholder(dtype=act_dtype,
                             shape=(batchsize, 64, 96, 3),  # N, H, W, C
                             name='input_layer')
start_time = time.process_time()
feature_extractor = FeatureExtractor(input_layer)
acts, acts_20, layer1, layer2, layer3, layer4, layer5 = feature_extractor.get_output()
end_time = time.process_time()
elapsed_time = end_time - start_time
print(elapsed_time)

# act_scale_factor = 128
# act_scale_factor = 32767
act_scale_factor = int(2 ** 3 - 1)
imagenet_mean = np.array([0, 0, 0]).astype(np.float32)
imagenet_std = np.array([1, 1, 1]).astype(np.float32)
# imagenet_mean = np.array([-0.093373775, -0.021101305, 0.118325084]).astype(np.float32)
# imagenet_std = np.array([1.2635281, 1.2418443, 1.2138414]).astype(np.float32)
input_scale_factors = {'input_layer': act_scale_factor}
input_means = {'input_layer': imagenet_mean * act_scale_factor}
input_stds = {'input_layer': imagenet_std * act_scale_factor}

print("quantizing...")
ng.quantize([layer5], input_scale_factors, input_means, input_stds)
# for aa in feature_extractor.layer1_3.a1s:
#     print(aa.cshamt_mul, aa.cshamt_sum, aa.cshamt_out)


base_dir = os.path.dirname(os.path.abspath(__file__))
inputs = np.load(os.path.join(base_dir, "params_org/inputs.npz"))
outputs = np.load(os.path.join(base_dir, "params_org/outputs.npz"))
mids = np.load(os.path.join(base_dir, "params_org/mids.npz"))

input_layer_value = inputs["input"].transpose(0, 2, 3, 1)

# input_layer_value = np.random.normal(size=input_layer.length).reshape(input_layer.shape)
# input_layer_value = input_layer_value * imagenet_std + imagenet_mean

input_layer_value = np.clip(input_layer_value, -3.0, 3.0)
input_layer_value = input_layer_value * act_scale_factor
input_layer_value = np.clip(input_layer_value,
                            -1 * 2 ** (act_dtype.width - 1) - 1, 2 ** (act_dtype.width - 1))
input_layer_value = np.round(input_layer_value.astype(np.float64)).astype(np.int64)

# from conv import Conv2d
# layer1_0 = Conv2d(input_layer, "layer1.0", 3, 32,
#                     kernel_size=3, stride=2, groups=1,
#                     act_func=ng.relu)
# a1 = layer1_0.get_output()
# ng.quantize([a1], input_scale_factors, input_means, input_stds)

print("evaluating...")
eval_outs = ng.eval(acts + acts_20 + [layer2], input_layer=input_layer_value)
# output_layer_value = eval_outs[0].transpose(0, 3, 1, 2)
# eval_outs = ng.eval([layer2], input_layer=input_layer_value)
# output_layer_value = np.array(eval_outs[0]).transpose(0, 3, 1, 2)

# print(np.sum(output_layer_value) / output_layer_value.size, np.std(output_layer_value.reshape(-1)))
# aa = (mids["a1"].transpose(0, 2, 3, 1) * 1370).astype('int32')
# print(np.sum(aa) / aa.size, np.std(aa.reshape(-1)))
# print(output_layer_value.shape)
# print(mids["a34"].shape)
# for i in range(mids["a34"].shape[1]):
#     if i == 0: print(output_layer_value[0, i])
#     if i == 0: print(mids["a34"][0, i])
#     print(np.corrcoef(output_layer_value[0, i].reshape(-1), mids["a34"][0, i].reshape(-1))[0, 1])


# params = np.load(os.path.join(base_dir, "params/params_scale.npz"))
# weight = params["layer1.0.weight"]
# bias = params["layer1.0.bias"]
# scale = params["layer1.0.scale"]
# import torch
# out_channels, in_channels, kernel_size, _ = weight.shape
# # weight = torch.tensor(filter.transpose(0, 3, 1, 2))
# conv = torch.nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=True)
# conv.weight = torch.nn.Parameter(torch.tensor(weight))
# # print(conv.weight)
# # print(inputs["input"].shape)
# if bias is not None:
#     conv.bias = torch.nn.Parameter(torch.tensor(bias))
#     # print(conv.bias)
# out = conv(torch.tensor(inputs["input"])).detach().numpy().copy()
# if scale is not None:
#     out *= scale[None, :, None, None]
#     # print(scale)
# m = torch.nn.ReLU()
# out = m(torch.tensor(out)).detach().numpy().copy()

# print(out.shape)
# print(output_layer_value.shape, out.shape)
# print(np.corrcoef(out.reshape(-1), mids["a1"].reshape(-1))[0, 1])
# print(np.corrcoef(output_layer_value.reshape(-1), out.reshape(-1))[0, 1])

output_layer_value = eval_outs[1].transpose(0, 3, 1, 2)
print(np.corrcoef(output_layer_value.reshape(-1), mids["a1"].reshape(-1))[0, 1])
output_layer_value = eval_outs[2].transpose(0, 3, 1, 2)
print(np.corrcoef(output_layer_value.reshape(-1), mids["a2"].reshape(-1))[0, 1])
output_layer_value = eval_outs[len(acts)+1].transpose(0, 3, 1, 2)
print(np.corrcoef(output_layer_value.reshape(-1), mids["a34"].reshape(-1))[0, 1])


# print(eval_outs[0])
# print(outputs["layer1"].transpose(0, 2, 3, 1))
# print(output_layer_value.shape, outputs["layer5"].shape)
output_layer_value = eval_outs[-1].transpose(0, 3, 1, 2)
print(np.corrcoef(output_layer_value.reshape(-1), outputs["layer2"].reshape(-1))[0, 1])



# # --------------------
# # (2) Assign weights to the NNgen operators
# # --------------------

# # In this example, random floating-point values are assigned.
# # In a real case, you should assign actual weight values
# # obtianed by a training on DNN framework.

# # If you don't you NNgen's quantizer, you can assign integer weights to each tensor.


# import numpy as np

# w0_value = np.random.normal(size=w0.length).reshape(w0.shape)
# w0_value = np.clip(w0_value, -3.0, 3.0)
# w0.set_value(w0_value)

# b0_value = np.random.normal(size=b0.length).reshape(b0.shape)
# b0_value = np.clip(b0_value, -3.0, 3.0)
# b0.set_value(b0_value)

# s0_value = np.ones(s0.shape)
# s0.set_value(s0_value)

# w1_value = np.random.normal(size=w1.length).reshape(w1.shape)
# w1_value = np.clip(w1_value, -3.0, 3.0)
# w1.set_value(w1_value)

# b1_value = np.random.normal(size=b1.length).reshape(b1.shape)
# b1_value = np.clip(b1_value, -3.0, 3.0)
# b1.set_value(b1_value)

# s1_value = np.ones(s1.shape)
# s1.set_value(s1_value)

# w2_value = np.random.normal(size=w2.length).reshape(w2.shape)
# w2_value = np.clip(w2_value, -3.0, 3.0)
# w2.set_value(w2_value)

# b2_value = np.random.normal(size=b2.length).reshape(b2.shape)
# b2_value = np.clip(b2_value, -3.0, 3.0)
# b2.set_value(b2_value)

# s2_value = np.ones(s2.shape)
# s2.set_value(s2_value)

# w3_value = np.random.normal(size=w3.length).reshape(w3.shape)
# w3_value = np.clip(w3_value, -3.0, 3.0)
# w3.set_value(w3_value)

# b3_value = np.random.normal(size=b3.length).reshape(b3.shape)
# b3_value = np.clip(b3_value, -3.0, 3.0)
# b3.set_value(b3_value)

# s3_value = np.ones(s3.shape)
# s3.set_value(s3_value)

# # Quantizing the floating-point weights by the NNgen quantizer.
# # Alternatively, you can assign integer weights by yourself to each tensor.

# imagenet_mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
# imagenet_std = np.array([0.229, 0.224, 0.225]).astype(np.float32)

# if act_dtype.width > 8:
#     act_scale_factor = 128
# else:
#     act_scale_factor = int(round(2 ** (act_dtype.width - 1) * 0.5))

# input_scale_factors = {'input_layer': act_scale_factor}
# input_means = {'input_layer': imagenet_mean * act_scale_factor}
# input_stds = {'input_layer': imagenet_std * act_scale_factor}

# ng.quantize([output_layer], input_scale_factors, input_means, input_stds)


# # --------------------
# # (3) Assign hardware attributes
# # --------------------

# # conv2d, matmul
# # par_ich: parallelism in input-channel
# # par_och: parallelism in output-channel
# # par_col: parallelism in pixel column
# # par_row: parallelism in pixel row

# par_ich = 2
# par_och = 2

# a0.attribute(par_ich=par_ich, par_och=par_och)
# a1.attribute(par_ich=par_ich, par_och=par_och)
# a2.attribute(par_ich=par_ich, par_och=par_och)
# output_layer.attribute(par_ich=par_ich, par_och=par_och)

# # cshamt_out: right shift amount after applying bias/scale
# # If you assign integer weights by yourself to each tensor,
# # cshamt (constant shift amount) must be assigned to each operator.

# # a0.attribute(cshamt_out=weight_dtype.width + 1)
# # a1.attribute(cshamt_out=weight_dtype.width + 1)
# # a2.attribute(cshamt_out=weight_dtype.width + 1)
# # output_layer.attribute(cshamt_out=weight_dtype.width + 1)

# # max_pool
# # par: parallelism in in/out channel

# par = par_och

# a0p.attribute(par=par)


# # --------------------
# # (4) Verify the DNN model behavior by executing the NNgen dataflow as a software
# # --------------------

# # In this example, random integer values are assigned.
# # In real case, you should assign actual integer activation values, such as an image.

# input_layer_value = np.random.normal(size=input_layer.length).reshape(input_layer.shape)
# input_layer_value = input_layer_value * imagenet_std + imagenet_mean
# input_layer_value = np.clip(input_layer_value, -3.0, 3.0)
# input_layer_value = input_layer_value * act_scale_factor
# input_layer_value = np.clip(input_layer_value,
#                             -1 * 2 ** (act_dtype.width - 1) - 1, 2 ** (act_dtype.width - 1))
# input_layer_value = np.round(input_layer_value).astype(np.int64)

# eval_outs = ng.eval([output_layer], input_layer=input_layer_value)
# output_layer_value = eval_outs[0]

# print(output_layer_value)


# # --------------------
# # (5) Convert the NNgen dataflow to a hardware description (Verilog HDL and IP-XACT)
# # --------------------

# silent = False
# axi_datawidth = 32

# # to Veriloggen object
# # targ = ng.to_veriloggen([output_layer], 'hello_nngen', silent=silent,
# #                        config={'maxi_datawidth': axi_datawidth})

# # to IP-XACT (the method returns Veriloggen object, as well as to_veriloggen)
# targ = ng.to_ipxact([output_layer], 'hello_nngen', silent=silent,
#                     config={'maxi_datawidth': axi_datawidth})
# print('# IP-XACT was generated. Check the current directory.')

# # to Verilog HDL RTL (the method returns a source code text)
# # rtl = ng.to_verilog([output_layer], 'hello_nngen', silent=silent,
# #                    config={'maxi_datawidth': axi_datawidth})


# # --------------------
# # (6) Save the quantized weights
# # --------------------

# # convert weight values to a memory image:
# # on a real FPGA platform, this image will be used as a part of the model definition.

# param_filename = 'hello_nngen.npz'
# chunk_size = 64

# param_data = ng.export_ndarray([output_layer], chunk_size)
# np.savez_compressed(param_filename, param_data)


# # --------------------
# # (7) Simulate the generated hardware by Veriloggen and Verilog simulator
# # --------------------

# # If you don't check the RTL behavior, exit here.
# # print('# Skipping RTL simulation. If you simulate the RTL behavior, comment out the next line.')
# # sys.exit()

# import math
# from veriloggen import *
# import veriloggen.thread as vthread
# import veriloggen.types.axi as axi

# outputfile = 'hello_nngen.out'
# filename = 'hello_nngen.v'
# # simtype = 'iverilog'
# simtype = 'verilator'

# param_bytes = len(param_data)

# variable_addr = int(
#     math.ceil((input_layer.addr + input_layer.memory_size) / chunk_size)) * chunk_size
# check_addr = int(math.ceil((variable_addr + param_bytes) / chunk_size)) * chunk_size
# tmp_addr = int(math.ceil((check_addr + output_layer.memory_size) / chunk_size)) * chunk_size

# memimg_datawidth = 32
# mem = np.zeros([1024 * 1024 * 256 // memimg_datawidth], dtype=np.int64)
# mem = mem + [100]

# # placeholder
# axi.set_memory(mem, input_layer_value, memimg_datawidth,
#                act_dtype.width, input_layer.addr,
#                max(int(math.ceil(axi_datawidth / act_dtype.width)), par_ich))

# # parameters (variable and constant)
# axi.set_memory(mem, param_data, memimg_datawidth,
#                8, variable_addr)

# # verification data
# axi.set_memory(mem, output_layer_value, memimg_datawidth,
#                act_dtype.width, check_addr,
#                max(int(math.ceil(axi_datawidth / act_dtype.width)), par_och))

# # test controller
# m = Module('test')
# params = m.copy_params(targ)
# ports = m.copy_sim_ports(targ)
# clk = ports['CLK']
# resetn = ports['RESETN']
# rst = m.Wire('RST')
# rst.assign(Not(resetn))

# # AXI memory model
# if outputfile is None:
#     outputfile = os.path.splitext(os.path.basename(__file__))[0] + '.out'

# memimg_name = 'memimg_' + outputfile

# memory = axi.AxiMemoryModel(m, 'memory', clk, rst,
#                             datawidth=axi_datawidth,
#                             memimg=mem, memimg_name=memimg_name,
#                             memimg_datawidth=memimg_datawidth)
# memory.connect(ports, 'maxi')

# # AXI-Slave controller
# _saxi = vthread.AXIMLite(m, '_saxi', clk, rst, noio=True)
# _saxi.connect(ports, 'saxi')

# # timer
# time_counter = m.Reg('time_counter', 32, initval=0)
# seq = Seq(m, 'seq', clk, rst)
# seq(
#     time_counter.inc()
# )


# def ctrl():
#     for i in range(100):
#         pass

#     ng.sim.set_global_addrs(_saxi, tmp_addr)

#     start_time = time_counter.value
#     ng.sim.start(_saxi)

#     print('# start')

#     ng.sim.wait(_saxi)
#     end_time = time_counter.value

#     print('# end')
#     print('# execution cycles: %d' % (end_time - start_time))

#     # verify
#     ok = True
#     for bat in range(output_layer.shape[0]):
#         for x in range(output_layer.shape[1]):
#             orig = memory.read_word(bat * output_layer.aligned_shape[1] + x,
#                                     output_layer.addr, act_dtype.width)
#             check = memory.read_word(bat * output_layer.aligned_shape[1] + x,
#                                      check_addr, act_dtype.width)

#             if vthread.verilog.NotEql(orig, check):
#                 print('NG (', bat, x,
#                       ') orig: ', orig, ' check: ', check)
#                 ok = False
#             else:
#                 print('OK (', bat, x,
#                       ') orig: ', orig, ' check: ', check)

#     if ok:
#         print('# verify: PASSED')
#     else:
#         print('# verify: FAILED')

#     vthread.finish()


# th = vthread.Thread(m, 'th_ctrl', clk, rst, ctrl)
# fsm = th.start()

# uut = m.Instance(targ, 'uut',
#                  params=m.connect_params(targ),
#                  ports=m.connect_ports(targ))

# # simulation.setup_waveform(m, uut)
# simulation.setup_clock(m, clk, hperiod=5)
# init = simulation.setup_reset(m, resetn, m.make_reset(), period=100, polarity='low')

# init.add(
#     Delay(10000000),
#     Systask('finish'),
# )

# # output source code
# if filename is not None:
#     m.to_verilog(filename)

# # run simulation
# sim = simulation.Simulator(m, sim=simtype)
# rslt = sim.run(outputfile=outputfile)

# print(rslt)
