import numpy as np
import nngen as ng

import math
from veriloggen import *
import veriloggen.thread as vthread
import veriloggen.types.axi as axi

class Simulator():

    def __init__(self, project_name, targ, param_data, axi_datawidth, chunk_size, par_ich, par_och, act_dtype):
        self.project_name = project_name
        self.targ = targ
        self.param_data = param_data
        self.axi_datawidth = axi_datawidth
        self.chunk_size = chunk_size
        self.par_ich = par_ich
        self.par_och = par_och
        self.act_dtype = act_dtype


    def _get_end_addr(self, addr, memory_size):
        chunk_size = self.chunk_size
        return int(math.ceil((addr + memory_size) / chunk_size)) * chunk_size


    def _get_variable_addr(self, layers):
        variable_addr = 0
        for l in layers:
            variable_addr = max(variable_addr, self._get_end_addr(l.addr, l.memory_size))
        return variable_addr


    def simulate(self, input_layers, input_layer_values, output_layer, output_layer_value):
        project_name = self.project_name
        targ = self.targ
        param_data = self.param_data
        axi_datawidth = self.axi_datawidth
        par_ich = self.par_ich
        par_och = self.par_och
        act_dtype = self.act_dtype

        # Simulate the generated hardware by Veriloggen and Verilog simulator

        outputfile = '%s_nngen.out' % project_name
        filename = '%s_nngen.v' % project_name
        # simtype = 'iverilog'
        simtype = 'verilator'

        param_bytes = len(param_data)

        variable_addr = self._get_variable_addr(input_layers)
        check_addr = self._get_end_addr(variable_addr, param_bytes)
        tmp_addr = self._get_end_addr(check_addr, output_layer.memory_size)

        # variable_addr = int(math.ceil((input_layer.addr + input_layer.memory_size) / chunk_size)) * chunk_size
        # check_addr = int(math.ceil((variable_addr + param_bytes) / chunk_size)) * chunk_size
        # tmp_addr = int(math.ceil((check_addr + output_layer.memory_size) / chunk_size)) * chunk_size
        # layers = input_layers + [output_layer]
        # size_max = int(math.ceil(self._max_memory_size(input_layers) / chunk_size)) * chunk_size
        # check_addr = self._max_addr(layers) + size_max
        # size_check = size_max
        # tmp_addr = check_addr + size_check

        memimg_datawidth = 32
        mem = np.zeros([1024 * 1024 * 1024 // memimg_datawidth], dtype=np.int64)
        # mem = mem + [100]

        # placeholder
        for i in range(len(input_layers)):
            axi.set_memory(mem, input_layer_values[i], memimg_datawidth,
                           act_dtype.width, input_layers[i].addr,
                           max(int(math.ceil(axi_datawidth / act_dtype.width)), par_ich))

        # parameters (variable and constant)
        axi.set_memory(mem, param_data, memimg_datawidth, 8, variable_addr)

        # verification data
        axi.set_memory(mem, output_layer_value, memimg_datawidth,
                       act_dtype.width, check_addr,
                       max(int(math.ceil(axi_datawidth / act_dtype.width)), par_och))

        # test controller
        m = Module('test')
        params = m.copy_params(targ) # targ に params を copy するために必要ではある？
        ports = m.copy_sim_ports(targ)
        clk = ports['CLK']
        resetn = ports['RESETN']
        rst = m.Wire('RST')
        rst.assign(Not(resetn))

        # AXI memory model
        memimg_name = 'memimg_' + outputfile

        memory = axi.AxiMemoryModel(m, 'memory', clk, rst,
                                    datawidth=axi_datawidth,
                                    memimg=mem, memimg_name=memimg_name,
                                    memimg_datawidth=memimg_datawidth)
        memory.connect(ports, 'maxi')

        # AXI-Slave controller
        _saxi = vthread.AXIMLite(m, '_saxi', clk, rst, noio=True)
        _saxi.connect(ports, 'saxi')

        # timer
        time_counter = m.Reg('time_counter', 32, initval=0)
        seq = Seq(m, 'seq', clk, rst)
        seq(
            time_counter.inc()
        )


        def ctrl():
            for _ in range(100):
                pass

            ng.sim.set_global_addrs(_saxi, tmp_addr)

            start_time = time_counter.value
            ng.sim.start(_saxi)

            print('# start')

            ng.sim.wait(_saxi)
            end_time = time_counter.value

            print('# end')
            print('# execution cycles: %d' % (end_time - start_time))

            # verify
            # output_layer.shape = (1, 64, 96, 1)
            # output_layer.aligned_shape = [1, 64, 96, 8]
            ok = True
            for bat in range(output_layer.shape[1]):
                for x in range(output_layer.shape[2]):
                    orig = memory.read_word((bat * output_layer.aligned_shape[2] + x) * output_layer.aligned_shape[3],
                                            output_layer.addr, act_dtype.width)
                    check = memory.read_word((bat * output_layer.aligned_shape[2] + x) * output_layer.aligned_shape[3],
                                             check_addr, act_dtype.width)

                    if vthread.verilog.NotEql(orig, check):
                        print('NG (%d, %d) orig: %d check: %d' % (bat, x, orig, check))
                        ok = False
                    # else:
                        # print('OK (%d, %d) orig: %d check: %d' % (bat, x, orig, check))

            if ok:
                print('# verify: PASSED')
            else:
                print('# verify: FAILED')

            vthread.finish()


        th = vthread.Thread(m, 'th_ctrl', clk, rst, ctrl)
        fsm = th.start()

        uut = m.Instance(targ, 'uut',
                         params=m.connect_params(targ),
                         ports=m.connect_ports(targ))

        # simulation.setup_waveform(m, uut)
        simulation.setup_clock(m, clk, hperiod=5)
        init = simulation.setup_reset(m, resetn, m.make_reset(), period=100, polarity='low')

        init.add(
            Delay(10000000),
            Systask('finish'),
        )

        # output source code
        m.to_verilog(filename)

        # run simulation
        sim = simulation.Simulator(m, sim=simtype)
        rslt = sim.run(outputfile=outputfile)

        print(rslt)
