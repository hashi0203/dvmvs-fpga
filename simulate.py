import numpy as np
import nngen as ng

import math
from veriloggen import *
import veriloggen.thread as vthread
import veriloggen.types.axi as axi

class Simulator():

    def __init__(self, project_name, targ, param_data, axi_datawidth, chunk_size, act_dtype):
        self.project_name = project_name
        self.targ = targ
        self.param_data = param_data
        self.axi_datawidth = axi_datawidth
        self.chunk_size = chunk_size
        self.act_dtype = act_dtype


    def simulate(self, input_layers, output_layer):
        project_name = self.project_name
        targ = self.targ
        param_data = self.param_data
        axi_datawidth = self.axi_datawidth
        chunk_size = self.chunk_size
        act_dtype = self.act_dtype

        # Simulate the generated hardware by Veriloggen and Verilog simulator

        outputfile = '%s_nngen.out' % project_name
        filename = '%s_nngen.v' % project_name
        # simtype = 'iverilog'
        simtype = 'verilator'

        param_bytes = len(param_data)

        variable_addr = int(
            math.ceil((input_layer.addr + input_layer.memory_size) / chunk_size)) * chunk_size
        check_addr = int(math.ceil((variable_addr + param_bytes) / chunk_size)) * chunk_size
        tmp_addr = int(math.ceil((check_addr + output_layer.memory_size) / chunk_size)) * chunk_size

        memimg_datawidth = 32
        mem = np.zeros([1024 * 1024 * 256 // memimg_datawidth], dtype=np.int64)
        mem = mem + [100]

        # placeholder
        axi.set_memory(mem, input_layer_value, memimg_datawidth,
                    act_dtype.width, input_layer.addr,
                    max(int(math.ceil(axi_datawidth / act_dtype.width)), par_ich))

        # parameters (variable and constant)
        axi.set_memory(mem, param_data, memimg_datawidth,
                    8, variable_addr)

        # verification data
        axi.set_memory(mem, output_layer_value, memimg_datawidth,
                    act_dtype.width, check_addr,
                    max(int(math.ceil(axi_datawidth / act_dtype.width)), par_och))

        # test controller
        m = Module('test')
        params = m.copy_params(targ)
        ports = m.copy_sim_ports(targ)
        clk = ports['CLK']
        resetn = ports['RESETN']
        rst = m.Wire('RST')
        rst.assign(Not(resetn))

        # AXI memory model
        if outputfile is None:
            outputfile = os.path.splitext(os.path.basename(__file__))[0] + '.out'

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
            ok = True
            for bat in range(output_layer.shape[0]):
                for x in range(output_layer.shape[1]):
                    orig = memory.read_word(bat * output_layer.aligned_shape[1] + x,
                                            output_layer.addr, act_dtype.width)
                    check = memory.read_word(bat * output_layer.aligned_shape[1] + x,
                                            check_addr, act_dtype.width)

                    if vthread.verilog.NotEql(orig, check):
                        print('NG (', bat, x,
                            ') orig: ', orig, ' check: ', check)
                        ok = False
                    else:
                        print('OK (', bat, x,
                            ') orig: ', orig, ' check: ', check)

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
        if filename is not None:
            m.to_verilog(filename)

        # run simulation
        sim = simulation.Simulator(m, sim=simtype)
        rslt = sim.run(outputfile=outputfile)

        print(rslt)
