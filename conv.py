import nngen as ng
import numpy as np
import os

# import torch

# class depth_wise_conv():
#     def __init__(self, in_channels, out_channels, kernel_size, stride, groups, act_func,
#                  w0_value, b0_value, s0_value):
#         self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, stride=stride, groups=groups, bias=False)
#         self.conv.weight = torch.nn.Parameter(torch.tensor(w0_value.transpose(0, 3, 1, 2)))
#         # self.conv.bias = torch.nn.Parameter(torch.tensor(b0_value))
#         # self.s0_value = s0_value

#         # assert act_func == ng.relu or act_func is None
#         # self.act_func = torch.nn.ReLU() if act_func == ng.relu else lambda x : x

#     def forward(self, a0):
#         out = self.conv(torch.tensor(a0.astype(np.float32).transpose(0, 3, 1, 2))).detach().numpy().copy().transpose(0, 2, 3, 1)
#         # out *= self.s0_value[None, None, None, :]
#         # out = self.act_func(torch.tensor(out)).detach().numpy().copy()
#         out = out if a0.dtype == np.float32 else np.round(out).astype(a0.dtype)
#         return out


class Conv2d():
    acts = []

    def __init__(self, a0, param_name, in_channels, out_channels,
                 kernel_size, stride, groups, apply_bias=True, act_func=None,
                 weight_dtype=ng.int8, bias_dtype=ng.int32, scale_dtype=ng.int8, act_dtype=ng.int64):

        self.acts.append(a0)
        self.groups = groups
        self.apply_bias = apply_bias

        base_dir = os.path.dirname(os.path.abspath(__file__))

        params = np.load(os.path.join(base_dir, "params/params_scale.npz"))
        assert param_name + ".weight" in params.files
        w0_value = params[param_name + ".weight"].transpose(0, 2, 3, 1)
        # print(np.sum((w0_value < -3.0) + (w0_value > 3.0)) / len(w0_value))
        # w0_value = np.clip(w0_value, -3.0, 3.0)
        if apply_bias:
            assert param_name + ".bias" in params.files
            b0_value = params[param_name + ".bias"]
            # b0_value = np.clip(b0_value, -3.0, 3.0)
            # print(np.sum((b0_value < -3.0) + (b0_value > 3.0)) / len(b0_value))

        apply_scale = param_name + ".scale" in params.files
        if apply_scale:
            s0_value = params[param_name + ".scale"]
            # s0_value = np.clip(s0_value, -3.0, 3.0)

        par_ich = 2
        par_och = 2

        if groups != 1:
            assert in_channels == out_channels == groups

            ######################
            # うまくいかないパターン # (b0, s0 が量子化されない)
            ######################
            # out_shape = a0.shape if stride == 1 else (a0.shape[0], a0.shape[1]//stride, a0.shape[2]//stride, a0.shape[3])
            # conv = depth_wise_conv(in_channels, out_channels, kernel_size, stride, groups, act_func,
            #                        w0_value, b0_value, s0_value if apply_scale else np.ones_like(s0_value))
            # a1_0 = ng.extern([a0], opcode=0x1, func=conv.forward)
            # a1_0.shape = out_shape # デフォルトでは a1_0.shape = a0.shape になるようになっている

            # b0 = ng.variable(dtype=bias_dtype,
            #                  shape=(out_shape), name=param_name+".bias")
            # b0.set_value(b0_value[None, None, None, :] + np.zeros(out_shape).astype(b0_value.dtype))
            # a1_1 = ng.add(a1_0, b0)

            # s0 = ng.variable(dtype=scale_dtype,
            #                  shape=(a1_0.shape), name=param_name+".scale")
            # s0.set_value(s0_value[None, None, None, :] + np.zeros(a1_0.shape).astype(s0_value.dtype) if apply_scale else np.ones(a1_0.shape))
            # a1_2 = ng.multiply(a1_1, s0)
            # a1 = act_func(a1_2) if act_func is not None else a1_2


            ######################
            # うまくいかないパターン # (concat で誤差が大きくなってそう)
            ######################
            # w0s = [ng.variable(weight_dtype,
            #                    shape=(1, kernel_size, kernel_size, 1),
            #                    name='%s.weight_%d' % (param_name, g)) for g in range(groups)]
            # b0s = [ng.variable(bias_dtype,
            #                    shape=(1,),
            #                    name='%s.bias_%d' % (param_name, g)) for g in range(groups)]
            # if apply_scale:
            #     s0s = [ng.variable(scale_dtype,
            #                    shape=(1,),
            #                    name='%s.scale_%d' % (param_name, g)) for g in range(groups)]
            # else:
            #     s0s = [None for _ in range(groups)]

            # # w0_values = [np.random.normal(size=w0.length).reshape(w0.shape) for w0 in w0s]
            # # w0_values = [np.clip(w0_value, -3.0, 3.0) for w0_value in w0_values]
            # for g in range(groups):
            #     w0s[g].set_value(w0_value[g].reshape(w0s[g].shape))

            # # b0_values = [np.random.normal(size=b0.length).reshape(b0.shape) for b0 in b0s]
            # # b0_values = [np.clip(b0_value, -3.0, 3.0) for b0_value in b0_values]
            # for g in range(groups):
            #     b0s[g].set_value(b0_value[g].reshape(b0s[g].shape))

            # # s0_value = np.ones(s0.shape)
            # if apply_scale:
            #     for g in range(groups):
            #     #     s0s[g].set_value(np.ones(s0s[g].shape))
            #         s0s[g].set_value(s0_value[g].reshape(s0s[g].shape))

            # a1s = [ng.conv2d(ng.slice_(a0, (0, 0, 0, g), (*a0.shape[:3], g+1), (1, 1, 1, 1)), w0s[g],
            #                  strides=(1, stride, stride, 1),
            #                  bias=b0s[g],
            #                  scale=s0s[g],
            #                  act_func=act_func,
            #                  dtype=act_dtype,
            #                  sum_dtype=ng.int32) for g in range(groups)]
            # for g in range(groups):
            #     a1s[g].attribute(par_ich=par_ich, par_och=par_och)
            # self.a1s = a1s

            # a1 = ng.concat(a1s, axis=3)
            # self.a1 = a1


            ###################
            # うまくいくパターン #
            ###################
            w0_value = np.array([[[[w0_value[i][j][k][0] if i == l else 0 for l in range(in_channels)]
                                    for k in range(kernel_size)] for j in range(kernel_size)] for i in range(out_channels)])

            w0 = ng.variable(dtype=weight_dtype,
                             shape=(out_channels, kernel_size, kernel_size, in_channels),
                             name=param_name+".weight")
            b0 = ng.variable(dtype=bias_dtype,
                             shape=(out_channels,), name=param_name+".bias")
            if apply_scale:
                s0 = ng.variable(dtype=scale_dtype,
                                shape=(out_channels,), name=param_name+".scale")
            else:
                s0 = None

            # w0_value = np.random.normal(size=w0.length).reshape(w0.shape)
            # w0_value = np.clip(w0_value, -3.0, 3.0)
            w0.set_value(w0_value)

            # b0_value = np.random.normal(size=b0.length).reshape(b0.shape)
            # b0_value = np.clip(b0_value, -3.0, 3.0)
            b0.set_value(b0_value)

            # s0_value = np.ones(s0.shape)
            # s0.set_value(np.ones(s0.shape))
            if apply_scale:
                s0.set_value(s0_value)

            a1 = ng.conv2d(a0, w0,
                           strides=(1, stride, stride, 1),
                           bias=b0,
                           scale=s0,
                           act_func=act_func,
                           dtype=act_dtype,
                           sum_dtype=ng.int32)
            a1.attribute(par_ich=par_ich, par_och=par_och)

            ############
            # ここは共通 #
            ############
            self.acts.append(a1)

        elif apply_bias:
            w0 = ng.variable(dtype=weight_dtype,
                             shape=(out_channels, kernel_size, kernel_size, in_channels),
                             name=param_name+".weight")
            b0 = ng.variable(dtype=bias_dtype,
                             shape=(out_channels,), name=param_name+".bias")
            if apply_scale:
                s0 = ng.variable(dtype=scale_dtype,
                                shape=(out_channels,), name=param_name+".scale")
            else:
                s0 = None

            # w0_value = np.random.normal(size=w0.length).reshape(w0.shape)
            # w0_value = np.clip(w0_value, -3.0, 3.0)
            w0.set_value(w0_value)

            # b0_value = np.random.normal(size=b0.length).reshape(b0.shape)
            # b0_value = np.clip(b0_value, -3.0, 3.0)
            b0.set_value(b0_value)

            # s0_value = np.ones(s0.shape)
            # s0.set_value(np.ones(s0.shape))
            if apply_scale:
                s0.set_value(s0_value)

            a1 = ng.conv2d(a0, w0,
                           strides=(1, stride, stride, 1),
                           bias=b0,
                           scale=s0,
                           act_func=act_func,
                           dtype=act_dtype,
                           sum_dtype=ng.int32)
            a1.attribute(par_ich=par_ich, par_och=par_och)
            self.acts.append(a1)

        else:
            w0 = ng.variable(dtype=weight_dtype,
                             shape=(out_channels, kernel_size, kernel_size, in_channels),
                             name=param_name+".weight")
            if apply_scale:
                s0 = ng.variable(dtype=scale_dtype,
                                shape=(out_channels,), name=param_name+".scale")
            else:
                s0 = None

            # w0_value = np.random.normal(size=w0.length).reshape(w0.shape)
            # w0_value = np.clip(w0_value, -3.0, 3.0)
            w0.set_value(w0_value)

            # s0_value = np.ones(s0.shape)
            # s0.set_value(np.ones(s0.shape))
            if apply_scale:
                s0.set_value(s0_value)

            a1 = ng.conv2d(a0, w0,
                           strides=(1, stride, stride, 1),
                           scale=s0,
                           act_func=act_func,
                           dtype=act_dtype,
                           sum_dtype=ng.int32)
            a1.attribute(par_ich=par_ich, par_och=par_och)
            self.acts.append(a1)

    def get_output(self):
        return self.acts[-1]
