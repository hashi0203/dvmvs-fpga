import nngen as ng
from conv import Conv2d

class _InvertedResidual():
    acts = []

    def __init__(self, a0, param_name, in_channels, out_channels,
                 kernel_size, stride, expansion_factor):

        assert kernel_size in [3, 5]
        assert stride in [1, 2]

        self.acts.append(a0)
        mid_channels = in_channels * expansion_factor

        self.layers0 = Conv2d(a0, param_name + ".layers.0",
                              in_channels, mid_channels,
                              kernel_size=1, stride=1, groups=1,
                              act_func=ng.relu)
        a1 = self.layers0.get_output()
        self.acts.append(a1)

        self.layers3 = Conv2d(a1, param_name + ".layers.3",
                              mid_channels, mid_channels,
                              kernel_size, stride, groups=mid_channels,
                              act_func=ng.relu)
        a2 = self.layers3.get_output()
        self.acts.append(a2)

        self.layers6 = Conv2d(a2, param_name + ".layers.6", mid_channels, out_channels,
                              kernel_size=1, stride=1, groups=1)
        a3 = self.layers6.get_output()
        self.acts.append(a3)

        self.apply_residual = (in_channels == out_channels and stride == 1)
        if self.apply_residual:
            a4 = ng.add(a3, a0)
            self.acts.append(a4)

    def get_output(self):
        return self.acts[-1]


class _stack():
    acts = []

    def __init__(self, a0, param_name, in_channels, out_channels,
                 kernel_size, stride, expansion_factor, repeats):
        """ Creates a stack of inverted residuals. """
        assert repeats >= 1
        self.acts.append(a0)
        # First one has no skip, because feature map size changes.
        self.first = _InvertedResidual(a0, param_name + ".0", in_channels, out_channels,
                                       kernel_size, stride, expansion_factor)
        a1 = self.first.get_output()
        self.acts.append(a1)

        self.remaining = []
        for r in range(1, repeats):
            self.remaining.append(_InvertedResidual(self.acts[-1], param_name + ".%d" % r,
                                                    out_channels, out_channels,
                                                    kernel_size, stride=1, expansion_factor=expansion_factor))
            self.acts.append(self.remaining[-1].get_output())

    def get_output(self):
        return self.acts[-1]