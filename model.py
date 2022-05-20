import nngen as ng
from conv import Conv2d
from mnasnet import _stack

class StandardLayer():
    acts = []

    def __init__(self, a0, param_name, channels, kernel_size):
        self.acts.append(a0)
        self.conv1 = Conv2d(a0, param_name + ".conv1.0", channels, channels,
                            kernel_size, stride=1, groups=1,
                            act_func=ng.relu)
        a1 = self.conv1.get_output()
        self.acts.append(a1)

        self.conv2 = Conv2d(a1, param_name + ".conv2.0", channels, channels,
                            kernel_size, stride=1, groups=1,
                            act_func=ng.relu)
        a2 = self.conv2.get_output()
        self.acts.append(a2)

    def get_output(self):
        return self.acts[-1]


class DownconvolutionLayer():
    acts = []

    def __init__(self, a0, param_name, in_channels, out_channels, kernel_size):
        self.acts.append(a0)
        self.down_conv = Conv2d(a0, param_name + ".down_conv.0", in_channels, out_channels,
                                kernel_size, stride=2, groups=1,
                                act_func=ng.relu)
        a1 = self.down_conv.get_output()
        self.acts.append(a1)

    def get_output(self):
        return self.acts[-1]


class UpconvolutionLayer():
    acts = []

    def __init__(self, a0, param_name, in_channels, out_channels, kernel_size):
        self.acts.append(a0)
        a1 = ng.upsampling2d(a0, factors=2)
        # x = torch.nn.functional.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=True)
        self.acts.append(a1)

        self.conv = Conv2d(a1, param_name + ".conv.0", in_channels, out_channels,
                           kernel_size, stride=1, groups=1, act_func=ng.relu)
        a2 = self.conv.get_output()
        self.acts.append(a2)

    def get_output(self):
        return self.acts[-1]


class EncoderBlock():
    acts = []

    def __init__(self, a0, param_name, in_channels, out_channels, kernel_size):
        self.acts.append(a0)
        self.down_convolution = DownconvolutionLayer(a0, param_name + ".down_convolution",
                                                     in_channels, out_channels, kernel_size)
        a1 = self.down_convolution.get_output()
        self.acts.append(a1)

        self.standard_convolution = StandardLayer(a0, param_name + ".standard_convolution",
                                                  out_channels, kernel_size)
        a2 = self.standard_convolution.get_output()
        self.acts.append(a2)

    def get_output(self):
        return self.acts[-1]


class DecoderBlock():
    acts = []

    def __init__(self, a0, param_name, skip, depth, in_channels, out_channels, kernel_size):
        self.acts.append(a0)
        # Upsample the input coming from previous layer
        self.up_convolution = UpconvolutionLayer(a0, param_name + ".up_convolution",
                                                 in_channels, out_channels, kernel_size)
        a1 = self.up_convolution.get_output()
        self.acts.append(a1)

        if depth is None:
            a2 = ng.concat([a1, skip], axis=3)
            # x = torch.cat([x, skip], dim=1)

            next_in_channels = in_channels
        else:
            depth = ng.upsampling2d(depth, factors=2)
            a2 = ng.concat([a1, skip, depth], axis=3)
            # depth = torch.nn.functional.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=True)
            # x = torch.cat([x, skip, depth], dim=1)

            next_in_channels = in_channels + 1
        self.acts.append(a2)

        # Aggregate skip and upsampled input
        self.convolution1 = Conv2d(a2, param_name + ".convolution1.0",
                                   next_in_channels, out_channels,
                                   kernel_size, stride=1, groups=1,
                                   act_func=ng.relu)
        a3 = self.convolution1.get_output()
        self.acts.append(a3)

        # Learn from aggregation
        self.convolution2 = Conv2d(a3, param_name + ".convolution2.0",
                                   out_channels, out_channels,
                                   kernel_size, stride=1, groups=1,
                                   act_func=ng.relu)
        a4 = self.convolution2.get_output()
        self.acts.append(a4)

    def get_output(self):
        return self.acts[-1]


class FeatureExtractor():
    acts = []

    def __init__(self, a0):
        self.acts.append(a0)
        in_channels = 3
        depths = [32, 16, 24, 40, 80, 96, 192, 320]

        self.layer1_0 = Conv2d(a0, "layer1.0", in_channels, depths[0],
                               kernel_size=3, stride=2, groups=1,
                               act_func=ng.relu)
        a1 = self.layer1_0.get_output()
        self.acts.append(a1)

        self.layer1_3 = Conv2d(a1, "layer1.3", depths[0], depths[0],
                               kernel_size=3, stride=1, groups=depths[0],
                               act_func=ng.relu)
        a2 = self.layer1_3.get_output()
        self.acts.append(a2)

        self.layer1_6 = Conv2d(a2, "layer1.6", depths[0], depths[1],
                               kernel_size=1, stride=1, groups=1)
        a3 = self.layer1_6.get_output()
        self.acts.append(a3)

        self.layer2_0 = _stack(a3, "layer2.0", depths[1], depths[2],
                               kernel_size=3, stride=2, expansion_factor=3, repeats=3)
        a4 = self.layer2_0.get_output()
        self.acts.append(a4)

        self.layer3_0 = _stack(a4, "layer3.0", depths[2], depths[3],
                               kernel_size=5, stride=2, expansion_factor=3, repeats=3)
        a5 = self.layer3_0.get_output()
        self.acts.append(a5)

        self.layer4_0 = _stack(a5, "layer4.0", depths[3], depths[4],
                               kernel_size=5, stride=2, expansion_factor=6, repeats=3)
        a6 = self.layer4_0.get_output()
        self.acts.append(a6)

        self.layer4_1 = _stack(a6, "layer4.1", depths[4], depths[5],
                               kernel_size=3, stride=1, expansion_factor=6, repeats=2)
        a7 = self.layer4_1.get_output()
        self.acts.append(a7)

        self.layer5_0 = _stack(a7, "layer5.0", depths[5], depths[6],
                               kernel_size=5, stride=2, expansion_factor=6, repeats=4)
        a8 = self.layer5_0.get_output()
        self.acts.append(a8)

        self.layer5_1 = _stack(a8, "layer5.1", depths[6], depths[7],
                               kernel_size=3, stride=1, expansion_factor=6, repeats=1)
        a9 = self.layer5_1.get_output()
        self.acts.append(a9)

    def get_output(self):
        return self.acts, self.acts[3], self.acts[4], self.acts[5], self.acts[7], self.acts[9]
        # return self.layer2_0.acts[1], self.acts[3], self.acts[4], self.acts[5], self.acts[7], self.acts[9]
