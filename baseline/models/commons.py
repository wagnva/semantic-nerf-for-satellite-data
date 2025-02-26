import torch
import numpy as np


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def get_nl(activation_function):
    return {"relu": torch.nn.ReLU, "siren": Siren, "softplus": torch.nn.Softplus}[
        activation_function
    ]


class Siren(torch.nn.Module):
    """
    Siren layer
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        result = torch.sin(self.w0 * input)
        return result


class Mapping(torch.nn.Module):
    def __init__(self, mapping_size, in_size, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super().__init__()
        self.N_freqs = mapping_size
        self.in_channels = in_size
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = self.in_channels * (len(self.funcs) * self.N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, self.N_freqs - 1, self.N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (self.N_freqs - 1), self.N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_channels)
        Outputs:
            out: (B, self.out_channels)
        """
        # out = [x]
        out = []
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, -1)
