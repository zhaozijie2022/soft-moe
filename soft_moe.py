import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import multiprocessing as mp


def identity(x):
    return x


class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_sizes,
            hidden_activation=F.relu,
            output_activation=identity,
            layer_norm=True,
            out_layer_norm=False,
            use_residual=False,
    ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.out_layer_norm = out_layer_norm
        self.use_residual = use_residual

        self.fcs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = self.m_init(nn.Linear(in_size, next_size))
            in_size = next_size
            self.fcs.append(fc)

            if self.layer_norm:
                ln = nn.LayerNorm(next_size)
                self.layer_norms.append(ln)

        self.last_fc = self.m_init(nn.Linear(in_size, output_size))
        if self.out_layer_norm:
            self.last_ln = nn.LayerNorm(output_size)

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x1 = fc(x)
            if self.layer_norm:
                x1 = self.layer_norms[i](x1)
            if self.use_residual and (x.shape[-1] == x1.shape[-1]):
                x = x + self.hidden_activation(x1)
            else:
                x = self.hidden_activation(x1)

        y = self.last_fc(x)
        if self.out_layer_norm:
            y = self.last_ln(y)

        if self.use_residual and (x.shape[-1] == y.shape[-1]):
            y = x + self.output_activation(y)
        else:
            y = self.output_activation(y)
        return y

    def m_init(self, module, gain=0.01, activate=False):
        if activate:
            gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(module.weight.data, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
        return module


class SoftMoE(nn.Module):
    def __init__(self, d_model, num_experts, num_slots):
        super().__init__()
        # hidden_size = max(input_size, output_size)
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.experts = nn.ModuleList([
            Mlp(d_model, d_model, [d_model],
                hidden_activation=F.relu, output_activation=identity,
                layer_norm=True, out_layer_norm=False, use_residual=True)
            for _ in range(num_experts)
        ])
        self.phi = nn.Parameter(torch.randn(d_model, num_experts, num_slots))

    def forward(self, x, mask=None):
        # x.shape [b, n, d], mask.shape [b, n] ; e: experts, s: slots

        weights = torch.einsum("b n d , d e s -> b n e s", x, self.phi)
        if mask is not None:
            mask = einops.rearrange(mask, "b n -> b n 1 1")
            weights = weights.masked_fill(~mask, -torch.finfo(weights.dtype).max)

        # dispatch tokens to experts
        dispatch_weights = F.softmax(weights, dim=1)
        experts_inputs = torch.einsum("b n e s, b n d -> b e s d", dispatch_weights, x)  # equal to batch mat mul

        # input s inputs per expert
        expert_outputs = torch.stack([self.experts[i](experts_inputs[:, i]) for i in range(self.num_experts)])
        expert_outputs = einops.rearrange(expert_outputs, "e b s d -> b (e s) d")

        # combine expert outputs
        combine_weights = einops.rearrange(weights, "b n e s -> b n (e s)")
        combine_weights = F.softmax(combine_weights, dim=-1)
        out = torch.einsum("b n z, b z d -> b n d", combine_weights, expert_outputs)
        return out
















