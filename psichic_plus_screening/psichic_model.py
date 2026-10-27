#!/usr/bin/env python3
"""Canonical single-file PSICHIC+ production model surface.

This module intentionally keeps the production PSICHIC+ architecture in one
file. It has no local repo model imports, so checkpoint loading and inference
can be validated without reaching back into the legacy ``codebase/models`` tree.
"""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import ModuleList, Sequential
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import GCNConv, MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import degree, softmax, subgraph, to_dense_adj, to_dense_batch
from torch_scatter import scatter

PROD_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = Path(
    os.environ.get(
        "PSICHIC_PLUS_MODEL_DIR",
        str(PROD_DIR / "pretrained_weights" / "PSICHIC_plus"),
    )
)

EPS = 1e-15
MOTIFPOOL_DENSE_STRATEGIES = ("pyg_to_dense", "prealloc_scatter", "precomputed_scatter", "dense_through")
PROTEIN_MINCUT_STRATEGIES = ("pyg_dense_mincut", "inference_no_loss")
PROTEIN_DENSE_CACHE_STRATEGIES = ("none", "protein_dense_inputs")
INFERENCE_PRESETS = {
    "none": {},
    "conservative": {
        "motifpool_dense_strategy": "pyg_to_dense",
        "protein_mincut_strategy": "pyg_dense_mincut",
        "protein_dense_cache_strategy": "none",
    },
    "fast_screening": {
        "motifpool_dense_strategy": "dense_through",
        "protein_mincut_strategy": "inference_no_loss",
        "protein_dense_cache_strategy": "protein_dense_inputs",
    },
}
INFERENCE_PRESET_CHOICES = tuple(INFERENCE_PRESETS)


def resolve_inference_preset(
    inference_preset: str,
    motifpool_dense_strategy: str,
    protein_mincut_strategy: str,
    protein_dense_cache_strategy: str,
) -> dict[str, str]:
    if inference_preset not in INFERENCE_PRESETS:
        valid = ", ".join(INFERENCE_PRESET_CHOICES)
        raise ValueError(f"Unknown inference preset '{inference_preset}'. Expected one of: {valid}")

    resolved = {
        "inference_preset": inference_preset,
        "motifpool_dense_strategy": _validate_motifpool_dense_strategy(motifpool_dense_strategy),
        "protein_mincut_strategy": _validate_protein_mincut_strategy(protein_mincut_strategy),
        "protein_dense_cache_strategy": _validate_protein_dense_cache_strategy(protein_dense_cache_strategy),
    }
    resolved.update(INFERENCE_PRESETS[inference_preset])
    return resolved


def _validate_motifpool_dense_strategy(strategy: str) -> str:
    if strategy not in MOTIFPOOL_DENSE_STRATEGIES:
        valid = ", ".join(MOTIFPOOL_DENSE_STRATEGIES)
        raise ValueError(f"Unknown MotifPool dense strategy '{strategy}'. Expected one of: {valid}")
    return strategy


def _validate_protein_mincut_strategy(strategy: str) -> str:
    if strategy not in PROTEIN_MINCUT_STRATEGIES:
        valid = ", ".join(PROTEIN_MINCUT_STRATEGIES)
        raise ValueError(f"Unknown protein MinCut strategy '{strategy}'. Expected one of: {valid}")
    return strategy


def _validate_protein_dense_cache_strategy(strategy: str) -> str:
    if strategy not in PROTEIN_DENSE_CACHE_STRATEGIES:
        valid = ", ".join(PROTEIN_DENSE_CACHE_STRATEGIES)
        raise ValueError(f"Unknown protein dense cache strategy '{strategy}'. Expected one of: {valid}")
    return strategy


def _build_motifpool_dense_layout(
    batch: Tensor,
    ptr: Tensor | None,
    include_mask: bool = True,
) -> dict[str, Any]:
    num_nodes = int(batch.numel())
    if num_nodes == 0:
        return {
            "batch_size": 0,
            "max_nodes": 0,
            "slots": 0,
            "flat_index": batch.new_empty((0,)),
            "mask": torch.empty((0, 0), dtype=torch.bool, device=batch.device),
        }

    if ptr is not None:
        counts = ptr[1:] - ptr[:-1]
        batch_size = int(ptr.numel()) - 1
        starts = torch.repeat_interleave(ptr[:-1], counts, output_size=num_nodes)
    else:
        batch_size = int(batch[-1].item()) + 1
        counts = torch.bincount(batch, minlength=batch_size)
        starts = torch.repeat_interleave(
            torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)[:-1]]),
            counts,
            output_size=num_nodes,
        )

    max_nodes = int(counts.max().item()) if counts.numel() else 0
    slots = batch_size * max_nodes
    column = torch.arange(num_nodes, device=batch.device) - starts
    flat_index = batch * max_nodes + column
    layout = {
        "batch_size": batch_size,
        "max_nodes": max_nodes,
        "slots": slots,
        "flat_index": flat_index,
    }
    if include_mask:
        mask = torch.zeros(slots, dtype=torch.bool, device=batch.device)
        mask.index_fill_(0, flat_index, True)
        layout["mask"] = mask.view(batch_size, max_nodes)
    return layout


def _sparse_to_dense_with_layout(x: Tensor, layout: dict[str, Any]) -> Tensor:
    batch_size = int(layout["batch_size"])
    max_nodes = int(layout["max_nodes"])
    slots = int(layout["slots"])
    dense_flat = x.new_zeros((slots, *x.shape[1:]))
    if x.size(0) > 0:
        dense_flat.index_copy_(0, layout["flat_index"], x)
    return dense_flat.view(batch_size, max_nodes, *x.shape[1:])


@contextmanager
def _nvtx_range(name: str, enabled: bool):
    if not enabled or not torch.cuda.is_available():
        with nullcontext():
            yield
        return

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


class DegreeScalerAggregation(Aggregation):
    """PNA aggregation with degree-dependent scaler terms."""

    def __init__(
        self,
        aggr: Union[str, List[str], Aggregation],
        scaler: Union[str, List[str]],
        deg: Tensor,
        aggr_kwargs: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__()

        if isinstance(aggr, (str, Aggregation)):
            self.aggr = aggr_resolver(aggr, **(aggr_kwargs or {}))
        elif isinstance(aggr, (tuple, list)):
            self.aggr = MultiAggregation(aggr, aggr_kwargs)
        else:
            raise ValueError(
                "Only strings, list, tuples and Aggregation instances are valid "
                f"aggregation schemes (got '{type(aggr)}')"
            )

        self.scaler = [scaler] if isinstance(aggr, str) else scaler

        deg = deg.to(torch.float)
        num_nodes = int(deg.sum())
        bin_degrees = torch.arange(deg.numel(), device=deg.device)
        self.avg_deg: Dict[str, float] = {
            'lin': float((bin_degrees * deg).sum()) / num_nodes,
            'log': float(((bin_degrees + 1).log() * deg).sum()) / num_nodes,
            'exp': float((bin_degrees.exp() * deg).sum()) / num_nodes,
        }

    def forward(
        self,
        x: Tensor,
        index: Optional[Tensor] = None,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
    ) -> Tensor:
        self.assert_index_present(index)

        out = self.aggr(x, index, ptr, dim_size, dim)

        assert index is not None
        deg = degree(index, num_nodes=dim_size, dtype=out.dtype).clamp_(1)
        size = [1] * len(out.size())
        size[dim] = -1
        deg = deg.view(size)

        outs = []
        for scaler in self.scaler:
            if scaler == 'identity':
                out_scaler = out
            elif scaler == 'amplification':
                out_scaler = out * (torch.log(deg + 1) / self.avg_deg['log'])
            elif scaler == 'attenuation':
                out_scaler = out * (self.avg_deg['log'] / torch.log(deg + 1))
            elif scaler == 'exponential':
                out_scaler = out * (torch.exp(deg) / self.avg_deg['exp'])
            elif scaler == 'linear':
                out_scaler = out * (deg / self.avg_deg['lin'])
            elif scaler == 'inverse_linear':
                out_scaler = out * (self.avg_deg['lin'] / deg)
            else:
                raise ValueError(f"Unknown scaler '{scaler}'")
            outs.append(out_scaler)

        return torch.cat(outs, dim=-1) if len(outs) > 1 else outs[0]


class PNAConv(MessagePassing):
    """Local PNAConv copy kept in-file for checkpoint-compatible inference."""

    def __init__(self, in_channels: int, out_channels: int,
                 aggregators: List[str], scalers: List[str], deg: Tensor,
                 edge_dim: Optional[int] = None, towers: int = 1,
                 pre_layers: int = 1, post_layers: int = 1,
                 act: Union[str, Callable, None] = "relu",
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 divide_input: bool = False, **kwargs):

        aggr = DegreeScalerAggregation(aggregators, scalers, deg)
        super().__init__(aggr=aggr, node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = self.out_channels // towers

        if self.edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, self.F_in)

        self.pre_nns = ModuleList()
        self.post_nns = ModuleList()
        for _ in range(towers):
            modules = [Linear((3 if edge_dim else 2) * self.F_in, self.F_in)]
            for _ in range(pre_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [Linear(self.F_in, self.F_in)]
            self.pre_nns.append(Sequential(*modules))

            in_channels = (len(aggregators) * len(scalers) + 1) * self.F_in
            modules = [Linear(in_channels, self.F_out)]
            for _ in range(post_layers - 1):
                modules += [activation_resolver(act, **(act_kwargs or {}))]
                modules += [Linear(self.F_out, self.F_out)]
            self.post_nns.append(Sequential(*modules))

        self.lin = Linear(out_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        if self.edge_dim is not None:
            self.edge_encoder.reset_parameters()
        for nn in self.pre_nns:
            reset(nn)
        for nn in self.post_nns:
            reset(nn)
        self.lin.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        if self.divide_input:
            x = x.view(-1, self.towers, self.F_in)
        else:
            x = x.view(-1, 1, self.F_in).repeat(1, self.towers, 1)

        # propagate_type: (x: Tensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

        out = torch.cat([x, out], dim=-1)
        outs = [nn(out[:, i]) for i, nn in enumerate(self.post_nns)]
        out = torch.cat(outs, dim=1)

        return self.lin(out)

    def message(self, x_i: Tensor, x_j: Tensor,
                edge_attr: OptTensor) -> Tensor:

        h: Tensor = x_i  # Dummy.
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = edge_attr.view(-1, 1, self.F_in)
            edge_attr = edge_attr.repeat(1, self.towers, 1)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)

        hs = [nn(h[:, i]) for i, nn in enumerate(self.pre_nns)]
        return torch.stack(hs, dim=1)

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, towers={self.towers}, '
                f'edge_dim={self.edge_dim})')

    @staticmethod
    def get_degree_histogram(loader) -> Tensor:
        max_degree = 0
        for data in loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes,
                       dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))
        # Compute the in-degree histogram tensor
        deg_histogram = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes,
                       dtype=torch.long)
            deg_histogram += torch.bincount(d, minlength=deg_histogram.numel())

        return deg_histogram


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


def _layer_stack(dims, layer_factory):
    return nn.ModuleList(layer_factory(dims[idx - 1], dims[idx]) for idx in range(1, len(dims)))


def _reset_layers(layers):
    for layer in layers:
        layer.reset_parameters()


def _hidden_then_output(layers, x, apply_layer):
    for layer in layers[:-1]:
        x = F.relu(apply_layer(layer, x))
    return apply_layer(layers[-1], x)


def _reset_optional_norms(*norms):
    for norm in norms:
        if norm is not None:
            norm.reset_parameters()


def _reset_conv_norm(conv, norm):
    conv.reset_parameters()
    norm.reset_parameters()


def _residual_conv_norm_dropout(x, conv, norm, edge_index, edge_attr, dropout, training):
    x_in = x
    x = x_in + F.relu(norm(conv(x, edge_index, edge_attr)))
    return F.dropout(x, dropout, training=training)


def _validate_dropout_probability(p):
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')


def _linear_heads(linear, x, heads, channels):
    return linear(x).view(-1, heads, channels)


class GCNCluster(torch.nn.Module):
    def __init__(self, dims, out_norm=False, in_norm=False):
        super().__init__()
        self.out_norm = out_norm
        self.in_norm = in_norm
        self.Conv_layers = _layer_stack(dims, GCNConv)
        self.hidden_layers = len(self.Conv_layers) - 1

        self.out_ln = LayerNorm(dims[-1]) if self.out_norm else None
        self.in_ln = LayerNorm(dims[0]) if self.in_norm else None

    def reset_parameters(self):
        _reset_layers(self.Conv_layers)
        _reset_optional_norms(self.out_ln, self.in_ln)

    def forward(self, x, edge_index):
        y = x
        if self.in_ln is not None:
            y = self.in_ln(y)

        y = _hidden_then_output(self.Conv_layers, y, lambda layer, value: layer(value, edge_index))

        if self.out_ln is not None:
            y = self.out_ln(y)

        return y


class PosLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, init_value=0.2,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        lower_bound, upper_bound = init_value / 2, init_value
        weight = torch.empty((out_features, in_features), **factory_kwargs)
        weight = nn.init.uniform_(weight, a=lower_bound, b=upper_bound)
        self.weight = nn.Parameter(torch.abs(weight).log())
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        if self.bias is not None:
            nn.init.uniform_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight.exp(), self.bias)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class MLP(nn.Module):

    def __init__(self, dims, out_norm=False, in_norm=False, bias=True):
        super().__init__()
        self.out_norm = out_norm
        self.in_norm = in_norm
        self.FC_layers = _layer_stack(dims, lambda in_dim, out_dim: nn.Linear(in_dim, out_dim, bias=bias))
        self.hidden_layers = len(self.FC_layers) - 1

        self.out_ln = LayerNorm(dims[-1]) if self.out_norm else None
        self.in_ln = LayerNorm(dims[0]) if self.in_norm else None

    def reset_parameters(self):
        _reset_layers(self.FC_layers)
        _reset_optional_norms(self.out_ln, self.in_ln)

    def forward(self, x):
        y = x
        if self.in_ln is not None:
            y = self.in_ln(y)

        y = _hidden_then_output(self.FC_layers, y, lambda layer, value: layer(value))

        if self.out_ln is not None:
            y = self.out_ln(y)

        return y


class Drug_PNAConv(nn.Module):
    def __init__(self, mol_deg, hidden_channels, edge_channels,
                 pre_layers=2, post_layers=2,
                 aggregators=['sum', 'mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'attenuation'],
                 num_towers=4,
                 dropout=0.1):
        super().__init__()

        self.bond_encoder = torch.nn.Linear(edge_channels, hidden_channels)

        self.atom_conv = PNAConv(
            in_channels=hidden_channels, out_channels=hidden_channels,
            edge_dim=hidden_channels, aggregators=aggregators,
            scalers=scalers, deg=mol_deg, pre_layers=pre_layers,
            post_layers=post_layers,towers=num_towers,divide_input=True,
        )
        self.atom_norm = LayerNorm(hidden_channels)

        self.dropout = dropout

    def reset_parameters(self):
        _reset_conv_norm(self.atom_conv, self.atom_norm)

    def forward(self, atom_x, bond_x, atom_edge_index):
        bond_x = self.bond_encoder(bond_x.squeeze())
        return _residual_conv_norm_dropout(
            atom_x, self.atom_conv, self.atom_norm,
            atom_edge_index, bond_x, self.dropout, self.training,
        )


class Protein_PNAConv(nn.Module):
    def __init__(self, prot_deg, hidden_channels, edge_channels,
                 pre_layers=2, post_layers=2,
                 aggregators=['sum', 'mean', 'min', 'max', 'std'],
                 scalers=['identity', 'amplification', 'attenuation'],
                 num_towers=4,
                 dropout=0.1):
        super().__init__()

        self.conv = PNAConv(in_channels=hidden_channels,
                            out_channels=hidden_channels,
                            edge_dim=edge_channels,
                            aggregators=aggregators,
                            scalers=scalers,
                            deg=prot_deg,
                            pre_layers=pre_layers,
                            post_layers=post_layers,
                            towers=num_towers,
                            divide_input=True,
                            )

        self.norm = LayerNorm(hidden_channels)
        self.dropout = dropout

    def reset_parameters(self):
        _reset_conv_norm(self.conv, self.norm)

    def forward(self, x, prot_edge_index, prot_edge_attr):
        return _residual_conv_norm_dropout(
            x, self.conv, self.norm,
            prot_edge_index, prot_edge_attr, self.dropout, self.training,
        )


class DrugProteinConv(MessagePassing):

    _alpha: OptTensor

    def __init__(
        self,
        atom_channels: int,
        residue_channels: int,
        heads: int = 1,
        t=0.2,
        dropout_attn_score=0.2,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        assert residue_channels % heads == 0
        assert atom_channels % heads == 0

        self.residue_out_channels = residue_channels // heads
        self.atom_out_channels = atom_channels // heads
        self.heads = heads
        self.edge_dim = edge_dim
        self._alpha = None

        self.lin_key = nn.Linear(residue_channels, heads * self.atom_out_channels, bias=False)
        self.lin_query = nn.Linear(atom_channels, heads * self.atom_out_channels, bias=False)
        self.lin_value = nn.Linear(residue_channels, heads * self.atom_out_channels, bias=False)
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * self.atom_out_channels, bias=False)
        else:
            self.lin_edge = self.register_parameter('lin_edge', None)

        self.lin_atom_value = nn.Linear(atom_channels, heads * self.residue_out_channels, bias=False)

        self.drug_in_norm = LayerNorm(atom_channels)
        self.residue_in_norm = LayerNorm(residue_channels)

        self.drug_out_norm = LayerNorm(heads * self.atom_out_channels)
        self.residue_out_norm = LayerNorm(heads * self.residue_out_channels)
        self.clique_mlp = MLP([atom_channels * 2, atom_channels * 2, atom_channels], out_norm=True)
        self.residue_mlp = MLP([residue_channels * 2, residue_channels * 2, residue_channels], out_norm=True)
        self.t = t
        self.dropout_attn_score = dropout_attn_score

    def reset_parameters(self):
        _reset_layers((self.lin_key, self.lin_query, self.lin_value))
        if self.edge_dim:
            self.lin_edge.reset_parameters()
        _reset_layers((
            self.lin_atom_value,
            self.drug_in_norm,
            self.residue_in_norm,
            self.drug_out_norm,
            self.residue_out_norm,
            self.clique_mlp,
            self.residue_mlp,
        ))

    def forward(self, drug_x, clique_x, clique_batch, residue_x, edge_index: Adj):
        H, aC = self.heads, self.atom_out_channels
        residue_hx = self.residue_in_norm(residue_x)
        query = _linear_heads(self.lin_query, drug_x, H, aC)
        key = _linear_heads(self.lin_key, residue_hx, H, aC)
        value = _linear_heads(self.lin_value, residue_hx, H, aC)

        drug_out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=None, size=None)
        alpha = self._alpha
        self._alpha = None

        drug_out = drug_out.view(-1, H * aC)
        drug_out = self.drug_out_norm(drug_out)
        clique_out = torch.cat([clique_x, drug_out[clique_batch]], dim=-1)
        clique_out = self.clique_mlp(clique_out)

        H, rC = self.heads, self.residue_out_channels
        drug_hx = self.drug_in_norm(drug_x)
        residue_value = _linear_heads(self.lin_atom_value, drug_hx, H, rC)[edge_index[1]]
        residue_out = residue_value * alpha.view(-1, H, 1)
        residue_out = residue_out.view(-1, H * rC)
        residue_out = self.residue_out_norm(residue_out)
        residue_out = torch.cat([residue_out, residue_x], dim=-1)
        residue_out = self.residue_mlp(residue_out)

        return clique_out, residue_out, (edge_index, alpha)

    def forward_dense_clique(
        self,
        drug_x: Tensor,
        clique_x: Tensor,
        clique_mask: Tensor,
        residue_x: Tensor,
        edge_index: Adj,
    ):
        H, aC = self.heads, self.atom_out_channels
        residue_hx = self.residue_in_norm(residue_x)
        query = _linear_heads(self.lin_query, drug_x, H, aC)
        key = _linear_heads(self.lin_key, residue_hx, H, aC)
        value = _linear_heads(self.lin_value, residue_hx, H, aC)

        drug_out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=None,
            size=None,
        )
        alpha = self._alpha
        self._alpha = None

        drug_out = drug_out.view(-1, H * aC)
        drug_out = self.drug_out_norm(drug_out)
        drug_out = drug_out.unsqueeze(1).expand(-1, clique_x.size(1), -1)
        clique_out = torch.cat([clique_x, drug_out], dim=-1)
        clique_out = self.clique_mlp(clique_out)
        clique_out = clique_out.masked_fill(~clique_mask.unsqueeze(-1), 0.0)

        H, rC = self.heads, self.residue_out_channels
        drug_hx = self.drug_in_norm(drug_x)
        residue_value = _linear_heads(self.lin_atom_value, drug_hx, H, rC)[edge_index[1]]
        residue_out = residue_value * alpha.view(-1, H, 1)
        residue_out = residue_out.view(-1, H * rC)
        residue_out = self.residue_out_norm(residue_out)
        residue_out = torch.cat([residue_out, residue_x], dim=-1)
        residue_out = self.residue_mlp(residue_out)

        return clique_out, residue_out, (edge_index, alpha)

    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.atom_out_channels)
        alpha = alpha / self.t

        alpha = F.dropout(alpha, p=self.dropout_attn_score, training=self.training)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha

        out = value_j
        out = out * alpha.view(-1, self.heads, 1)

        return out


def unbatch(src, batch, dim: int = 0):
    """Split a tensor by an ordered PyG batch vector."""
    sizes = degree(batch, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def dropout_node(edge_index, p, num_nodes, batch, training):
    """Drop graph nodes while keeping at least one node per graph."""
    _validate_dropout_probability(p)

    if not training or p == 0.0:
        node_mask = edge_index.new_ones(num_nodes, dtype=torch.bool)
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask, node_mask
    
    prob = torch.rand(num_nodes, device=edge_index.device)
    node_mask = prob > p
    
    batch_tf = global_add_pool(node_mask.view(-1, 1), batch).flatten()
    unbatched_node_mask = unbatch(node_mask, batch)
    node_mask_list = []
    
    for true_false, sub_node_mask in zip(batch_tf, unbatched_node_mask):
        if true_false.item():
            node_mask_list.append(sub_node_mask)
        else:
            perm = torch.randperm(sub_node_mask.size(0))
            idx = perm[:1]
            sub_node_mask[idx] = True
            node_mask_list.append(sub_node_mask)
            
    node_mask = torch.cat(node_mask_list)
    
    edge_index, _, edge_mask = subgraph(node_mask, edge_index,
                                        num_nodes=num_nodes,
                                        return_edge_mask=True)
    return edge_index, edge_mask, node_mask


def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    """Drop graph edges with optional undirected-pair handling."""
    _validate_dropout_probability(p)

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask

    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


class MotifPool(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        heads,
        pe_walk_length=20,
        dropout_attn_score=0,
        dense_strategy: str = "pyg_to_dense",
    ):
        super().__init__()
        assert hidden_channels % heads == 0

        self.pe_lin = torch.nn.Linear(pe_walk_length, hidden_channels, bias=False)
        self.pe_norm = LayerNorm(hidden_channels)

        self.atom_proj = torch.nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.atom_norm = LayerNorm(hidden_channels)

        self.attn = torch.nn.MultiheadAttention(
                hidden_channels,
                heads,
                batch_first=True
            )
        self.clique_norm = LayerNorm(hidden_channels)
        mh_hidden_channels = hidden_channels // heads
        
        self.score_proj = torch.nn.ModuleList()
        for _ in range(heads): 
            self.score_proj.append(MLP([mh_hidden_channels, mh_hidden_channels * 2, 1]))
        
        self.heads = heads
        self.mh_hidden_channels = mh_hidden_channels
        self.dropout_attn_score = dropout_attn_score
        self.dense_strategy = _validate_motifpool_dense_strategy(dense_strategy)
        self._dense_cache = None
        self._dense_cache_shape = None
        self._mask_cache = None
        self._mask_cache_slots = 0

    def reset_parameters(self):
        _reset_layers((self.pe_lin, self.pe_norm, self.atom_proj, self.atom_norm, self.clique_norm))
        self.attn._reset_parameters()
        _reset_layers(self.score_proj)

    def set_dense_strategy(self, strategy: str) -> None:
        self.dense_strategy = _validate_motifpool_dense_strategy(strategy)

    def _ensure_dense_cache(self, x: Tensor, slots: int) -> tuple[Tensor, Tensor]:
        feature_shape = tuple(x.shape[1:])
        cache_shape = (x.device, x.dtype, feature_shape)
        if (
            self._dense_cache is None
            or self._dense_cache.size(0) < slots
            or self._dense_cache_shape != cache_shape
        ):
            self._dense_cache = x.new_zeros((slots, *feature_shape))
            self._dense_cache_shape = cache_shape
        if (
            self._mask_cache is None
            or self._mask_cache.device != x.device
            or self._mask_cache_slots < slots
        ):
            self._mask_cache = torch.zeros(slots, dtype=torch.bool, device=x.device)
            self._mask_cache_slots = slots
        return self._dense_cache[:slots], self._mask_cache[:slots]

    def _prealloc_scatter_to_dense(
        self,
        x: Tensor,
        batch: Tensor,
        ptr: Tensor | None = None,
        layout: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if x.size(0) == 0:
            return (
                x.new_empty((0, 0, *x.shape[1:])),
                torch.empty((0, 0), dtype=torch.bool, device=x.device),
            )

        use_precomputed_layout = layout is not None
        if layout is None:
            layout = _build_motifpool_dense_layout(batch, ptr, include_mask=False)

        batch_size = int(layout["batch_size"])
        max_nodes = int(layout["max_nodes"])
        slots = int(layout["slots"])
        flat_index = layout["flat_index"]
        dense_flat, mask_flat = self._ensure_dense_cache(x, slots)

        # Padding values are not zero-filled on reuse: MultiheadAttention masks
        # padded keys, and padded query outputs are dropped by h[mask].
        dense_flat.index_copy_(0, flat_index, x)
        if use_precomputed_layout:
            mask = layout["mask"]
        else:
            mask_flat.zero_()
            mask_flat.index_fill_(0, flat_index, True)
            mask = mask_flat.view(batch_size, max_nodes)
        return (
            dense_flat.view(batch_size, max_nodes, *x.shape[1:]),
            mask,
        )

    def _dense_batch(
        self,
        x: Tensor,
        batch: Tensor,
        ptr: Tensor | None = None,
        layout: dict[str, Any] | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.dense_strategy == "pyg_to_dense":
            return to_dense_batch(x, batch)
        return self._prealloc_scatter_to_dense(x, batch, ptr, layout)

    def forward_dense_through(
        self,
        x: Tensor,
        clique_x: Tensor,
        clique_pe: Tensor,
        atom2clique_index: Tensor,
        layout: dict[str, Any],
        atom2clique_flat_index: Tensor,
        nvtx_enabled: bool = False,
    ):
        row, _ = atom2clique_index
        H = self.heads
        C = self.mh_hidden_channels
        batch_size = int(layout["batch_size"])
        max_nodes = int(layout["max_nodes"])
        slots = int(layout["slots"])
        flat_index = layout["flat_index"]
        mask = layout["mask"]

        with _nvtx_range("motif_pool_pe_encode", nvtx_enabled):
            clique_pe = self.pe_norm(self.pe_lin(clique_pe))
            clique_pe = _sparse_to_dense_with_layout(clique_pe, layout)
        with _nvtx_range("motif_pool_atom_to_clique_scatter", nvtx_enabled):
            clique_hx = scatter(
                x[row],
                atom2clique_flat_index,
                dim=0,
                dim_size=slots,
                reduce='mean',
            ).view(batch_size, max_nodes, -1)
            clique_hx = self.atom_norm(self.atom_proj(clique_hx))
            clique_x = clique_x + clique_pe + clique_hx

        with _nvtx_range("motif_pool_dense_batch", nvtx_enabled):
            h = clique_x
        with _nvtx_range("motif_pool_attention", nvtx_enabled):
            h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
        with _nvtx_range("motif_pool_restore_sparse", nvtx_enabled):
            clique_x = self.clique_norm(clique_x + h)
            clique_x_by_head = clique_x.view(batch_size, max_nodes, H, C)

        with _nvtx_range("motif_pool_score_projection", nvtx_enabled):
            clique_x_by_head_flat = clique_x_by_head.reshape(slots, H, C)
            score = torch.cat(
                [mlp(clique_x_by_head_flat[:, i]) for i, mlp in enumerate(self.score_proj)],
                dim=-1,
            ).view(batch_size, max_nodes, H)
        with _nvtx_range("motif_pool_softmax_pool", nvtx_enabled):
            score = F.dropout(score, p=self.dropout_attn_score, training=self.training)
            score = score.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            alpha_dense = torch.softmax(score, dim=1)
            alpha_dense = alpha_dense.masked_fill(~mask.unsqueeze(-1), 0.0)

            drug_feat = (clique_x_by_head * alpha_dense.unsqueeze(-1)).sum(dim=1)
            drug_feat = drug_feat.reshape(batch_size, H * C)
            alpha = alpha_dense.reshape(slots, H).index_select(0, flat_index)

        return drug_feat, clique_x, alpha

    def forward(
        self,
        x,
        clique_x,
        clique_pe,
        atom2clique_index,
        clique_batch,
        clique_ptr=None,
        clique_dense_layout=None,
        nvtx_enabled: bool = False,
    ):
        row, col = atom2clique_index
        H = self.heads
        C = self.mh_hidden_channels

        with _nvtx_range("motif_pool_pe_encode", nvtx_enabled):
            clique_pe = self.pe_norm(self.pe_lin(clique_pe))
        with _nvtx_range("motif_pool_atom_to_clique_scatter", nvtx_enabled):
            clique_hx = scatter(x[row], col, dim=0, dim_size=clique_x.size(0), reduce='mean')
            clique_hx = self.atom_norm(self.atom_proj(clique_hx))
            clique_x = clique_x + clique_pe + clique_hx

        with _nvtx_range("motif_pool_dense_batch", nvtx_enabled):
            h, mask = self._dense_batch(clique_x, clique_batch, clique_ptr, clique_dense_layout)
        with _nvtx_range("motif_pool_attention", nvtx_enabled):
            h, _ = self.attn(h, h, h, key_padding_mask=~mask,
                             need_weights=False)
        with _nvtx_range("motif_pool_restore_sparse", nvtx_enabled):
            h = h[mask]
            clique_x = clique_x + h
            clique_x = self.clique_norm(clique_x)
            clique_x_by_head = clique_x.view(-1, H, C)
        with _nvtx_range("motif_pool_score_projection", nvtx_enabled):
            score = torch.cat([mlp(clique_x_by_head[:, i]) for i, mlp in enumerate(self.score_proj)], dim=-1)
        with _nvtx_range("motif_pool_softmax_pool", nvtx_enabled):
            score = F.dropout(score, p=self.dropout_attn_score, training=self.training)
            alpha = softmax(score, clique_batch)

            drug_feat = clique_x_by_head * alpha.view(-1, H, 1)
            drug_feat = drug_feat.view(-1, H * C)
            drug_feat = global_add_pool(drug_feat, clique_batch)

        return drug_feat, clique_x, alpha


def _as_dense_batch(tensor):
    return tensor.unsqueeze(0) if tensor.dim() == 2 else tensor


def dense_mincut_pool(x, adj, s, mask=None, cluster_drop_node=None):
    """Dense MinCut protein pooling plus MinCut and orthogonality losses."""
    x = _as_dense_batch(x)
    adj = _as_dense_batch(adj)
    s = _as_dense_batch(s)

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        s = s * mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x_mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)

        if cluster_drop_node is not None:
            x_mask = cluster_drop_node.view(batch_size, num_nodes, 1).to(x.dtype)

        x = x * x_mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    mincut_num = _rank3_trace(out_adj)
    d_flat = torch.einsum('ijk->ij', adj)
    d = _rank3_diag(d_flat)
    mincut_den = _rank3_trace(
        torch.matmul(torch.matmul(s.transpose(1, 2), d), s))
    mincut_loss = -(mincut_num / mincut_den)
    mincut_loss = torch.mean(mincut_loss)

    ss = torch.matmul(s.transpose(1, 2), s)
    i_s = torch.eye(k).type_as(ss)
    ortho_loss = torch.norm(
        ss / torch.norm(ss, dim=(-1, -2), keepdim=True) -
        i_s / torch.norm(i_s), dim=(-1, -2))
    ortho_loss = torch.mean(ortho_loss)

    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return s, out, out_adj, mincut_loss, ortho_loss


def dense_mincut_pool_inference_no_loss(x, adj, s, mask=None, cluster_drop_node=None):
    """Prediction-equivalent dense MinCut pooling without auxiliary losses."""
    x = _as_dense_batch(x)
    adj = _as_dense_batch(adj)
    s = _as_dense_batch(s)

    (batch_size, num_nodes, _), k = x.size(), s.size(-1)

    s = torch.softmax(s, dim=-1)

    if mask is not None:
        s = s * mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x_mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)

        if cluster_drop_node is not None:
            x_mask = cluster_drop_node.view(batch_size, num_nodes, 1).to(x.dtype)

        x = x * x_mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    ind = torch.arange(k, device=out_adj.device)
    out_adj[:, ind, ind] = 0
    d = torch.einsum('ijk->ij', out_adj)
    d = torch.sqrt(d)[:, None] + EPS
    out_adj = (out_adj / d) / d.transpose(1, 2)

    return s, out, out_adj


def _rank3_trace(x):
    return torch.einsum('ijj->i', x)


def _rank3_diag(x):
    eye = torch.eye(x.size(1)).type_as(x)
    out = eye * x.unsqueeze(2).expand(*x.size(), x.size(1))
    return out


def _rbf(
    distance: torch.Tensor,
    D_min: float = 0.0,
    D_max: float = 1.0,
    D_count: int = 16,
) -> torch.Tensor:
    """Reference RBF helper retained for parity diagnostics."""
    distance = torch.clamp(distance.float(), max=D_max)
    device = distance.device
    D_mu = torch.linspace(D_min, D_max, D_count, device=device, dtype=distance.dtype)
    D_mu = D_mu.view(1, -1)
    D_sigma = (D_max - D_min) / D_count
    D_expand = distance.unsqueeze(-1)
    return torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)




class PSICHICPlusNet(nn.Module):
    """Inference-focused PSICHIC+ architecture with local compile-friendly cleanup."""

    def __init__(
        self,
        mol_deg,
        prot_deg,
        mol_in_channels: int = 43,
        mol_edge_channels: int = 30,
        clique_pe_walk_length: int = 20,
        prot_in_channels: int = 33,
        prot_evo_channels: int = 1280,
        hidden_channels: int = 200,
        pre_layers: int = 2,
        post_layers: int = 1,
        aggregators=None,
        scalers=None,
        total_layer: int = 3,
        K=None,
        t: int = 1,
        heads: int = 5,
        dropout: float = 0.0,
        dropout_attn_score: float = 0.2,
        drop_residue: float = 0.0,
        dropout_cluster_edge: float = 0.0,
        gaussian_noise: float = 0.0,
        device: str = "cuda:0",
        motifpool_dense_strategy: str = "pyg_to_dense",
        protein_mincut_strategy: str = "pyg_dense_mincut",
        protein_dense_cache_strategy: str = "none",
    ) -> None:
        super().__init__()

        if aggregators is None:
            aggregators = ["mean", "min", "max", "std"]
        if scalers is None:
            scalers = ["identity", "amplification", "linear"]
        if K is None:
            K = [5, 10, 20]
        if isinstance(K, int):
            K = [K] * total_layer

        self.total_layer = total_layer
        self.num_cluster = K
        self.t = t
        self.dropout = dropout
        self.drop_residue = drop_residue
        self.dropout_cluster_edge = dropout_cluster_edge
        self.gaussian_noise = gaussian_noise
        self.prot_edge_dim = hidden_channels
        self.forward_nvtx_enabled = False
        self.motifpool_dense_strategy = _validate_motifpool_dense_strategy(motifpool_dense_strategy)
        self.protein_mincut_strategy = _validate_protein_mincut_strategy(protein_mincut_strategy)
        self.protein_dense_cache_strategy = _validate_protein_dense_cache_strategy(protein_dense_cache_strategy)
        self._protein_dense_input_cache: dict[tuple[Any, ...], dict[str, Tensor | int]] = {}
        self._protein_dense_cache_hits = 0
        self._protein_dense_cache_misses = 0
        self._protein_dense_cache_rows = 0
        self._protein_dense_cache_bypasses = 0
        # Precomputed RBF centers for residue edge-feature construction. Registered
        # as a non-persistent buffer so the state_dict shape matches the legacy
        # checkpoint (strict load still passes) while eliminating a per-forward
        # torch.linspace allocation in the hot path. The division-by-D_sigma
        # sequence inside _apply_rbf stays identical to the module-level _rbf so
        # outputs are bit-equivalent at fp32.
        rbf_centers = torch.linspace(0.0, 1.0, self.prot_edge_dim, dtype=torch.float32).view(1, -1)
        self.register_buffer("_rbf_centers", rbf_centers, persistent=False)
        self._rbf_sigma = 1.0 / self.prot_edge_dim

        self.atom_type_encoder = nn.Embedding(20, hidden_channels)
        self.atom_feat_encoder = MLP([mol_in_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.clique_encoder = nn.Embedding(4, hidden_channels)

        self.prot_evo = MLP([prot_evo_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.prot_aa = MLP([prot_in_channels, hidden_channels * 2, hidden_channels], out_norm=True)

        for name in (
            "mol_convs", "prot_convs", "mol_gn2", "prot_gn2",
            "inter_convs", "cluster", "mol_pools", "prot_norms",
            "atom_lins", "residue_lins", "c2a_mlps", "c2r_mlps",
        ):
            setattr(self, name, nn.ModuleList())

        for idx in range(total_layer):
            self._append_interaction_layer(
                idx, mol_deg, prot_deg, mol_edge_channels, hidden_channels,
                pre_layers, post_layers, aggregators, scalers, heads,
                clique_pe_walk_length, dropout, dropout_attn_score, t,
                self.motifpool_dense_strategy,
            )

        self.atom_attn_lin = PosLinear(heads * total_layer, 1, bias=False, init_value=1 / heads)
        self.residue_attn_lin = PosLinear(heads * total_layer, 1, bias=False, init_value=1 / heads)

        self.mol_out = MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.prot_out = MLP([hidden_channels, hidden_channels * 2, hidden_channels], out_norm=True)
        self.mu_out = MLP([hidden_channels * 2, hidden_channels, 1])
        self.sigma_out = MLP([hidden_channels * 2, hidden_channels, 1])
        self.softplus = nn.Softplus()

    def _append_interaction_layer(
        self,
        idx,
        mol_deg,
        prot_deg,
        mol_edge_channels,
        hidden_channels,
        pre_layers,
        post_layers,
        aggregators,
        scalers,
        heads,
        clique_pe_walk_length,
        dropout,
        dropout_attn_score,
        t,
        motifpool_dense_strategy,
    ):
        pna_kwargs = dict(
            pre_layers=pre_layers,
            post_layers=post_layers,
            aggregators=aggregators,
            scalers=scalers,
            num_towers=heads,
            dropout=dropout,
        )
        self.mol_convs.append(
            Drug_PNAConv(mol_deg, hidden_channels, edge_channels=mol_edge_channels, **pna_kwargs)
        )
        self.prot_convs.append(
            Protein_PNAConv(prot_deg, hidden_channels, edge_channels=hidden_channels, **pna_kwargs)
        )
        self.cluster.append(GCNCluster([hidden_channels, hidden_channels * 2, self.num_cluster[idx]], in_norm=True))
        self.inter_convs.append(
            DrugProteinConv(
                atom_channels=hidden_channels,
                residue_channels=hidden_channels,
                heads=heads,
                t=t,
                dropout_attn_score=dropout_attn_score,
            )
        )
        self.mol_pools.append(
            MotifPool(
                hidden_channels,
                heads,
                clique_pe_walk_length,
                dropout_attn_score,
                dense_strategy=motifpool_dense_strategy,
            )
        )
        self.prot_norms.append(LayerNorm(hidden_channels))
        self.atom_lins.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.residue_lins.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.c2a_mlps.append(MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False))
        self.c2r_mlps.append(MLP([hidden_channels, hidden_channels * 2, hidden_channels], bias=False))
        self.mol_gn2.append(GraphNorm(hidden_channels))
        self.prot_gn2.append(GraphNorm(hidden_channels))

    def set_forward_nvtx(self, enabled: bool) -> None:
        self.forward_nvtx_enabled = bool(enabled)

    def set_motifpool_dense_strategy(self, strategy: str) -> None:
        self.motifpool_dense_strategy = _validate_motifpool_dense_strategy(strategy)
        for pool in self.mol_pools:
            pool.set_dense_strategy(self.motifpool_dense_strategy)

    def set_protein_mincut_strategy(self, strategy: str) -> None:
        self.protein_mincut_strategy = _validate_protein_mincut_strategy(strategy)
        self.clear_protein_dense_cache()

    def set_protein_dense_cache_strategy(self, strategy: str) -> None:
        self.protein_dense_cache_strategy = _validate_protein_dense_cache_strategy(strategy)
        self.clear_protein_dense_cache()

    def clear_protein_dense_cache(self) -> None:
        self._protein_dense_input_cache.clear()
        self._protein_dense_cache_hits = 0
        self._protein_dense_cache_misses = 0
        self._protein_dense_cache_rows = 0
        self._protein_dense_cache_bypasses = 0

    def protein_dense_cache_stats(self) -> dict[str, int | str | bool]:
        return {
            "strategy": self.protein_dense_cache_strategy,
            "enabled": self.protein_dense_cache_strategy != "none",
            "entries": len(self._protein_dense_input_cache),
            "hits": self._protein_dense_cache_hits,
            "misses": self._protein_dense_cache_misses,
            "rows": self._protein_dense_cache_rows,
            "bypasses": self._protein_dense_cache_bypasses,
        }

    def _apply_rbf(self, distance: torch.Tensor) -> torch.Tensor:
        """Hot-path RBF using precomputed centers on the module device.

        Bit-equivalent to the module-level ``_rbf`` helper when ``D_min=0``,
        ``D_max=1``, and ``D_count=self.prot_edge_dim``. Only the D_mu buffer
        is cached; the division-by-D_sigma sequence is preserved so fp32
        output matches the frozen baseline to the last ULP.
        """
        distance = torch.clamp(distance.float(), max=1.0)
        D_expand = distance.unsqueeze(-1)
        return torch.exp(-((D_expand - self._rbf_centers) / self._rbf_sigma) ** 2)

    def _protein_dense_cache_enabled(self, data, save_cluster: bool) -> bool:
        if self.protein_dense_cache_strategy != "protein_dense_inputs":
            return False
        if self.training or save_cluster:
            self._protein_dense_cache_bypasses += 1
            return False
        prot_keys = getattr(data, "prot_key", None)
        if prot_keys is None:
            self._protein_dense_cache_bypasses += 1
            return False
        return True

    def _protein_dense_cache_key(
        self,
        prot_key: str,
        residue_x: Tensor,
        residue_evo_x: Tensor,
        residue_edge_weight: Tensor,
    ) -> tuple[Any, ...]:
        return (
            str(prot_key),
            str(residue_x.device),
            residue_x.dtype,
            residue_evo_x.dtype,
            residue_edge_weight.dtype,
            self.protein_mincut_strategy,
        )

    def _compute_layer0_protein_dense_entry(
        self,
        residue_x: Tensor,
        residue_evo_x: Tensor,
        residue_edge_index: Tensor,
        residue_edge_weight: Tensor,
    ) -> dict[str, Tensor | int]:
        num_nodes = int(residue_x.size(0))
        residue_edge_attr = self._apply_rbf(residue_edge_weight)
        residue_x = self.prot_aa(residue_x) + self.prot_evo(residue_evo_x)
        residue_x = self.prot_convs[0](residue_x, residue_edge_index, residue_edge_attr)
        s_logits = self.cluster[0](residue_x, residue_edge_index)

        residue_hx = residue_x.unsqueeze(0)
        residue_mask = torch.ones((1, num_nodes), dtype=torch.bool, device=residue_x.device)
        s_dense = s_logits.unsqueeze(0)
        residue_adj = to_dense_adj(residue_edge_index, max_num_nodes=num_nodes)

        if self.protein_mincut_strategy == "inference_no_loss":
            s_dense, cluster_x, _ = dense_mincut_pool_inference_no_loss(
                residue_hx,
                residue_adj.float(),
                s_dense.float(),
                residue_mask,
                None,
            )
            cluster_loss = torch.zeros((), device=residue_x.device, dtype=torch.float32)
            ortho_loss = torch.zeros((), device=residue_x.device, dtype=torch.float32)
        else:
            s_dense, cluster_x, _, cluster_loss, ortho_loss = dense_mincut_pool(
                residue_hx,
                residue_adj.float(),
                s_dense.float(),
                residue_mask,
                None,
            )

        cluster_x = self.prot_norms[0](cluster_x)
        return {
            "num_nodes": num_nodes,
            "residue_x": residue_x.detach(),
            "s": s_dense.squeeze(0).detach(),
            "cluster_x": cluster_x.squeeze(0).detach(),
            "cluster_loss": cluster_loss.detach(),
            "ortho_loss": ortho_loss.detach(),
        }

    def _protein_node_ptr(self, prot_batch: Tensor, ptr: Tensor | None) -> Tensor:
        if ptr is not None:
            return ptr
        batch_size = int(prot_batch[-1].item()) + 1 if prot_batch.numel() else 0
        counts = torch.bincount(prot_batch, minlength=batch_size)
        return torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)])

    def _get_layer0_protein_dense_cache(
        self,
        prot_keys,
        raw_residue_x: Tensor,
        raw_residue_evo_x: Tensor,
        residue_edge_index: Tensor,
        residue_edge_weight: Tensor,
        prot_batch: Tensor,
        prot_ptr: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if isinstance(prot_keys, Tensor):
            prot_keys = [str(value.item()) for value in prot_keys]
        elif isinstance(prot_keys, str):
            prot_keys = [prot_keys]
        else:
            prot_keys = [str(value) for value in prot_keys]

        ptr = self._protein_node_ptr(prot_batch, prot_ptr)
        entries: list[dict[str, Tensor | int]] = []
        first_row_for_key: dict[str, int] = {}
        for row_index, prot_key in enumerate(prot_keys):
            first_row_for_key.setdefault(prot_key, row_index)

        for prot_key in prot_keys:
            source_row = first_row_for_key[prot_key]
            node_start = int(ptr[source_row].item())
            node_end = int(ptr[source_row + 1].item())
            key = self._protein_dense_cache_key(
                prot_key,
                raw_residue_x,
                raw_residue_evo_x,
                residue_edge_weight,
            )
            entry = self._protein_dense_input_cache.get(key)
            if entry is None:
                edge_mask = (
                    (residue_edge_index[0] >= node_start)
                    & (residue_edge_index[0] < node_end)
                    & (residue_edge_index[1] >= node_start)
                    & (residue_edge_index[1] < node_end)
                )
                local_edge_index = residue_edge_index[:, edge_mask] - node_start
                local_edge_weight = residue_edge_weight[edge_mask]
                entry = self._compute_layer0_protein_dense_entry(
                    raw_residue_x[node_start:node_end],
                    raw_residue_evo_x[node_start:node_end],
                    local_edge_index,
                    local_edge_weight,
                )
                self._protein_dense_input_cache[key] = entry
                self._protein_dense_cache_misses += 1
            else:
                self._protein_dense_cache_hits += 1
            entries.append(entry)

        batch_size = len(entries)
        max_nodes = max(int(entry["num_nodes"]) for entry in entries) if entries else 0
        residue_x = torch.cat([entry["residue_x"] for entry in entries], dim=0)
        cluster_x = torch.stack([entry["cluster_x"] for entry in entries], dim=0)
        s_template = entries[0]["s"]
        s = s_template.new_zeros((batch_size, max_nodes, s_template.size(-1)))
        residue_mask = torch.zeros((batch_size, max_nodes), dtype=torch.bool, device=s_template.device)
        for row_index, entry in enumerate(entries):
            num_nodes = int(entry["num_nodes"])
            s[row_index, :num_nodes] = entry["s"]
            residue_mask[row_index, :num_nodes] = True

        cluster_losses = torch.stack([entry["cluster_loss"] for entry in entries])
        ortho_losses = torch.stack([entry["ortho_loss"] for entry in entries])
        self._protein_dense_cache_rows += batch_size
        return (
            residue_x,
            s,
            residue_mask,
            cluster_x,
            cluster_losses.mean(),
            ortho_losses.mean(),
        )

    def forward(self, data, save_cluster: bool = False):
        use_nvtx = bool(getattr(self, "forward_nvtx_enabled", False))

        with _nvtx_range("forward_embed_inputs", use_nvtx):
            device = data.mol_x.device

            mol_x = data.mol_x
            mol_x_feat = data.mol_x_feat
            bond_x = data.mol_edge_attr
            atom_edge_index = data.mol_edge_index

            clique_x = data.clique_x
            clique_x_pe = data.clique_x_pe
            atom2clique_index = data.atom2clique_index
            clique_ptr = getattr(data, "clique_x_ptr", None)

            raw_residue_x = data.prot_node_aa
            raw_residue_evo_x = data.prot_node_evo
            residue_edge_index = data.prot_edge_index
            residue_edge_weight = data.prot_edge_weight

            mol_batch = data.mol_x_batch
            prot_batch = data.prot_node_aa_batch
            prot_ptr = getattr(data, "prot_node_aa_ptr", None)
            clique_batch = data.clique_x_batch

            residue_edge_attr = self._apply_rbf(residue_edge_weight)
            use_protein_dense_cache = self._protein_dense_cache_enabled(data, save_cluster)
            if use_protein_dense_cache:
                residue_x = None
            else:
                residue_x = self.prot_aa(raw_residue_x) + self.prot_evo(raw_residue_evo_x)
            atom_x = self.atom_type_encoder(mol_x.squeeze()) + self.atom_feat_encoder(mol_x_feat)
            clique_x = self.clique_encoder(clique_x.squeeze())

            ortho_loss = torch.zeros((), device=device, dtype=torch.float32)
            cluster_loss = torch.zeros((), device=device, dtype=torch.float32)

            clique_scores = []
            residue_scores = []
            layer_s = {}
            use_dense_through = self.motifpool_dense_strategy == "dense_through"
            clique_dense_layout = (
                _build_motifpool_dense_layout(clique_batch, clique_ptr)
                if self.motifpool_dense_strategy in {"precomputed_scatter", "dense_through"}
                else None
            )
            atom2clique_flat_index = None
            if use_dense_through:
                clique_x = _sparse_to_dense_with_layout(clique_x, clique_dense_layout)
                atom2clique_flat_index = clique_dense_layout["flat_index"].index_select(0, atom2clique_index[1])

        for idx in range(self.total_layer):
            with _nvtx_range(f"layer_{idx}_ligand_pna", use_nvtx):
                atom_x = self.mol_convs[idx](atom_x, bond_x, atom_edge_index)

            if idx == 0 and use_protein_dense_cache:
                with _nvtx_range("layer_0_protein_dense_cache", use_nvtx):
                    (
                        residue_x,
                        s,
                        residue_mask,
                        cluster_x,
                        cl_loss,
                        o_loss,
                    ) = self._get_layer0_protein_dense_cache(
                        data.prot_key,
                        raw_residue_x,
                        raw_residue_evo_x,
                        residue_edge_index,
                        residue_edge_weight,
                        prot_batch,
                        prot_ptr,
                    )
                    if self.protein_mincut_strategy != "inference_no_loss":
                        ortho_loss = ortho_loss + o_loss.float() / self.total_layer
                        cluster_loss = cluster_loss + cl_loss.float() / self.total_layer
            else:
                with _nvtx_range(f"layer_{idx}_protein_pna", use_nvtx):
                    residue_x = self.prot_convs[idx](residue_x, residue_edge_index, residue_edge_attr)

            with _nvtx_range(f"layer_{idx}_motif_pool", use_nvtx):
                if use_dense_through:
                    drug_x, clique_x, clique_score = self.mol_pools[idx].forward_dense_through(
                        atom_x,
                        clique_x,
                        clique_x_pe,
                        atom2clique_index,
                        clique_dense_layout,
                        atom2clique_flat_index,
                        nvtx_enabled=use_nvtx,
                    )
                else:
                    drug_x, clique_x, clique_score = self.mol_pools[idx](
                        atom_x,
                        clique_x,
                        clique_x_pe,
                        atom2clique_index,
                        clique_batch,
                        clique_ptr=clique_ptr,
                        clique_dense_layout=clique_dense_layout,
                        nvtx_enabled=use_nvtx,
                    )
                clique_scores.append(clique_score)

            if not (idx == 0 and use_protein_dense_cache):
                with _nvtx_range(f"layer_{idx}_cluster_gcn", use_nvtx):
                    dropped_residue_edge_index, _ = dropout_edge(
                        residue_edge_index,
                        p=self.dropout_cluster_edge,
                        force_undirected=True,
                        training=self.training,
                    )
                    s = self.cluster[idx](residue_x, dropped_residue_edge_index)

                with _nvtx_range(f"layer_{idx}_protein_dense_setup", use_nvtx):
                    residue_hx, residue_mask = to_dense_batch(residue_x, prot_batch)
                    if save_cluster:
                        layer_s[idx] = s

                    s, _ = to_dense_batch(s, prot_batch)
                    residue_adj = to_dense_adj(residue_edge_index, prot_batch)
                    cluster_mask = residue_mask

                    cluster_drop_mask = None
                    if self.drop_residue != 0 and self.training:
                        _, _, residue_drop_mask = dropout_node(
                            residue_edge_index,
                            self.drop_residue,
                            residue_x.size(0),
                            prot_batch,
                            self.training,
                        )
                        residue_drop_mask, _ = to_dense_batch(residue_drop_mask.reshape(-1, 1), prot_batch)
                        cluster_drop_mask = residue_mask * residue_drop_mask.squeeze()

                with _nvtx_range(f"layer_{idx}_dense_mincut", use_nvtx):
                    if self.protein_mincut_strategy == "inference_no_loss" and not self.training:
                        s, cluster_x, _residue_adj = dense_mincut_pool_inference_no_loss(
                            residue_hx,
                            residue_adj.float(),
                            s.float(),
                            cluster_mask,
                            cluster_drop_mask,
                        )
                    else:
                        s, cluster_x, _residue_adj, cl_loss, o_loss = dense_mincut_pool(
                            residue_hx,
                            residue_adj.float(),
                            s.float(),
                            cluster_mask,
                            cluster_drop_mask,
                        )
                        ortho_loss = ortho_loss + o_loss.float() / self.total_layer
                        cluster_loss = cluster_loss + cl_loss.float() / self.total_layer
                    cluster_x = self.prot_norms[idx](cluster_x)

            with _nvtx_range(f"layer_{idx}_drug_protein_interaction", use_nvtx):
                batch_size = s.size(0)
                num_k = self.num_cluster[idx]
                cluster_residue_batch = torch.arange(batch_size, device=device).repeat_interleave(num_k)
                cluster_x = cluster_x.reshape(batch_size * num_k, -1)
                p2m_edge_index = torch.stack(
                    [
                        torch.arange(batch_size * num_k, device=device),
                        cluster_residue_batch,
                    ]
                )

                if use_dense_through:
                    clique_x, cluster_x, inter_attn = self.inter_convs[idx].forward_dense_clique(
                        drug_x,
                        clique_x,
                        clique_dense_layout["mask"],
                        cluster_x,
                        p2m_edge_index,
                    )
                else:
                    clique_x, cluster_x, inter_attn = self.inter_convs[idx](
                        drug_x,
                        clique_x,
                        clique_batch,
                        cluster_x,
                        p2m_edge_index,
                    )
                inter_attn = inter_attn[1]

            with _nvtx_range(f"layer_{idx}_scatter_back_projection", use_nvtx):
                row, col = atom2clique_index
                if use_dense_through:
                    clique_x_for_atom = clique_x.reshape(int(clique_dense_layout["slots"]), -1).index_select(
                        0,
                        atom2clique_flat_index,
                    )
                else:
                    clique_x_for_atom = clique_x[col]
                atom_x = atom_x + F.relu(
                    self.atom_lins[idx](
                        scatter(
                            clique_x_for_atom,
                            row,
                            dim=0,
                            dim_size=atom_x.size(0),
                            reduce="mean",
                        )
                    )
                )
                atom_x = atom_x + self.c2a_mlps[idx](atom_x)

                residue_hx, _ = to_dense_batch(cluster_x, cluster_residue_batch)
                residue_x = residue_x + F.relu(self.residue_lins[idx]((s @ residue_hx)[residue_mask]))
                residue_x = residue_x + self.c2r_mlps[idx](residue_x)

            with _nvtx_range(f"layer_{idx}_graph_norm_and_scores", use_nvtx):
                atom_x = self.mol_gn2[idx](atom_x, mol_batch)
                residue_x = self.prot_gn2[idx](residue_x, prot_batch)

                inter_attn, _ = to_dense_batch(inter_attn, cluster_residue_batch)
                inter_attn = (s @ inter_attn)[residue_mask]
                residue_scores.append(inter_attn)

        with _nvtx_range("forward_final_attention_pool", use_nvtx):
            row, col = atom2clique_index
            clique_scores = torch.cat(clique_scores, dim=-1)
            atom_scores = scatter(clique_scores[col], row, dim=0, dim_size=atom_x.size(0), reduce="mean")
            atom_score = softmax(self.atom_attn_lin(atom_scores), mol_batch)
            mol_pool_feat = global_add_pool(atom_x * atom_score, mol_batch)

            residue_scores = torch.cat(residue_scores, dim=-1)
            residue_score = softmax(self.residue_attn_lin(residue_scores), prot_batch)
            prot_pool_feat = global_add_pool(residue_x * residue_score, prot_batch)

        with _nvtx_range("forward_final_beta_heads", use_nvtx):
            mol_pool_feat = self.mol_out(mol_pool_feat)
            prot_pool_feat = self.prot_out(prot_pool_feat)
            mol_prot_feat = torch.cat([mol_pool_feat, prot_pool_feat], dim=-1)

            mu_out = self.mu_out(mol_prot_feat).float()
            sigma_out = self.sigma_out(mol_prot_feat).float()
            mu = torch.sigmoid(mu_out)
            phi = self.softplus(sigma_out)
            reg_alpha = (mu * phi + 1.0).squeeze()
            reg_beta = ((1.0 - mu) * phi + 1.0).squeeze()

        with _nvtx_range("forward_attention_dict", use_nvtx):
            attention_dict = {
                "residue_final_score": residue_score,
                "atom_final_score": atom_score,
                "clique_layer_scores": clique_scores,
                "residue_layer_scores": residue_scores,
                "drug_atom_index": mol_batch,
                "drug_clique_index": clique_batch,
                "protein_residue_index": prot_batch,
                "mol_feature": mol_pool_feat,
                "prot_feature": prot_pool_feat,
                "cluster_s": layer_s,
                "interaction_fingerprint": mol_prot_feat,
            }

        return reg_alpha, reg_beta, ortho_loss, cluster_loss, attention_dict


class PSICHICPlusModel(nn.Module):
    """Production-facing wrapper around the canonical local PSICHIC+ model."""

    def __init__(
        self,
        model_dir: Path,
        device: torch.device | str,
        motifpool_dense_strategy: str = "pyg_to_dense",
        protein_mincut_strategy: str = "pyg_dense_mincut",
        protein_dense_cache_strategy: str = "none",
    ) -> None:
        super().__init__()
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.motifpool_dense_strategy = _validate_motifpool_dense_strategy(motifpool_dense_strategy)
        self.protein_mincut_strategy = _validate_protein_mincut_strategy(protein_mincut_strategy)
        self.protein_dense_cache_strategy = _validate_protein_dense_cache_strategy(protein_dense_cache_strategy)
        self.config = self._load_config()
        self.model = self._build_model()

    def _load_config(self) -> dict:
        with open(self.model_dir / "config.json") as handle:
            return json.load(handle)

    def _build_model(self) -> nn.Module:
        degree_dict = torch.load(self.model_dir / "degree.pt", map_location=self.device, weights_only=False)
        params = self.config["params"]
        model = PSICHICPlusNet(
            degree_dict["ligand_deg"],
            degree_dict["protein_deg"],
            mol_in_channels=params["mol_in_channels"],
            mol_edge_channels=params["mol_edge_channels"],
            clique_pe_walk_length=params["clique_pe_walk_length"],
            prot_in_channels=params["prot_in_channels"],
            prot_evo_channels=params["prot_evo_channels"],
            hidden_channels=params["hidden_channels"],
            pre_layers=params["pre_layers"],
            post_layers=params["post_layers"],
            aggregators=params["aggregators"],
            scalers=params["scalers"],
            total_layer=params["total_layer"],
            K=params["K"],
            heads=params["heads"],
            dropout=params["dropout"],
            dropout_attn_score=params["dropout_attn_score"],
            device=str(self.device),
            motifpool_dense_strategy=self.motifpool_dense_strategy,
            protein_mincut_strategy=self.protein_mincut_strategy,
            protein_dense_cache_strategy=self.protein_dense_cache_strategy,
        ).to(self.device)
        state_dict = torch.load(self.model_dir / "model.pt", map_location=self.device, weights_only=False)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        device: torch.device | str = "cuda:0",
        motifpool_dense_strategy: str = "pyg_to_dense",
        protein_mincut_strategy: str = "pyg_dense_mincut",
        protein_dense_cache_strategy: str = "none",
    ) -> "PSICHICPlusModel":
        return cls(
            Path(model_dir),
            device,
            motifpool_dense_strategy=motifpool_dense_strategy,
            protein_mincut_strategy=protein_mincut_strategy,
            protein_dense_cache_strategy=protein_dense_cache_strategy,
        )

    def forward(self, data):
        return self.model(data)

    def set_forward_nvtx(self, enabled: bool) -> None:
        if hasattr(self.model, "set_forward_nvtx"):
            self.model.set_forward_nvtx(enabled)
        else:
            setattr(self.model, "forward_nvtx_enabled", bool(enabled))

    def set_motifpool_dense_strategy(self, strategy: str) -> None:
        self.motifpool_dense_strategy = _validate_motifpool_dense_strategy(strategy)
        if hasattr(self.model, "set_motifpool_dense_strategy"):
            self.model.set_motifpool_dense_strategy(self.motifpool_dense_strategy)

    def set_protein_mincut_strategy(self, strategy: str) -> None:
        self.protein_mincut_strategy = _validate_protein_mincut_strategy(strategy)
        if hasattr(self.model, "set_protein_mincut_strategy"):
            self.model.set_protein_mincut_strategy(self.protein_mincut_strategy)

    def set_protein_dense_cache_strategy(self, strategy: str) -> None:
        self.protein_dense_cache_strategy = _validate_protein_dense_cache_strategy(strategy)
        if hasattr(self.model, "set_protein_dense_cache_strategy"):
            self.model.set_protein_dense_cache_strategy(self.protein_dense_cache_strategy)

    def protein_dense_cache_stats(self) -> dict[str, int | str | bool]:
        if hasattr(self.model, "protein_dense_cache_stats"):
            return self.model.protein_dense_cache_stats()
        return {
            "strategy": self.protein_dense_cache_strategy,
            "enabled": False,
            "entries": 0,
            "hits": 0,
            "misses": 0,
            "rows": 0,
            "bypasses": 0,
        }

    def predict_batch(self, data):
        reg_alpha, reg_beta, _, _, attention_dict = self.model(data)
        reg_pred = reg_alpha / (reg_alpha + reg_beta) * 12.0
        return reg_pred, reg_alpha, reg_beta, attention_dict
