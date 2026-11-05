"""Predict PSICHIC's PyG source without constructing or executing a model."""

import hashlib
from pathlib import Path


def expected_source():
    from jinja2 import Environment, FileSystemLoader
    import psichic_model
    import torch_geometric
    from torch_geometric.inspector import Inspector, Signature
    from torch_geometric.nn.conv import message_passing

    from applications.scoring.runtime import RuntimeMismatch

    if torch_geometric.__version__ != '2.7.0':
        raise RuntimeMismatch('Generated PSICHIC source requires the qualified PyG 2.7.0 renderer')
    template = Environment(loader=FileSystemLoader(
        Path(message_passing.__file__).resolve().parent)).get_template('propagate.jinja')
    records = []
    contracts = ((psichic_model.PNAConv, False), (psichic_model.DrugProteinConv, False),
                 (psichic_model.GCNConv, True))
    for cls, fuse in contracts:
        inspector = Inspector(cls)
        for method, exclude in (('message', None), ('aggregate', [0, 'aggr']),
                                ('message_and_aggregate', [0]), ('update', [0]), ('edge_update', None)):
            inspector.inspect_signature(method, exclude=exclude)
        # GCNCluster uses GCNConv's default add aggregation, which enables fusion.
        if inspector.implements('message_and_aggregate') != fuse or inspector.implements('edge_update'):
            raise RuntimeMismatch(f'Unsupported generated source contract for {cls.__module__}.{cls.__name__}')
        update = inspector.get_signature('update')
        signature = Signature(
            inspector.get_params_from_method_call('propagate', exclude=[0, 'edge_index', 'size']),
            update.return_type, update.return_type_repr,
        )
        source = template.render(
            modules=inspector._modules, collect_name='collect', signature=signature,
            collect_param_dict=inspector.get_flat_param_dict(['message', 'aggregate', 'update']),
            message_args=inspector.get_param_names('message'),
            aggregate_args=inspector.get_param_names('aggregate'),
            message_and_aggregate_args=inspector.get_param_names('message_and_aggregate'),
            update_args=inspector.get_param_names('update'), fuse=fuse,
        )
        records.append({
            'module': f'{cls.__module__}_{cls.__name__}_propagate',
            'sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
            'generator': 'pyg-2.7.0-propagate-v1',
            'class': f'{cls.__module__}.{cls.__qualname__}',
            'template': 'torch_geometric/nn/conv/propagate.jinja', 'fuse': fuse,
        })
    return records
