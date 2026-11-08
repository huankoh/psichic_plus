"""Opt-in same-pass distribution transport; legacy requests remain lists."""

from dataclasses import asdict
import math

try:
    from ._outcomes import distribution_outcome
except ModuleNotFoundError as error:
    if error.name != __package__ + '._outcomes':
        raise
    from applications.scorer import distribution_outcome

VERSION = 1


def request(smiles, output):
    if output == 'affinity':
        return smiles
    if output != 'distribution':
        raise ValueError('Output must be affinity or distribution')
    return {'schema_version': VERSION, 'output': output, 'smiles': smiles}


def decode_request(value):
    if isinstance(value, list):
        return value, 'affinity'
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'output', 'smiles'}
            or type(value['schema_version']) is not int or value['schema_version'] != VERSION
            or value['output'] != 'distribution'):
        raise ValueError('Unsupported rich-output request protocol')
    return value['smiles'], value['output']


def distribution(affinity, alpha, beta):
    return asdict(distribution_outcome(affinity, alpha, beta))


def validate_distribution(affinity, value):
    if not isinstance(value, dict):
        raise ValueError('Missing requested distribution outcome')
    if value.get('state') == 'distribution-failure':
        if set(value) != {'state', 'reason', 'raw_alpha', 'raw_beta'} or any(
                not isinstance(value[k], str) or not value[k] for k in ('reason', 'raw_alpha', 'raw_beta')):
            raise ValueError('Invalid distribution failure')
        return value
    if set(value) != {'state', 'alpha', 'beta', 'concentration'} or value['state'] != 'valid':
        raise ValueError('Missing requested distribution fields')
    if (any(type(value[k]) not in (int, float) or not math.isfinite(value[k]) for k in ('alpha', 'beta', 'concentration'))
            or distribution(affinity, value['alpha'], value['beta']) != value):
        raise ValueError('Invalid distribution values')
    return value
