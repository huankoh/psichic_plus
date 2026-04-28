from numbers import Number, Real

import torch
from torch.distributions import constraints
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.exp_family import ExponentialFamily
from torch.distributions.utils import broadcast_all

class Beta(ExponentialFamily):
    arg_constraints = {
        "concentration1": constraints.positive,
        "concentration0": constraints.positive,
    }
    support = constraints.unit_interval
    has_rsample = True

    def __init__(self, concentration1, concentration0, validate_args=None):
        if isinstance(concentration1, Real) and isinstance(concentration0, Real):
            concentration1_concentration0 = torch.tensor(
                [float(concentration1), float(concentration0)]
            )
        else:
            concentration1, concentration0 = broadcast_all(
                concentration1, concentration0
            )
            concentration1_concentration0 = torch.stack(
                [concentration1, concentration0], -1
            )
        self._dirichlet = Dirichlet(
            concentration1_concentration0, validate_args=validate_args
        )
        super().__init__(self._dirichlet._batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Beta, _instance)
        batch_shape = torch.Size(batch_shape)
        new._dirichlet = self._dirichlet.expand(batch_shape)
        super(Beta, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @property
    def mean(self):
        return self.concentration1 / (self.concentration1 + self.concentration0)

    @property
    def mode(self):
        return self._dirichlet.mode[..., 0]

    @property
    def variance(self):
        total = self.concentration1 + self.concentration0
        return self.concentration1 * self.concentration0 / (total.pow(2) * (total + 1))

    def rsample(self, sample_shape=()):
        return self._dirichlet.rsample(sample_shape).select(-1, 0)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        heads_tails = torch.stack([value, 1.0 - value], -1)
        return self._dirichlet.log_prob(heads_tails)

    def entropy(self):
        return self._dirichlet.entropy()

    @property
    def concentration1(self):
        return self._dirichlet.concentration[..., 0]

    @property
    def concentration0(self):
        return self._dirichlet.concentration[..., 1]

    @property
    def _natural_params(self):
        return (self.concentration1, self.concentration0)

    def _log_normalizer(self, x, y):
        return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)

    def cdf(self, value, npts=500):
        if self._validate_args:
            self._validate_sample(value)
        x = torch.linspace(0, value, npts, device=value.device)
        prob = self.log_prob(x.unsqueeze(-1)).exp()
        return torch.trapz(prob, x, dim=0)