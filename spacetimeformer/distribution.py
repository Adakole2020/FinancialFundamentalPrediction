import math
from numbers import Number, Real

import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all


class SkewNormal(Distribution):
    r"""
    Creates a skew normal distribution parameterized by
    :attr:`loc`, :attr:`scale`, and :attr:`alpha`.

    Example::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> m = SkewNormal(torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.0]))
        >>> m.sample()  # skew normal distribution with loc=0, scale=1, and alpha=0
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
        alpha (float or Tensor): skewness parameter of the distribution
    """
    arg_constraints = {"loc": constraints.real, "scale": constraints.positive, "alpha": constraints.real}
    support = constraints.real
    has_rsample = True

    @property
    def mean(self):
        """
        Computes the mean of the skew normal distribution.

        Returns:
            Tensor: The mean of the distribution.
        """
        delta = self.alpha / torch.sqrt(1 + self.alpha.pow(2))
        return self.loc + self.scale * delta * math.sqrt(2 / math.pi)

    @property
    def stddev(self):
        """
        Computes the standard deviation of the skew normal distribution.

        Returns:
            Tensor: The standard deviation of the distribution.
        """
        return torch.sqrt(self.variance)
    
    @property
    def variance(self):
        """
        Computes the variance of the skew normal distribution.

        Returns:
            Tensor: The variance of the distribution.
        """
        delta = self.alpha / torch.sqrt(1 + self.alpha.pow(2))
        return self.scale.pow(2) * (1 - 2 * delta.pow(2) / math.pi)
    
    @property
    def skewness(self):
        """
        Computes the skewness of the skew normal distribution.

        Returns:
            Tensor: The skewness of the distribution.
        """
        delta = self.alpha / torch.sqrt(1 + self.alpha.pow(2))
        return (4 - math.pi) / 2 * (delta * torch.sqrt(2 / math.pi)).pow(3) / (1 - 2 * delta.pow(2) / math.pi).pow(1.5)

    def __init__(self, loc, scale, alpha, validate_args=None):
        """
        Initializes a skew normal distribution.

        Args:
            loc (float or Tensor): mean of the distribution (often referred to as mu)
            scale (float or Tensor): standard deviation of the distribution
                (often referred to as sigma)
            alpha (float or Tensor): skewness parameter of the distribution
            validate_args (bool, optional): Whether to validate input arguments. Default: None
        """
        self.loc, self.scale, self.alpha = broadcast_all(loc, scale, alpha)
        if isinstance(loc, Number) and isinstance(scale, Number) and isinstance(alpha, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        """
        Returns a new SkewNormal instance with expanded batch shape.

        Args:
            batch_shape (torch.Size): The desired expanded batch shape.
            _instance (SkewNormal, optional): The instance to be expanded. Default: None

        Returns:
            SkewNormal: The expanded SkewNormal instance.
        """
        new = self._get_checked_instance(SkewNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.alpha = self.alpha.expand(batch_shape)
        super(SkewNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()): 
        """
        Generates a sample from the skew normal distribution.

        Args:
            sample_shape (torch.Size, optional): The desired shape of the sample. Default: torch.Size()

        Returns:
            Tensor: A sample from the skew normal distribution.
        """
        # https://stats.stackexchange.com/questions/316314/sampling-from-skew-normal-distribution
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            U = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
            V = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        
            # Adjust for skewness using the alpha parameter
            Z = U + self.alpha * torch.abs(V)

            return self.loc + self.scale * Z / torch.sqrt(1 + self.alpha.pow(2))

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a reparameterized sample from the skew normal distribution.

        Args:
            sample_shape (torch.Size, optional): The desired shape of the sample. Default: torch.Size()

        Returns:
            Tensor: A reparameterized sample from the skew normal distribution.
        """
        shape = self._extended_shape(sample_shape)
        U = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        V = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
    
        # Adjust for skewness using the alpha parameter
        Z = U + self.alpha * torch.abs(V)

        return self.loc + self.scale * Z / torch.sqrt(1 + self.alpha.pow(2))

    def log_prob(self, value):
        """
        Computes the log probability density/mass function (PDF/PMF) of the skew normal distribution.

        Args:
            value (Tensor): The value(s) at which to compute the log probability.

        Returns:
            Tensor: The log probability density/mass function (PDF/PMF) of the distribution evaluated at the given value(s).
        """
        if self._validate_args:
            self._validate_sample(value)
        z = (value - self.loc) * self.scale.reciprocal()
        phi = -0.5 * z.pow(2) - math.log(math.sqrt(2 * math.pi))
        Phi = torch.log(1 + torch.clamp(torch.erf(self.alpha * z / math.sqrt(2)), min=-0.99999))
        log_scale = (
            math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        )
        
        return phi + Phi - log_scale

    def cdf(self, value):
        """
        Computes the cumulative distribution function (CDF) of the skew normal distribution.

        Args:
            value (Tensor): The value(s) at which to compute the CDF.

        Raises:
            NotImplementedError: The CDF is not implemented due to its complexity.
        """
        if self._validate_args:
            self._validate_sample(value)
        raise NotImplementedError("CDF is not implemented due to its complexity.")

    def icdf(self, value):
        """
        Computes the inverse cumulative distribution function (ICDF) of the skew normal distribution.

        Args:
            value (Tensor): The value(s) at which to compute the ICDF.

        Raises:
            NotImplementedError: The ICDF is not implemented due to its complexity.
        """
        raise NotImplementedError("Inverse CDF is not implemented due to its complexity.")

    def entropy(self):
        """
        Computes the entropy of the skew normal distribution.

        Raises:
            NotImplementedError: The entropy is not implemented due to its complexity.
        """
        raise NotImplementedError("Entropy is not implemented due to its complexity.")