import enum
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import relu, elu, linear
from typing import Tuple
import jax.random as jrnd
import jax


class ActivationFunction(enum.Enum):
    RELU = enum.auto()
    ELU = enum.auto()
    SIN = enum.auto()
    SOFTPLUS = enum.auto()


def get_activation_function(activation_function: ActivationFunction):
    return {
        ActivationFunction.RELU: relu,
        ActivationFunction.ELU: elu,
        ActivationFunction.SIN: jnp.sin,
        ActivationFunction.SOFTPLUS: safe_softplus,
    }[activation_function]


class MLP(nn.Module):
    layer_size: int = 256
    n_layers: int = 8
    skip_layers: Tuple[int, ...] = (4,)
    activation_function: ActivationFunction = ActivationFunction.SOFTPLUS
    geometry_init: bool = True
    init_radius: float = 0.5

    @nn.compact
    def __call__(self, x):
        input_x = x
        dim_x = x.shape[-1]

        actication_function = get_activation_function(self.activation_function)
        kernel_init = zero_mean if self.geometry_init else linear.default_kernel_init
        for i in range(self.n_layers):
            if i in self.skip_layers:
                x = jnp.concatenate([x, input_x], axis=-1) / jnp.sqrt(2)
            layer_size = self.layer_size if i + 1 not in self.skip_layers else self.layer_size - dim_x
            x = nn.Dense(features=layer_size, name=f'dense_{i}', kernel_init=kernel_init)(x)
            x = actication_function(x)
        kernel_init_final = non_zero_mean if self.geometry_init else linear.default_kernel_init
        bias_init_final = jax.nn.initializers.constant(-self.init_radius if self.geometry_init else 0.)
        x = nn.Dense(features=1, name=f'dense_{self.n_layers}', kernel_init=kernel_init_final, bias_init=bias_init_final)(x)
        return x.squeeze()

def non_zero_mean(key, shape, dtype=jnp.float32):
    normal_random_values = jrnd.normal(key, shape, dtype=dtype)
    mu = jnp.sqrt(jnp.pi) / jnp.sqrt(shape[0])
    return  mu + 0.00001 * normal_random_values

def zero_mean(key, shape, dtype=jnp.float32):
    normal_random_values = jrnd.normal(key, shape, dtype=dtype)
    sigma = jnp.sqrt(2) / jnp.sqrt(shape[1])
    return sigma * normal_random_values

def softplus(x, beta=100):
    return jnp.logaddexp(0, beta * x) / beta

def safe_softplus(x, beta=100):
    # revert to linear function for large inputs, same as pytorch
    return jnp.where(x * beta > 20, x, softplus(x))
