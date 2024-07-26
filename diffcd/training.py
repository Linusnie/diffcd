from typing import Callable
from flax.training.train_state import TrainState
import flax
import jax.numpy as jnp

class ShapeTrainState(TrainState):
    # train sate that also includes the upper/lower bound for function inputs
    lower_bound: jnp.array = flax.struct.field(pytree_node=False)
    upper_bound: jnp.array = flax.struct.field(pytree_node=False)
