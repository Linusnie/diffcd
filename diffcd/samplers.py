from scipy.spatial import cKDTree
from functools import partial
import numpy as np
import jax
from typing import Optional
import jax.random as jrnd
import jax.numpy as jnp
import flax
import trimesh
from dataclasses import field

from evaluation import meshing
from diffcd.newton import NewtonConfig
from diffcd.closest_point import closest_point_newton_batch


@flax.struct.dataclass
class SamplingConfig:
    # Number of points to sample around each query point, standard deviation determined by local sigma
    samples_per_point: int = 1

    # Number of neighbours to use for local sigma calculation
    k: int = 50

    # Number of global samples
    global_samples: Optional[int] = None

    # scale to apply to local sigma
    local_sigma_scale: float = .2

@flax.struct.dataclass
class SurfaceSamplingConfig:
    # number of mesh samples to generate each mesh interval
    num_samples: int = 30000

    # number of iterations between each recomputation of the surface mesh, which is used for initializing samples
    mesh_interval: int = 1000

    # config for closest point calculation, only max_iters is used
    newton: NewtonConfig = field(default_factory=lambda: NewtonConfig())

    # config for mesh computation
    surface_meshing: meshing.Meshing = field(default_factory=lambda: meshing.Meshing())

@partial(jax.jit, static_argnames=("n_points", "n_dims"))
def generate_global_samples(key, lower, upper, n_points, n_dims):
    return jrnd.uniform(
        key,
        shape=(n_points, n_dims),
        minval=jnp.array(lower),
        maxval=jnp.array(upper),
    )

@partial(jax.jit, static_argnames="samples_per_point")
def generate_local_samples(key, query_points, samples_per_point, local_sigma):
    num_points, dims = query_points.shape
    noise = jrnd.normal(key, (num_points, samples_per_point, dims))
    query_samples = query_points[:, None, :] +  noise * local_sigma[:, None, None]
    return jnp.reshape(query_samples, (-1, dims))

@partial(jax.jit, static_argnames='n_samples')
def sample_array(key, array, n_samples):
    sample_indices = jrnd.choice(
        key, len(array),
        (min(n_samples, len(array)),),
        replace=False,
    )
    return array[sample_indices], sample_indices

def compute_local_sigma(points, k):
    if k >= len(points):
        raise ValueError(f"Cannot find {k=} neighbours with {points.shape=}")

    if k == 0:
        return np.zeros(len(points))
    sigmas = []
    ptree = cKDTree(points)

    for points_batch in np.array_split(points, 100, axis=0):
        distances = ptree.query(points_batch, k + 1)
        sigmas.append(distances[0][:, -1])
    return np.concatenate(sigmas)

@flax.struct.dataclass
class DescentState:
    i: int
    x: jnp.array

def step(apply_fn, params, state):
    f, g = jax.value_and_grad(apply_fn, argnums=1)(params, state.x)
    return DescentState(state.i+1, state.x - f * g / jnp.linalg.norm(g))

@partial(jax.jit, static_argnames=["apply_fn", "n_steps"])
def sdf_descent(apply_fn, params, query_point, n_steps):
    """Compute surface point by iterating x_{k+1} = x_k - f(params, x_k)g(params, x_k)/||g(params, x_k)||"""
    state = jax.lax.while_loop(
        lambda state: state.i < n_steps,
        partial(step, apply_fn, params),
        DescentState(0, query_point),
    )
    f = apply_fn(params, state.x)
    return state.x, jnp.abs(f) < 1e-3

@partial(jax.jit, static_argnames=["f", "newton_config"])
def generate_surface_samples(f, params, mesh_samples, newton_config: NewtonConfig):
    """Generate samples from implicit surface defined by f(params, x) = 0 by sampling from approxmiating mesh and computing a neary surface point."""
    surface_points, valid = jax.vmap(sdf_descent, in_axes=(None, None, 0, None))(
        f, params, mesh_samples, newton_config.max_iters,
    )
    return surface_points, valid
