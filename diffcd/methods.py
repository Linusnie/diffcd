import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax import jit, vmap
from dataclasses import field
from functools import partial
from typing import Union, Literal, Optional
import flax
from scipy.spatial import cKDTree
import numpy as np
import trimesh

from evaluation import meshing
from diffcd.closest_point import sq_norm, closest_point_newton, NewtonConfig
from diffcd import samplers
from diffcd.samplers import SamplingConfig, SurfaceSamplingConfig
from diffcd.training import ShapeTrainState


def any_nans(pytree):
    """Returns True if any leaf of the pytree contains a nan value."""
    return jnp.array(jax.tree_util.tree_flatten(jax.tree_map(lambda a: jnp.isnan(a).any(), pytree))[0]).any()

def valid_mean(array, valid_mask):
    return (array * valid_mask).sum() / valid_mask.sum()

def safe_apply_grads(state, grads):
    nan_grads = any_nans(grads)
    state = jax.lax.cond(nan_grads, lambda: state, lambda: state.apply_gradients(grads=grads))
    return state, nan_grads

def soft_norm(x, eps=1e-12):
    """Use l2 for large values and squared l2 for small values to avoid grad=nan at x=0."""
    return eps * (jnp.sqrt(sq_norm(x) / eps ** 2 + 1) - 1)

def get_eikonal_loss(f, params, point):
    point_grad = jax.grad(f, argnums=1)(params, point)
    return (1 - soft_norm(point_grad)) ** 2

def safe_normalize(x, eps=1e-12):
    x_norm = soft_norm(x)
    return x / jnp.array([x_norm, eps]).max()

def _surface_point(f, params, point):
    """Makes point differentiable as a function of params, assuming f(params, x) = 0 and gradient x wrt any parameter is restricted to the normal of the surface."""
    return point

def grad_norm(f, inputs):
    return jnp.linalg.norm(jax.grad(f)(inputs))

def surface_point_fwd(f, params, point):
    return point, (point, params)

def surface_point_bwd(f, info, tangent):
    point, params = info
    g_x = jax.grad(f, argnums=1)(params, point)
    tg = tangent @ g_x / sq_norm(g_x)
    g_params = jax.grad(f)(params, point)
    return jax.tree_map(lambda g_param: -g_param * tg, g_params), tangent

surface_point = jax.custom_vjp(_surface_point, nondiff_argnums=(0,))
surface_point.defvjp(surface_point_fwd, surface_point_bwd)

@flax.struct.dataclass
class SurfaceSamples:
    points: jnp.array
    closest_train_points: jnp.array
    valid: jnp.array

def implicit_distance(f, points, distance_metric):
    f_values = f(points)
    if distance_metric == 'squared_l2':
        implicit_distance = (f_values ** 2).mean()
    elif distance_metric == 'l2':
        implicit_distance = jnp.abs(f_values).mean()
    else:
        raise ValueError(f'Unrecognized distance metric {distance_metric=}.')
    return implicit_distance

def distance_loss(point1, point2, distance_metric):
    match distance_metric:
        case 'l2':
            loss = soft_norm(point1 - point2, eps=1e-12)
        case 'squared_l2':
            loss = sq_norm(point1 - point2)
        case _:
            raise ValueError(f'Unrecognized distance metric {distance_metric=}')
    return loss

@flax.struct.dataclass
class DiffCDState:
    point_cloud: jnp.array
    local_sigma: jnp.array
    point_cloud_kd_tree: cKDTree = flax.struct.field(pytree_node=False)
    mesh_samples: jnp.array

@flax.struct.dataclass
class DiffCD:
    newton_config: NewtonConfig = field(default_factory=lambda: NewtonConfig())
    p2s_loss: Literal['closest-point', 'implicit'] = 'implicit'

    eikonal_weight: float = 0.1
    s2p_weight: float = 1.

    surface_area_weight: float = 0.
    alpha: float = 100
    surface_area_samples: int = 5000

    distance_metric: Literal['l2', 'squared_l2'] = 'l2'
    sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig())
    surface_sampling: SurfaceSamplingConfig = field(default_factory=lambda: SurfaceSamplingConfig())

    def init_state(self, key, point_cloud, save_dir=None):
        local_sigma = samplers.compute_local_sigma(point_cloud, self.sampling.k) * self.sampling.local_sigma_scale
        if save_dir is not None:
            np.save(save_dir / 'local_sigma.npy', local_sigma)
        return DiffCDState(point_cloud, local_sigma, cKDTree(point_cloud), None)

    @partial(jit, static_argnames=("self", "apply_fn", "batch_size"))
    def get_surface_samples(self, key, apply_fn, params, mesh_samples, point_cloud, batch_size):
        # compute batch of surface samples and their closest point cloud points
        batch_mesh_samples, _ = samplers.sample_array(key, mesh_samples, batch_size)
        surface_samples, valid = samplers.generate_surface_samples(
            apply_fn, params,
            batch_mesh_samples,
            self.surface_sampling.newton,
        )
        closest_train_points = vmap(point_cloud_closest_point, in_axes=(0, None))(surface_samples, point_cloud)
        return SurfaceSamples(surface_samples, closest_train_points, valid)

    def get_batch(self, train_state: ShapeTrainState, state: DiffCDState, key, batch_size):
        _, *batch = IGR.get_batch(self, train_state, state, key, batch_size)
        if self.s2p_weight != 0:
            if (train_state.step % self.surface_sampling.mesh_interval == 0):
                # update surface sampling mesh
                sampling_mesh = meshing.extract_mesh(
                    self.surface_sampling.surface_meshing,
                    f=partial(train_state.apply_fn, train_state.params),
                )
                mesh_samples = trimesh.sample.sample_surface(sampling_mesh, self.surface_sampling.num_samples)[0]
                state = DiffCDState(state.point_cloud, state.local_sigma, state.point_cloud_kd_tree, mesh_samples)

            surface_samples = self.get_surface_samples(
                key, train_state.apply_fn, train_state.params, state.mesh_samples, state.point_cloud, batch_size
            )
        else:
            surface_samples = SurfaceSamples(jnp.array([]), jnp.array([]), jnp.array([]))
        return state, *batch, surface_samples

    def closest_point_loss(self, f, params, query_point):
        closest_point, newton_state, valid = closest_point_newton(f, params, query_point, None, self.newton_config)
        loss = distance_loss(closest_point, query_point, self.distance_metric)

        return loss, newton_state, valid

    def surface_sample_loss(self, f, params, surface_sample_point, point_cloud_point):
        surface_sample_point = surface_point(f, params, surface_sample_point)
        loss = distance_loss(surface_sample_point, point_cloud_point, self.distance_metric)
        return loss

    def batch_loss(self, apply_fn, params, query_points, sample_points, uniform_samples, surface_samples: SurfaceSamples):
        metrics = {}
        if self.p2s_loss == 'closest-point':
            p2s_losses, newton_state, valid = jax.vmap(self.closest_point_loss, in_axes=(None, None, 0))(apply_fn, params, query_points)
            p2s_loss = valid_mean(p2s_losses, valid)
            metrics = {
                'mean_n_valid': valid.mean(),
                'mean_n_converged': newton_state.converged.mean(),
                'mean_n_newton_steps': newton_state.step.mean(),
            }
        elif self.p2s_loss == 'implicit':
            p2s_loss = implicit_distance(partial(apply_fn, params), query_points, self.distance_metric)
        else:
            raise ValueError(f'Unrecognized p2s metric {self.p2s_metric=}.')

        if self.s2p_weight != 0.:
            s2p_losses = vmap(
                self.surface_sample_loss, in_axes=(None, None, 0, 0)
            )(apply_fn, params, surface_samples.points, surface_samples.closest_train_points)
            s2p_loss = valid_mean(s2p_losses, surface_samples.valid)
        else:
            s2p_loss = 0.

        if self.eikonal_weight != 0.:
            eikonal_loss = vmap(get_eikonal_loss, in_axes=(None, None, 0))(
                apply_fn, params, jnp.vstack([query_points, sample_points])
            ).mean()
        else:
            eikonal_loss = 0.

        if self.surface_area_weight != 0.:
            surface_area_loss = get_implicit_surface_area_loss(partial(apply_fn, params), uniform_samples, self.alpha).mean()
        else:
            surface_area_loss = 0.

        loss = p2s_loss
        loss += self.s2p_weight * s2p_loss
        loss /= 1. + self.s2p_weight
        loss += self.eikonal_weight * eikonal_loss
        loss += self.surface_area_weight * surface_area_loss

        metrics = {
            'loss': loss,
            'points_to_surface_loss': p2s_loss,
            'surface_to_points_loss': s2p_loss,
            'eikonal_loss': eikonal_loss,
            'implicit_surface_area_loss': surface_area_loss,
            'mean_n_valid_surface_sample': surface_samples.valid.mean(),
            **metrics,
        }
        return loss, metrics

    @partial(jit, static_argnames="self")
    def step(self, state: ShapeTrainState, query_points, sample_points, uniform_samples, surface_points):
        grads, metrics = jax.grad(
            self.batch_loss, argnums=1, has_aux=True
        )(state.apply_fn, state.params, query_points, sample_points, uniform_samples, surface_points)
        state, nan_grads = safe_apply_grads(state, grads)
        return metrics, state, nan_grads

def get_implicit_surface_area_loss(f, points, alpha):
    return jnp.exp(-alpha * jnp.abs(f(points)))

@flax.struct.dataclass
class IGRState:
    point_cloud: jnp.array
    local_sigma: jnp.array

@flax.struct.dataclass
class IGR:
    eikonal_weight: float = 0.1
    distance_metric: Literal['l2', 'squared_l2'] = 'l2'
    sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig())
    surface_area_weight: float = 0.
    alpha: float = 100
    surface_area_samples: int = 5000

    def init_state(self, key, point_cloud, save_dir=None):
        local_sigma = samplers.compute_local_sigma(point_cloud, self.sampling.k) * self.sampling.local_sigma_scale
        if save_dir is not None:
            np.save(save_dir / 'local_sigma.npy', local_sigma)
        return IGRState(point_cloud, local_sigma)

    def get_loss(self, apply_fn, params, query_points, sample_points, uniform_sample_points):
        implicit_distance_loss = implicit_distance(partial(apply_fn, params), query_points, self.distance_metric)
        eikonal_loss = vmap(get_eikonal_loss, in_axes=(None, None, 0))(apply_fn, params, sample_points).mean()
        surface_area_loss = get_implicit_surface_area_loss(partial(apply_fn, params), uniform_sample_points, self.alpha).mean()
        loss = implicit_distance_loss + self.eikonal_weight * eikonal_loss + self.surface_area_weight * surface_area_loss
        return loss, (implicit_distance_loss, eikonal_loss, surface_area_loss)

    @partial(jit, static_argnames=("self", "batch_size"))
    def get_batch(self, train_state: ShapeTrainState, state: IGRState, key, batch_size):
        point_batch, batch_indices = samplers.sample_array(key, state.point_cloud, batch_size)

        key_local, key_global, key_area = jrnd.split(key, 3)
        local_samples = samplers.generate_local_samples(
            key_local,
            point_batch,
            self.sampling.samples_per_point,
            state.local_sigma[batch_indices],
        )
        global_samples = samplers.generate_global_samples(
            key_global,
            lower=train_state.lower_bound,
            upper=train_state.upper_bound,
            n_points=self.sampling.global_samples if self.sampling.global_samples is not None else len(local_samples) // 8,
            n_dims=state.point_cloud.shape[-1],
        )
        uniform_samples = samplers.generate_global_samples(
            key_area,
            lower=train_state.lower_bound,
            upper=train_state.upper_bound,
            n_points=self.surface_area_samples if self.surface_area_weight != 0 else 0,
            n_dims=state.point_cloud.shape[-1],
        )
        return state, point_batch, jnp.concatenate([local_samples, global_samples]), uniform_samples

    @partial(jit, static_argnames="self")
    def step(self, state: ShapeTrainState, query_points, sample_points, uniform_sample_points):
        (loss, (distance_loss, eikonal_loss, surface_area_loss)), grads = jax.value_and_grad(
            self.get_loss, argnums=1, has_aux=True
        )(state.apply_fn, state.params, query_points, sample_points, uniform_sample_points)
        state, nan_grads = safe_apply_grads(state, grads)
        metrics = {
            'loss': loss,
            f'implicit_{self.distance_metric}_loss': distance_loss,
            'eikonal_loss': eikonal_loss,
            'implicit_surface_area_loss': surface_area_loss,
        }
        return metrics, state, nan_grads

def pull_point(f, point):
    f_value, f_grad = jax.value_and_grad(f)(point)
    return point - f_value * safe_normalize(f_grad)

def point_cloud_closest_point(query_point, point_cloud):
    distances = sq_norm(point_cloud - query_point, axis=-1)
    return point_cloud[jnp.argmin(distances)]

def get_neural_pull_loss(apply_fn, params, target_point, sample_point, distance_metric):
    pulled_point = pull_point(partial(apply_fn, params), sample_point)
    return distance_loss(target_point, pulled_point, distance_metric)

@flax.struct.dataclass
class NeuralPullState:
    sample_points: jnp.array
    target_points: jnp.array

@flax.struct.dataclass
class NeuralPull:
    eikonal_weight: float = 0.
    distance_metric: Literal['l1', 'l2', 'squared_l2'] = 'l2'
    sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig(samples_per_point=10))

    def init_state(self, key, point_cloud, save_dir=None):
        local_sigma = samplers.compute_local_sigma(point_cloud, self.sampling.k) * self.sampling.local_sigma_scale
        sample_points = samplers.generate_local_samples(
            key,
            point_cloud,
            self.sampling.samples_per_point, # specify total samples instead?
            local_sigma,
        )
        target_indices = cKDTree(point_cloud).query(sample_points)[1]
        target_points = np.array(point_cloud[target_indices])
        if save_dir is not None:
            np.save(save_dir / 'local_sigma.npy', local_sigma)
            np.save(save_dir / 'sample_points.npy', sample_points)
            np.save(save_dir / 'target_points.npy', target_points)

        return NeuralPullState(sample_points, target_points)

    @partial(jit, static_argnames=("self", "batch_size"))
    def get_batch(self, train_state: ShapeTrainState, state: NeuralPullState, key, batch_size, *args):
        batch_indices = jrnd.choice(
            key, len(state.sample_points),
            (min(batch_size, len(state.sample_points)),),
            replace=False
        )
        return state, state.sample_points[batch_indices], state.target_points[batch_indices]

    def batch_loss(self, apply_fn, params, sample_points, target_points):
        neural_pull_loss = vmap(get_neural_pull_loss, in_axes=(None, None, 0, 0, None))(
            apply_fn, params, target_points, sample_points, self.distance_metric
        ).mean()

        if self.eikonal_weight != 0:
            eikonal_loss = vmap(get_eikonal_loss, in_axes=(None, None, 0))(apply_fn, params, sample_points).mean()
        else:
            eikonal_loss = 0.

        return neural_pull_loss + self.eikonal_weight * eikonal_loss, (neural_pull_loss, eikonal_loss)

    @partial(jit, static_argnames="self")
    def step(self, train_state: ShapeTrainState, sample_points, target_points):
        (loss, (distance_loss, eikonal_loss)), grads = jax.value_and_grad(self.batch_loss, argnums=1, has_aux=True)(
            train_state.apply_fn, train_state.params, sample_points, target_points
        )
        train_state, nan_grads = safe_apply_grads(train_state, grads)
        metrics = {
            'loss': loss,
            f'distance_loss': distance_loss,
            'eikonal_loss': eikonal_loss,
        }
        return metrics, train_state, nan_grads

Methods = Union[
    DiffCD,
    IGR,
    NeuralPull,
]
