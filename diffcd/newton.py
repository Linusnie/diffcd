from functools import partial
import jax
import jax.numpy as jnp
import flax
from typing import Optional

@flax.struct.dataclass
class NewtonState:
    # variables
    z: jnp.array

    # variables from previous steps (including current step)
    z_steps: jnp.array

    # current step inex
    step: int

    # gradient of laplacian
    g: jnp.array

    # hessian of laplacian
    H: jnp.array

    # True if algorithm has converged
    converged: bool

@flax.struct.dataclass
class NewtonConfig:
    # maximum number of iterations
    max_iters: int = 4

    # converged when norm(grad(laplacian)(z)) <= eps
    grad_norm_eps: float = 1e-3

    # weather to stop when convergence critera is reached
    stop_when_converged: bool = False

def newton_step(laplacian, config: NewtonConfig, args, state: NewtonState):
    z_next = state.z - jnp.linalg.lstsq(state.H, state.g)[0]
    g_next=jax.grad(laplacian, argnums=-1)(*args, z_next)
    H_next=jax.hessian(laplacian, argnums=-1)(*args, z_next)
    return NewtonState(
        z=z_next,
        z_steps=state.z_steps.at[state.step + 1].set(z_next),
        step=state.step + 1,
        g=g_next,
        H=H_next,
        converged=jnp.linalg.norm(g_next) < config.grad_norm_eps,
    )

def should_continue(config: NewtonConfig, state: NewtonState):
    return jnp.logical_and(
        state.step < config.max_iters,
        jnp.logical_not(jnp.logical_and(state.converged, config.stop_when_converged)),
    )

def _newton_kkt(laplacian, config: NewtonConfig, z0, *args):
    """Find kkt point z* where dL(params, z*)/dz = 0 using Newton's method."""
    g0 = jax.grad(laplacian, argnums=-1)(*args, z0)
    H0 = jax.hessian(laplacian, argnums=-1)(*args, z0)
    init_state = NewtonState(
        z=z0,
        z_steps=jnp.repeat(jnp.zeros_like(z0)[None], config.max_iters+1, axis=0).at[0].set(z0),
        step=0,
        g=g0,
        H=H0,
        converged=jnp.linalg.norm(g0) < config.grad_norm_eps,
    )
    final_state = jax.lax.while_loop(
        partial(should_continue, config),
        partial(newton_step, laplacian, config, args),
        init_state,
    )
    return final_state.z, final_state

def newton_kkt_fwd(laplacian, config: NewtonConfig, z0, *args):
    z, final_state = _newton_kkt(laplacian, config, z0, *args)
    return (z, final_state), (final_state, args)

def newton_kkt_bwd(laplacian, config: NewtonConfig, info, tangent):
    final_state, args = info
    Hinvt = jnp.linalg.lstsq(final_state.H, tangent[0])[0]

    # -tangent @ Hinv @ d^2L/d(theta)dz
    jvp = jax.grad(
        lambda args: jax.jvp(partial(laplacian, *args), (final_state.z,), (-Hinvt,))[1]
    )(args)

    # gradient wrt z0 is 0
    return (0., *jvp,)

newton_kkt = jax.custom_vjp(_newton_kkt, nondiff_argnums=(0, 1))
newton_kkt.defvjp(newton_kkt_fwd, newton_kkt_bwd)
