from functools import partial
from diffcd.newton import newton_kkt, NewtonState, NewtonConfig
import flax
import jax.numpy as jnp
import jax

def laplacian(f, params, query_point, z):
    '''L(x, mu) = .5 * ||query_point - x||^2 + mu * f(x), z = [x; mu]'''
    x, mu = z[:-1], z[-1]
    return .5 * sq_norm(x - query_point) + mu * f(params, x)

@partial(jax.jit, static_argnames=("f", "newton_config"))
def closest_point_newton(f, params, query_point, init_point, newton_config: NewtonConfig):
    """Compute the point closest to query_point on the surface defined by f(params, x) = 0 using Newton's method."""
    f_val, g = jax.value_and_grad(f, argnums=1)(params, query_point)

    if init_point is None:
        # take initial step which will reach the surface if f is a perfect SDF
        # this corresponds to a Newton step with x0=query_point and mu0=0, but avoids the hessian calculation.
        x0 = query_point - g * f_val / sq_norm(g)
        z0 = jnp.array([*x0, f_val / sq_norm(g)])
    else:
        mu0 = jnp.linalg.norm(query_point - init_point) / jnp.linalg.norm(g) * jnp.sign((query_point - init_point)[0] * g[0])
        z0 = jnp.array([*init_point, mu0])

    z_kkt, newton_state = newton_kkt(
        partial(laplacian, f),
        newton_config,
        jax.lax.stop_gradient(z0),
        params,
        query_point,
    )
    valid_kkt = check_kkt(
        newton_state,
        f(params, query_point)
    )
    valid = jnp.logical_and(valid_kkt, newton_state.converged)
    return z_kkt[:-1], newton_state, valid

@partial(jax.jit, static_argnames=("f", "newton_config"))
def closest_point_newton_batch(f, params, query_points, newton_config: NewtonConfig):
    """Compute closest point for a batch of query points, using the same config for each point."""
    return jax.vmap(
        closest_point_newton,
        in_axes=(None, None, 0, None, None),
    )(f, params, query_points, None, newton_config)

def get_curve_eigvals(g, H):
    """Compute the eigenvalues of the n-by-n matrix H restricted to the subspace orthogonal to n-dim vector g."""
    P = jnp.linalg.svd(jax.lax.stop_gradient(jnp.outer(g, g)))[0][:, 1:]
    return jnp.linalg.eigvalsh(P.T @ H @ P)

def check_kkt(state: NewtonState, f_query: float):
    curve_eigvals = get_curve_eigvals(state.H[-1, :-1], state.H[:-1, :-1])
    is_local_min = curve_eigvals[0] > 0
    valid_sign = jnp.sign(state.z[-1]) == jnp.sign(f_query)
    return jnp.logical_and(is_local_min, valid_sign)

def get_distance_derivative(f, theta, query_point, closest_point):
    """Compute the gradient of x*(theta) = min_x ||x - xq||^2 s.t. f(theta, x) = 0 wrt. theta."""
    H = jax.hessian(f, argnums=1)(theta, closest_point)
    g_x = jax.grad(f, argnums=1)(theta, closest_point)
    mu = (query_point - closest_point)[0] / g_x[0]

    A = jnp.block([
        [jnp.eye(len(query_point)) + mu * H, g_x[:, None]],
        [g_x, 0]
    ])
    A_inv = jnp.linalg.inv(A)[:-1]
    w = - A_inv.T @ (closest_point - query_point)

    return jax.grad(
        lambda theta: 2 * w[-1] * f(theta, closest_point) +
        jax.jvp(partial(f, theta), (closest_point,), (w[:-1],))[1]
    )(theta)

def sq_norm(a, *args, **kwargs):
    return (a ** 2).sum(*args, **kwargs)
