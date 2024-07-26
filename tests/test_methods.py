import numpy as np
import jax
import jax.numpy as jnp
import unittest

import diffcd

def first(f):
    return lambda *args, **kwargs: f(*args, **kwargs)[0]

class TestMethods(unittest.TestCase):

    def test_point_cloud_closest_point(self):
        point_cloud = jnp.array([
            [0., 0., 0.],
            [1., 1., 1.],
            [5., -1., 3.],
        ])
        for point in point_cloud:
            closest_point = diffcd.methods.point_cloud_closest_point(
                point + jnp.array([.1, .1, .1]), point_cloud
            )
            np.testing.assert_array_equal(point, closest_point)

    def test_eikonal_loss_exact_sdf(self):
        f = lambda r, x: r - jnp.linalg.norm(x)
        eikonal_loss = diffcd.methods.get_eikonal_loss(f, 2., jnp.array([1., 2., 3.]))
        self.assertAlmostEqual(eikonal_loss, 0.)

    def test_eikonal_loss_approx_sdf(self):
        f = lambda r, x: r ** 2 - diffcd.closest_point.sq_norm(x)
        x = jnp.array([1., 2., 3.])
        eikonal_loss = diffcd.methods.get_eikonal_loss(f, 2., x)
        self.assertAlmostEqual(eikonal_loss, (1 - jnp.linalg.norm(2 * x)) ** 2)

        x = jnp.array([0., 0., 0.])
        eikonal_loss = diffcd.methods.get_eikonal_loss(f, 2., x)
        self.assertAlmostEqual(eikonal_loss, (1 - jnp.linalg.norm(2 * x)) ** 2)

    def test_igr_loss(self):
        f = lambda r, x: r - jnp.linalg.norm(x)
        x, r = jnp.array([1., 2., 3.]), 2.

        # l2
        igr_l2 = diffcd.methods.IGR(eikonal_weight=0., distance_metric='l2')
        igr_loss, _ = igr_l2.get_loss(f, r, x, x, x)
        self.assertAlmostEqual(igr_loss, jnp.linalg.norm(x - r * x / jnp.linalg.norm(x)), places=6)
        grad, _ = jax.grad(igr_l2.get_loss, argnums=1, has_aux=True)(f, r, x, x, x)
        self.assertAlmostEqual(grad, -1.)

        # squared l2
        igr_sql2 = diffcd.methods.IGR(eikonal_weight=0., distance_metric='squared_l2')
        igr_loss, _ = igr_sql2.get_loss(f, r, x, x, x)
        self.assertAlmostEqual(igr_loss, jnp.linalg.norm(x - r * x / jnp.linalg.norm(x)) ** 2, places=6)
        grad, _ = jax.grad(igr_sql2.get_loss, argnums=1, has_aux=True)(f, r, x, x, x)
        self.assertAlmostEqual(grad, 2 * (r - jnp.linalg.norm(x)))

        # l2 with f(x) = 0
        f = lambda r, x: r ** 2 - diffcd.closest_point.sq_norm(x)
        x = jnp.array([0., 0., 0.])
        (igr_loss, _), grad = jax.value_and_grad(igr_l2.get_loss, argnums=1, has_aux=True)(f, r, x, x, x)
        self.assertAlmostEqual(igr_loss, r ** 2)
        self.assertAlmostEqual(grad, 2 * r)

    def test_pull_point(self):
        """Pulled point should equal closest point for SDF."""
        radius = 2
        f = lambda x: radius - jnp.linalg.norm(x)
        x = jnp.array([1., 2., 3.])

        pulled_point = diffcd.methods.pull_point(f, x)
        np.testing.assert_array_almost_equal(
            pulled_point, x / jnp.linalg.norm(x) * radius
        )

    def test_pull_point_zero_grad(self):
        """Check that pull_points handles points with gradient=0."""
        radius = 2
        f = lambda x: radius ** 2 - diffcd.closest_point.sq_norm(x)
        x = jnp.array([0., 0., 0.])

        pulled_point = diffcd.methods.pull_point(f, x)
        np.testing.assert_array_almost_equal(pulled_point, x)

        loss, grad = jax.value_and_grad(lambda x: ((diffcd.methods.pull_point(f, x) - jnp.ones(3)) ** 2).mean())(x)
        np.testing.assert_almost_equal(loss, 1.)
        self.assertFalse(jnp.isnan(grad).any())

    def test_closest_point(self):
        f = lambda radius, x: radius ** 2 - diffcd.closest_point.sq_norm(x)
        x = jnp.array([1., 2., 3.])

        radius, eps, max_iters = 2, 1e-6, 10

        # without stop when converged
        closest_point, newton_state, valid = diffcd.closest_point.closest_point_newton(
            f, radius, x, x, diffcd.closest_point.NewtonConfig(grad_norm_eps=eps, max_iters=max_iters, stop_when_converged=False)
        )
        self.assertTrue(valid)
        np.testing.assert_array_almost_equal(closest_point, x / jnp.linalg.norm(x) * radius)
        self.assertTrue(newton_state.converged)
        self.assertEqual(newton_state.step, max_iters)
        laplacian_grad = jax.grad(diffcd.closest_point.laplacian, argnums=-1)(f, radius, x, newton_state.z_steps[-1])
        self.assertLess(jnp.linalg.norm(laplacian_grad), eps)

        # with stop when converged
        closest_point, newton_state, valid = diffcd.closest_point.closest_point_newton(
            f, radius, x, x, diffcd.closest_point.NewtonConfig(grad_norm_eps=eps, max_iters=max_iters, stop_when_converged=True)
        )
        self.assertTrue(valid)
        np.testing.assert_array_almost_equal(closest_point, x / jnp.linalg.norm(x) * radius)
        self.assertTrue(newton_state.converged)
        self.assertLess(newton_state.step, max_iters)
        laplacian_grad = jax.grad(diffcd.closest_point.laplacian, argnums=-1)(f, radius, x, newton_state.z_steps[newton_state.step])
        self.assertLess(jnp.linalg.norm(laplacian_grad), eps)


    def test_closest_point_grad(self):
        f = lambda radius, x: radius ** 2 - diffcd.closest_point.sq_norm(x)
        x = jnp.array([1., 2., 3.])

        radius = 2.
        closest_point_grad = jax.jacrev(first(diffcd.closest_point.closest_point_newton), argnums=1)(
            f, radius, x, x, diffcd.closest_point.NewtonConfig()
        )
        np.testing.assert_array_almost_equal(
            closest_point_grad, x / jnp.linalg.norm(x)
        )

if __name__ == '__main__':
    unittest.main()
