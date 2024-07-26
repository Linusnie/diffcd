import numpy as np
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import jax.random as jrnd
import optax
import tempfile
import trimesh
import os

import diffcd
import fit_implicit
from evaluation import meshing

def make_test_ply(output_file: Path):
    vertices = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5]
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 4, 7],
        [0, 7, 3],
        [1, 5, 6],
        [1, 6, 2],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6]
    ])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    meshing.save_ply(mesh, output_file)

def make_test_npy(output_file: Path):
    points = np.array([
        [-0.5, -0.5, -0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5,  0.5, -0.5],
        [-0.5,  0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5,  0.5],
        [-0.5,  0.5,  0.5]
    ])
    np.save(output_file, points)

class TestTraining(unittest.TestCase):

    def assertFileExists(self, file_path: Path):
        self.assertTrue(file_path.exists(), f'File {file_path} does not exist.')

    def test_fit_implicit_point_cloud(self):
        """Run fit_implicit and check that all outputs are created correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            make_test_npy(tmp_dir / 'cube.npy')
            make_test_ply(tmp_dir / 'cube.ply')

            config = fit_implicit.TrainingConfig(
                output_dir=tmp_dir / 'outputs',
                model=diffcd.networks.MLP(10, 3),
                checkpoint_options=fit_implicit.CustomCheckpointManagerOptions(save_interval_steps=2),
                method=diffcd.methods.NeuralPull(
                    sampling=diffcd.samplers.SamplingConfig(k=3),
                ),
                final_mesh_points_per_axis=32,
                batch_size=12,
                n_batches=4,
                with_timestamp=False,
                experiment_name='test_experiment',
                dataset=diffcd.datasets.PointCloud(path=tmp_dir / 'cube.npy'),
                gt_mesh=diffcd.datasets.EvaluationMesh(path=tmp_dir / 'cube.ply')
            )
            fit_implicit.run(config)
            experiment_dir = tmp_dir / 'outputs' / 'test_experiment'
            self.assertFileExists(experiment_dir)

            file_names = [
                'config.pickle',
                'config.yaml',
                'eval_metrics.csv',
                'eval_metrics_final_4.csv',
                'local_sigma.npy',
                'sample_points.npy',
                'target_points.npy',
                'mesh_final_4.ply',
                'train_metrics.csv',
                'train_points.npy',
            ]
            for file_name in file_names:
                self.assertFileExists(experiment_dir / file_name)

            # check that there are no extra files
            self.assertEqual(len(next(os.walk(experiment_dir))[2]), len(file_names))

            for checkpoint_index in [0, 2, 4]:
                self.assertFileExists(experiment_dir / f'checkpoints/{checkpoint_index}')
                self.assertFileExists(experiment_dir / f'meshes/mesh_{checkpoint_index}.ply')

    def test_step(self):
        model = diffcd.networks.MLP(
            layer_size=10,
            n_layers=4,
            skip_layers=(2,)
        )
        key = jrnd.PRNGKey(0)
        params = model.init(key, jnp.zeros(3) * 1.)

        self.assertEqual(len(params['params']), model.n_layers + 1)
        for i in range(model.n_layers):
            in_dim = 3 if i == 0 else model.layer_size
            out_dim = model.layer_size if i + 1 not in model.skip_layers else model.layer_size - 3
            self.assertEqual(params['params'][f'dense_{i}']['kernel'].shape, (in_dim, out_dim))
            self.assertEqual(params['params'][f'dense_{i}']['bias'].shape, (out_dim,))
        self.assertEqual(params['params'][f'dense_{model.n_layers}']['kernel'].shape, (model.layer_size, 1))
        self.assertEqual(params['params'][f'dense_{model.n_layers}']['bias'].shape, (1,))

        for method_class in [
            diffcd.methods.IGR,
            diffcd.methods.NeuralPull,
            lambda **kwargs: diffcd.methods.DiffCD(**kwargs, p2s_loss='closest-point'),
            lambda **kwargs: diffcd.methods.DiffCD(**kwargs, p2s_loss='implicit'),
        ]:
            with self.subTest(method_class.__name__):
                method = method_class(sampling=diffcd.samplers.SamplingConfig(k=3))
                train_state = diffcd.training.ShapeTrainState.create(
                    apply_fn=jax.jit(model.apply),
                    tx=optax.adam(learning_rate=1e-3,),
                    params=params,
                    lower_bound=(-1.8, -1.8, -1.8),
                    upper_bound=(1.8, 1.8, 1.8),
                )
                train_points = jnp.ones((10, 3))
                method_state = method.init_state(key, train_points, None)
                _, *batch = method.get_batch(train_state, method_state, key, 5)
                metrics, train_state, nan_grads = method.step(
                    train_state, *batch
                )
                self.assertFalse(jnp.any(nan_grads))

if __name__ == '__main__':
    unittest.main()
