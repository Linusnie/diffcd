import trimesh
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Union, Tuple, Optional
import jax.numpy as jnp
import jax.random as jrnd

@dataclass
class Circle:
    n_points: int = 50
    n_points_eval: int = 250
    radius: float = 2.
    sigma: float = 0.5
    lower_bound:Tuple[float, float] = (-3., -3.)
    upper_bound:Tuple[float, float] = (3., 3.)
    n_dims: Literal[2] = 2

    def get_data(self, key):
        alphas = jnp.linspace(0, 2 * jnp.pi, self.n_points + 1)[:-1]
        points = self.radius * jnp.array([jnp.cos(alphas), jnp.sin(alphas)]).T
        points = points + jrnd.normal(key, points.shape) * self.sigma
        return points

    def get_data_eval(self, key):
        alphas = jnp.linspace(0, 2 * jnp.pi, self.n_points_eval + 1)[:-1]
        points = self.radius * jnp.array([jnp.cos(alphas), jnp.sin(alphas)]).T
        return points

    def get_train_eval_points(self, key):
        key_train, key_eval = jrnd.split(key, 2)
        train_points = self.get_data(key_train)
        eval_points = self.get_data_eval(key_eval)
        return train_points, eval_points


@dataclass
class PointCloud:
    # Path to file containing point cloud. Either .npy with xyz coordinates, or .ply file with mesh to sample points from
    path: Path

    # Number of training points, defaults to all points for .npy files or n_vertices for .ply files
    n_points: Optional[int] = None

    # Standard deviation of gaussian noise to add to each point
    sigma: float = 0.

    # If true, subtract center of bounding box from point cloud and then divide my maximum side length
    auto_scale: bool = True

    n_dims: Literal[3] = 3
    _scale_factor = None
    _center_point = None

    def apply_normalization(self, points):
        if len(points) > 0:
            return (points - self._center_point) / self._scale_factor
        else:
            return points

    def undo_normalization(self, points):
        if len(points) > 0:
            return points * self._scale_factor + self._center_point
        else:
            return points

    def get_normalized_points(self, key):
        extensions = ['.npy', '.xyz']
        if self.path.suffix in extensions:
            point_cloud = jnp.load(self.path)

            n_points = self.n_points if self.n_points is not None else len(point_cloud)
            point_cloud = jrnd.choice(key, point_cloud, (n_points,), replace=False)

            if self.auto_scale:
                lower, upper = point_cloud.min(axis=0), point_cloud.max(axis=0)
                self._center_point = (lower + upper) / 2
                self._scale_factor = (upper - lower).max()
            else:
                self._center_point = jnp.zeros(3, dtype=jnp.float32)
                self._scale_factor = 1.

            point_cloud = self.apply_normalization(point_cloud)
        else:
            raise ValueError(f'File extension {self.path.suffix} not recognized for file {self.path}. Expected {extensions}')

        point_cloud += jrnd.normal(key, point_cloud.shape) * self.sigma
        return point_cloud


@dataclass
class EvaluationMesh:
    path: Optional[Path] = None
    n_samples: int = 30000

    _mesh = None

    @property
    def mesh(self):
        if (self._mesh is None) and (self.path is not None):
            self._mesh = trimesh.load_mesh(self.path)
        return self._mesh


Datasets = Union[
    Circle,
    PointCloud,
]
