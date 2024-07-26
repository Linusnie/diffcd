import jax.numpy as jnp
import flax
import numpy as np
from skimage.measure import marching_cubes
import trimesh
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

@flax.struct.dataclass
class Meshing:
    points_per_axis: int = 256
    lower_bound: Tuple[float, float, float] = (-.8, -.8, -.8)
    upper_bound: Tuple[float, float, float] = (.8, .8, .8)
    f_batch_size: int = 64 ** 3
    levelset: float = 0.

def get_grid(lower, upper, n):
    axis_values = (jnp.linspace(lower[i], upper[i], n) for i in range(len(lower)))
    grid_values = jnp.meshgrid(*axis_values, indexing='ij')
    return jnp.stack(grid_values, axis=-1)

def iter_grid(lower, upper, n, batch_size):
    n_dims = len(lower)
    axis_batch_size = max(1, int(batch_size / n ** (n_dims - 1)))

    axis_values = [jnp.linspace(lower[i], upper[i], n) for i in range(len(lower))]
    for i in range(0, n, axis_batch_size):
        grid_values = jnp.meshgrid(axis_values[0][i:i+axis_batch_size], *axis_values[1:], indexing='ij')
        yield jnp.stack(grid_values, axis=-1)

def extract_mesh(config: Meshing, f) -> trimesh.Trimesh:
    lower, upper = config.lower_bound, config.upper_bound
    n = config.points_per_axis

    outputs_numpy = []
    for inputs in iter_grid(lower, upper, n, config.f_batch_size):
        outputs_numpy.append(np.array(f(inputs)))
    outputs_numpy = np.concatenate(outputs_numpy).reshape(n, n, n)

    try:
        vertices, faces, normals, _ = marching_cubes(
            volume=outputs_numpy,
            level=config.levelset,
            spacing=(
                    (upper[0] - lower[0]) / config.points_per_axis,
                    (upper[1] - lower[1]) / config.points_per_axis,
                    (upper[2] - lower[2]) / config.points_per_axis,
            )
        )
        vertices += np.array(lower)[None]
    except ValueError:
        print('marching cubes: no 0-level set found')
        vertices, faces, normals = np.array([]).reshape((0, 3)), [], None
    return trimesh.Trimesh(vertices, faces, vertex_normals=normals)

def save_ply(mesh: trimesh.Trimesh, output_file):
    Path(output_file).parent.mkdir(exist_ok=True)
    with open(output_file, 'wb') as ply_file:
        ply_file.write(trimesh.exchange.ply.export_ply(mesh))
