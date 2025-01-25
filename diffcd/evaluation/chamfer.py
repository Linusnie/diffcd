
"""from https://github.com/otaheri/chamfer_distance """

from functools import partial
import numpy as np
from scipy.spatial import cKDTree as KDTree
import trimesh


__all__ = ['compute_chamfer', 'compute_hausdorff']

def one_sided_chamfer(points1, points2):
    """Calculate (arg)min_j \|points1[i] - points2[j]\| for each i."""
    points2_kd_tree = KDTree(points2)
    distances12, closest_indices2 = points2_kd_tree.query(points1)
    return distances12, closest_indices2

def normal_angle_deg(normals1, normals2):
    dot_products = np.einsum('...i, ...i -> ...', normalize(normals1), normalize(normals2))
    return np.arccos(np.clip(dot_products, -1, 1)) * 180 / np.pi

def normalize(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)

def compute_shape_metrics(dist_fn, metric_name, shape1: trimesh.Trimesh, shape2: trimesh.Trimesh, num_mesh_samples):
    """
    Compute nearest-neighbor based metrics for two meshes or point clouds.

    distance metric: distance from each point in shape i to the closest point in shape j
    normal metric: angle between normal of each point in shape i and the normal of the closest point in shape j

    Each metric is aggregated over points using dist_fn which maps an array to a single value.
    """
    if (shape1 is None) or (shape2 is None) or (len(shape1.vertices) == 0) or (len(shape2.vertices) == 0):
        distance_metric, distance_square_metric, normal_angle_metric = np.nan, np.nan, np.nan
    else:
        points1, face_indices1 = trimesh.sample.sample_surface(shape1, num_mesh_samples)
        points2, face_indices2 = trimesh.sample.sample_surface(shape2, num_mesh_samples)

        distances12, closest_indices2 = one_sided_chamfer(points1, points2)
        distances21, closest_indices1 = one_sided_chamfer(points2, points1)
        distance_metric = dist_fn([dist_fn(distances12), dist_fn(distances21)])
        distance_square_metric = dist_fn([dist_fn(distances12 ** 2), dist_fn(distances21 ** 2)])

        if (face_indices1 is not None) and (face_indices2 is not None):
            normals1 = shape1.face_normals[face_indices1]
            normals2 = shape2.face_normals[face_indices2]
            normal_angle12 = normal_angle_deg(normals1, normals2[closest_indices2])
            normal_angle21 = normal_angle_deg(normals2, normals1[closest_indices1])
            normal_angle_metric = dist_fn([dist_fn(normal_angle12), dist_fn(normal_angle21)])

            # compute normal metric again with the normals of one of the meshes flipped, and select whichever is smallest
            normal_angle12 = normal_angle_deg(normals1, -normals2[closest_indices2])
            normal_angle21 = normal_angle_deg(-normals2, normals1[closest_indices1])
            normal_angle_metric_flipped = dist_fn([dist_fn(normal_angle12), dist_fn(normal_angle21)])

            normal_angle_metric = min(normal_angle_metric, normal_angle_metric_flipped)
        else:
            normal_angle_metric = np.nan
    return {
        f'{metric_name}_distance': distance_metric,
        f'{metric_name}_square_distance': distance_square_metric,
        f'{metric_name}_normal_angle': normal_angle_metric,
    }

compute_chamfer = partial(compute_shape_metrics, np.mean, 'chamfer')
compute_hausdorff = partial(compute_shape_metrics, np.max, 'hausdorff')
