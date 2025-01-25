import numpy as np
from dataclasses import dataclass
import contourpy

@dataclass
class Contour:
    vertices: np.array
    edges: np.array

    def save(self, output_dir):
        np.savez(output_dir, vertices=self.vertices, segments=self.edges)

    @classmethod
    def load(self, contour_dir):
        return Contour(**np.load(contour_dir))

def get_contour(inputs, sdf_values):
    lines, codes = contourpy.contour_generator(
        inputs[..., 0], inputs[..., 1], sdf_values,
        name='mpl2014', corner_mask=True,
        line_type=contourpy.LineType.SeparateCode,
        fill_type=contourpy.FillType.OuterCode,
        chunk_size=0
    ).lines(0.)

    edges, points = [], []
    start_index = 0
    for segment_points, segment_codes in zip(lines, codes):
        segment_edges = np.vstack([ # TODO: handle case when line is closed
            np.arange(0, len(segment_points)-1),
            np.arange(1, len(segment_points)),
        ]).T + start_index
        points.append(segment_points)
        edges.append(segment_edges)
        if segment_codes[-1] == 79:
            edges.append(np.array([start_index, start_index + len(segment_points) - 1]))
        start_index += len(segment_points)
    return Contour(np.vstack(points), np.vstack(edges))


def get_sample_points(contour: Contour, n_samples, seed=0):
    distances = np.linalg.norm(contour.vertices[contour.edges[..., 1]] - contour.vertices[contour.edges[..., 0]], axis=-1)
    cumulative_distances = np.hstack([0., np.cumsum(distances)])

    sample_distances = np.random.default_rng(seed).random(n_samples) * cumulative_distances[-1]
    edge_indices = np.searchsorted(cumulative_distances, sample_distances) - 1

    alphas = ((sample_distances - cumulative_distances[edge_indices]) / distances[edge_indices])[..., None]
    starts = contour.vertices[contour.edges[edge_indices, 0]]
    ends = contour.vertices[contour.edges[edge_indices, 1]]
    sample_points =  starts * (1 - alphas) + ends * alphas

    diffs = ends - starts
    normals = np.array([-diffs[:, 1], diffs[:, 0]]).T
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    return sample_points, normals