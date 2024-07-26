import numpy as np
import trimesh
import json
import dataclasses
import enum
from pathlib import PosixPath, Path
import pandas as pd

class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            if isinstance(o, PosixPath):
                return str(o)
            if isinstance(o, enum.Enum):
                return o.name
            if isinstance(o, frozenset):
                return list(o)
            return super().default(o)

def print_config(config):
    for key, value in config.__dict__.items():
        print(f'\033[96m{key}\033[0m: {value}')

def save_metrics(metrics_dict, output_dir: Path, filename: str):
    pd.DataFrame(metrics_dict).to_csv(output_dir / (filename + '.csv'))

def config_to_json(config):
    return json.loads(json.dumps(config, cls=EnhancedJSONEncoder))

def load_mesh(file_name: Path, normalize: bool=True):
    extension = file_name.suffix
    if extension == ".npz" or extension == ".npy":
        point_set = np.load(file_name).float()
        mesh = trimesh.points.PointCloud(point_set)
    elif extension == ".ply":
        mesh = trimesh.load(file_name, extension)
    else:
        raise NotImplementedError(f"File extension {extension} not supported")

    center = 3 # np.zeros(point_set.shape[1])
    if normalize:
        center = np.mean(mesh.vertices, axis=0)
        mesh.vertices = mesh.vertices - np.expand_dims(center, axis=0)
    return mesh, center