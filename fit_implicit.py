import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable
import tyro
from tqdm.auto import tqdm
import jax
from jax import numpy as jnp
from jax import random as jrnd
from datetime import datetime
import pickle
import time
from orbax.checkpoint import CheckpointManagerOptions, CheckpointManager, PyTreeCheckpointer
import optax
from flax.training import orbax_utils
import numpy as np
import wandb
from functools import partial
import pandas as pd
import subprocess
from trimesh import Trimesh

from diffcd import training, utils, methods, datasets, networks
from evaluation import chamfer, meshing

# override default checkpoint manager options
@dataclass
class CustomCheckpointManagerOptions(CheckpointManagerOptions):
    save_interval_steps: int = 1000

    # name of metric to use for checkpointing
    best_metric: Optional[str] = None

def replace_none(value, replace_value):
    """replaces `value` with `replace_value` if `value` is None`"""
    return replace_value if value is None else value

@dataclass
class TrainingConfig:
    # dataset config
    dataset: datasets.PointCloud

    # Path to directory where experiment results will be saved
    output_dir: Path

    # Config for implicit function f(theta, x)
    model: networks.MLP

    # Config for model checkpointing
    checkpoint_options: CustomCheckpointManagerOptions

    # Config for ground truth mesh for computing shape metrics
    gt_mesh: datasets.EvaluationMesh = field(default_factory=lambda: datasets.EvaluationMesh())

    # Config for converting the estimated SDF to a mesh
    eval_meshing: meshing.Meshing = field(default_factory=lambda: meshing.Meshing())

    # Number of points per axis to use for meshing final shape
    final_mesh_points_per_axis: int = 512

    # Name of current experiment (a folder with this name will be created in output_dir)
    experiment_name: str = "experiment"

    # Whether to append a timestamp to the experiment name
    with_timestamp: bool = True

    # Path to yaml config file with default settings
    yaml_config: Path = None

    # Whether to copy the datasets to the output directory.
    copy_datasets: bool = True

    # Whether to save a .ply file at each evaluation step (only for 3D datasets)
    save_ply: bool = True

    learning_rate: float = 1e-3
    learning_rate_warmup: int = 1000

    batch_size: int = 5000
    n_batches: int = 40000
    rng_seed: int = 0

    # wandb logging settings, override wandb_project to enable logging
    wandb_project: Optional[str] = None
    wandb_entity: str = 'dcp-sdf'
    wandb_name: Optional[str] = None

    method: methods.Methods = field(default_factory=lambda: methods.DiffCD())

    @property
    def experiment_dir(self):
        return self.output_dir / self.experiment_name

def save_config(config: TrainingConfig, output_dir: Path, name: str='config'):
    with open(output_dir / f'{name}.yaml', 'w') as yaml_file:
        yaml_file.write(tyro.extras.to_yaml(config))


    # save as pickle as well since yaml loading can break between versions
    with open(output_dir / f'{name}.pickle', 'wb') as pickle_file:
        pickle.dump(config, pickle_file)

def load_config(experiment_dir: Path, name: str='config'):
    try:
        with open(experiment_dir / f'{name}.yaml', 'r') as yaml_file:
            return tyro.extras.from_yaml(TrainingConfig, yaml_file)
    except Exception as e:
        print(f'WARNING: failed to load config from yaml config from {experiment_dir} due to "{e}", probably a result of version mismatch. Loading pickle file instead.')
        with open(experiment_dir / f'{name}.pickle', 'rb') as pickle_file:
            return pickle.load(pickle_file)

def check_best(metrics: list[dict], latest_metrics: dict, metric_name: str):
    if (metric_name is None) or (len(metrics) == 0):
        return True
    else:
        return latest_metrics[metric_name] < min([m[metric_name] for m in metrics])

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = {f'gpu{i}': int(x.split()[0]) for i, x in enumerate(memory_free_info)}
    return memory_free_values

def eval(gt_mesh: Trimesh, n_samples: int, meshing_config: meshing.Meshing, f: Callable, transform: Callable, batch_index: int):
    """
    Evaluate implicit surface

    Args:
        gt_mesh: ground truth mesh
        n_samples: number of surface samples to use for metric calculations
        meshing_config: config for converting implicit surface to a mesh
        f: function defining implicit surface via f(x) = 0
        transform: transform to apply to vertices of extracted mesh
        batch_index: index of current batch
    """
    estimated_mesh = meshing.extract_mesh(meshing_config, f)
    estimated_mesh.vertices = transform(estimated_mesh.vertices)

    chamfer_metrics = chamfer.compute_chamfer(gt_mesh, estimated_mesh, n_samples)
    hausdorff_metrics = chamfer.compute_hausdorff(gt_mesh, estimated_mesh, n_samples)
    metrics = {
        'step': batch_index,
        **chamfer_metrics,
        **hausdorff_metrics,
        **get_gpu_memory(),
    }
    return metrics, estimated_mesh


def save_checkpoint(checkpoint_manager, train_state, checkpoint_info, save_args, batch_index):
    checkpoint_manager.save(
        step=batch_index,
        items={'model': train_state, **checkpoint_info},
        save_kwargs={'save_args': save_args},
        force=True,
    )

def cos_with_warmup(init_lr, warm_up, max_iters, step):
    lr = jnp.where(step < warm_up, step / warm_up, 0.5 * (jnp.cos((step - warm_up)/(max_iters - warm_up) * jnp.pi) + 1))
    return lr * init_lr

def run(config: TrainingConfig):
    '''Run training'''
    if config.yaml_config is not None:
        print(f"\033[92mYAML config {config.yaml_config} provided.")
        with open(config.yaml_config, 'r') as yaml_file:
            defaults = tyro.extras.from_yaml(TrainingConfig, yaml_file)
            config = tyro.cli(TrainingConfig, default=defaults)

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S") if config.with_timestamp else ''
    config.experiment_name += timestamp_str
    experiment_dir = config.output_dir / config.experiment_name

    print(f'---\033[93m{config.experiment_name}\033[0m---')
    print(f'\033[96mexperiment dir\033[0m: {config.experiment_dir}')
    utils.print_config(config)

    config.experiment_dir.mkdir(parents=True, exist_ok=False)
    save_config(config, config.experiment_dir)

    np.random.seed(config.rng_seed)
    key = jrnd.PRNGKey(config.rng_seed)
    train_points = config.dataset.get_normalized_points(key)

    init_input = jnp.ones(config.dataset.n_dims)

    key, = jrnd.split(key, 1)
    params = config.model.init(key, init_input)

    train_metrics, eval_metrics = [], []
    checkpoint_manager = CheckpointManager(
        directory=config.experiment_dir.resolve() / 'checkpoints',
        checkpointers=PyTreeCheckpointer(),
        options=config.checkpoint_options,
    )

    lr_function = partial(cos_with_warmup, config.learning_rate, config.learning_rate_warmup, config.n_batches)

    train_state = training.ShapeTrainState.create(
        apply_fn=config.model.apply,
        tx=optax.adam(learning_rate=lr_function),
        params=params,
        lower_bound=config.eval_meshing.lower_bound,
        upper_bound=config.eval_meshing.upper_bound,
    )
    checkpoint_info = {
        'data': [init_input],
        'scale_factor': config.dataset._scale_factor,
        'center_point': config.dataset._center_point,
    }
    save_args = orbax_utils.save_args_from_target({'model': train_state, **checkpoint_info})

    if config.wandb_project is not None:
        wandb.init(
            project=config.wandb_project,
            config=utils.config_to_json(config),
            name=replace_none(config.wandb_name, config.experiment_name),
            entity=config.wandb_entity,
        )

    key, = jrnd.split(key, 1)
    method_state = config.method.init_state(key, train_points, config.experiment_dir if config.copy_datasets else None)
    if config.copy_datasets:
        np.save(config.experiment_dir / 'train_points.npy', train_points)

    apply_fn = jax.jit(train_state.apply_fn)

    start_time = time.time()
    for batch_index in tqdm(range(config.n_batches)):
        if checkpoint_manager.should_save(batch_index):
            save_checkpoint(checkpoint_manager, train_state, checkpoint_info, save_args, batch_index)
            utils.save_metrics(train_metrics, experiment_dir, 'train_metrics')

            metrics, estimated_mesh = eval(config.gt_mesh.mesh, config.gt_mesh.n_samples, config.eval_meshing, partial(apply_fn, train_state.params), config.dataset.undo_normalization, batch_index)
            if config.save_ply:
                meshing.save_ply(estimated_mesh, config.experiment_dir / f'meshes/mesh_{batch_index}.ply')
            if config.wandb_project is not None:
                wandb.log({'eval': metrics})

            eval_metrics.append({**metrics, 'time': time.time() - start_time})
            utils.save_metrics(eval_metrics, experiment_dir, 'eval_metrics')


        # training step
        train_step_time = time.time()
        key, = jrnd.split(key, 1)
        method_state, *batch = config.method.get_batch(train_state, method_state, key, config.batch_size)
        batch_metrics, train_state, nan_grads = config.method.step(train_state, *batch)
        train_step_time = time.time() - train_step_time

        # stop if there were nans in gradients and save state for debugging
        if nan_grads:
            save_checkpoint(checkpoint_manager, train_state, checkpoint_info, save_args, batch_index)
            np.save(config.experiment_dir / 'debug_key.npy', key)
            raise ValueError("nan encountered in gradients. Checkpoint saved for debugging.")
        batch_metrics = {
            'step': batch_index,
            **batch_metrics,
            'train_step_time': train_step_time,
            'learning_rate': lr_function(train_state.step),
            'time': time.time() - start_time,
        }
        train_metrics.append(batch_metrics)
        if config.wandb_project is not None:
            wandb.log({'train': batch_metrics})

    print('\033[92mdone!\033[0m saving metrics...')
    batch_index += 1
    save_checkpoint(checkpoint_manager, train_state, checkpoint_info, save_args, batch_index)
    utils.save_metrics(train_metrics, experiment_dir, 'train_metrics')

    metrics, estimated_mesh = eval(config.gt_mesh.mesh, config.gt_mesh.n_samples, config.eval_meshing, partial(apply_fn, train_state.params), config.dataset.undo_normalization, batch_index)
    eval_metrics.append(metrics)
    if config.save_ply:
        meshing.save_ply(estimated_mesh, config.experiment_dir / f'meshes/mesh_{batch_index}.ply')
    if config.wandb_project is not None:
        wandb.log({'eval': metrics})

    eval_metrics.append({**metrics, 'time': time.time() - start_time})
    utils.save_metrics(eval_metrics, experiment_dir, 'eval_metrics')


    # do final eval with higher resolution marching cubes
    final_mesh_config = meshing.Meshing(
        config.final_mesh_points_per_axis,
        config.eval_meshing.lower_bound,
        config.eval_meshing.upper_bound,
    )
    final_metrics, final_mesh = eval(config.gt_mesh.mesh, config.gt_mesh.n_samples, final_mesh_config, partial(apply_fn, train_state.params), config.dataset.undo_normalization, batch_index)
    meshing.save_ply(final_mesh, config.experiment_dir / f'mesh_final_{batch_index}.ply')
    pd.DataFrame([final_metrics]).to_csv(config.experiment_dir / f'eval_metrics_final_{batch_index}.csv')

if __name__ == '__main__':
    run(tyro.cli(TrainingConfig))