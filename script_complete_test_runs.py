import torch
from pathlib import Path
import json

from lib import data_loading, utils, modulesIPPNet
from lib.utils import SHAPE_TARGET, FREQUENCY, RX_HEIGHT


def find_incomplete_runs(
        logs_base_dir: Path, 
        test_dir_name: str
    ) -> list[Path]:
    """
    Find log directories that have a checkpoint but are missing complete test results.
    
    Args:
        logs_base_dir: Base logs directory (e.g., 'logs/')
        test_dir_name: Name of the test directory to check for
        
    Returns:
        List of paths to incomplete run directories
    """
    incomplete_runs = []
    test_log_name = 'TestLog.txt'
    
    for model_dir in logs_base_dir.iterdir():
        if not model_dir.is_dir() or model_dir.name.endswith('.csv') or model_dir.stem == 'ray-tracing':
            continue
            
        for run_dir in model_dir.iterdir():
            if not run_dir.is_dir():
                continue
                
            # Check if this run has a checkpoint and parameters
            checkpoint_file = run_dir / 'BestModel.pt'
            params_file = run_dir / 'parameters.json'
            test_dir = run_dir / test_dir_name
            test_log_file = test_dir / test_log_name
            
            if not (checkpoint_file.exists() and params_file.exists()):
                continue
                
            test_incomplete = False
            
            if not test_dir.exists():
                test_incomplete = True
            elif not test_log_file.exists():
                test_incomplete = True
                print(f"Found run missing {test_log_name}: {run_dir}")
            else:
                try:
                    if test_log_file.stat().st_size < 50:
                        test_incomplete = True
                except:
                    test_incomplete = True
            
            if test_incomplete:
                incomplete_runs.append(run_dir)
    
    return incomplete_runs


def load_config_and_create_dataset(
        params_file: Path, 
        pert : bool | str, 
        env_raster_subdir : str | None, 
        observation_percentage : float | None = None
    ) -> tuple[dict, data_loading.DatasetSketch]:
    """
    Load configuration and create test dataset from parameters file.
    
    Args:
        params_file: Path to parameters.json file
        pert: Perturbation setting (True/False/'tx'/'rasters')
        env_raster_subdir: Optional override for environment raster subdirectory
        observation_percentage: Optional override for observation percentage
        
    Returns:
        Tuple of (config dict, test dataset)
    """
    with open(params_file, 'r') as f:
        config = json.load(f)

    if env_raster_subdir is not None:
        if 'bin' in env_raster_subdir:
            assert 'bin' in config['env_raster_subdir'], f'{config['env_raster_subdir']=} doesnt contain bin'
        else:
            assert not 'bin' in config['env_raster_subdir'], f'{config['env_raster_subdir']=} does contain bin'

    if isinstance(pert, str):
        if pert=='tx':
            pert_tx = True
            pert_rasters = False
        elif pert=='rasters':
            pert_tx = False
            pert_rasters = True
        else:
            raise ValueError(f'{pert=}')
    else:
        pert_tx = pert_rasters = None
    
    Radio_test = data_loading.DatasetSketch(
        phase='test',
        dataset_dir=config['dataset_dir'],
        env_raster_subdir=config['env_raster_subdir'] if env_raster_subdir is None else env_raster_subdir,
        env_radiomap_subdir=config['env_radiomap_subdir'],
        sample_id_file="split_rectangular_L-shaped_val=0.1_test=0.1.json",
        shape_target=SHAPE_TARGET,
        n_copies=10 if pert else 0,
        use_tx_one_hot=config['use_tx_one_hot'],
        use_tx_distance=config['use_tx_distance'],
        use_Tx_distToRx=config['use_Tx_distToRx'],
        use_log_distance=config['use_log_distance'],
        use_fspl=config['use_fspl'],
        rx_height=RX_HEIGHT,
        frequency_hz=FREQUENCY,
        observation_percentage=config.get('observation_percentage', 1.0) if observation_percentage is None else observation_percentage,
        use_augmentation=False,  # Always False for testing
        use_observation_mask_as_input=config.get('use_observation_mask_as_input', False),
        use_material_classes=config.get('use_material_classes', True),
        use_material_properties=config.get('use_material_properties', False),
        perturb_tx=pert_tx,
        perturb_rasters=pert_rasters,
    )

    return config, Radio_test


def create_model_from_config(
        config: dict, 
        Radio_test: data_loading.DatasetSketch
    ) -> torch.nn.Module:
    """
    Create model instance from configuration.
    
    Args:
        config: Configuration dictionary
        Radio_test: Test dataset (needed for input channel calculation)
        
    Returns:
        Model instance
    """
    model_class_name = config['model_class_name']
    
    model_class = getattr(modulesIPPNet, model_class_name, None)
    if model_class is None:
        raise ValueError(f"Model class '{model_class_name}' not found in modulesIPPNet.")
    
    model_args = {
        'inputs': data_loading.calculate_num_input(Radio_test),
        'initial_downsampling': int(config['raster_size'] / 32),
    }
    
    model = model_class(**model_args)
    
    return model


def complete_test_run(
        run_dir: Path, 
        device: torch.device, 
        pert : bool | str,
        test_dir_name : str, 
        skip_images : bool,
        env_raster_subdir : str | None
    ) -> bool:
    """
    Complete the test evaluation for a single run.
    
    Args:
        run_dir: Path to the run directory
        device: PyTorch device to use
        pert: Perturbation setting (True/False/'tx'/'rasters')
        test_dir_name: Name of test directory to create
        skip_images: Whether to skip saving prediction images
        env_raster_subdir: Optional override for environment raster subdirectory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        test_dir = run_dir / test_dir_name
        test_log_file = test_dir / 'TestLog.txt'
        
        # Skip if test already completed
        if test_log_file.exists():
            return True
        
        print(f"\nProcessing: {run_dir}")
        
        # Load configuration and create dataset
        params_file = run_dir / 'parameters.json'
        config, Radio_test = load_config_and_create_dataset(params_file, pert=pert, env_raster_subdir=env_raster_subdir)
        
        model = create_model_from_config(config, Radio_test)
        model = model.to(device)
        
        checkpoint_file = run_dir / 'BestModel.pt'
        model.load_state_dict(torch.load(checkpoint_file, map_location=device))
        
        batch_size = config['batch_size']
        test_dir = run_dir / test_dir_name
        
        utils.test_model(
            model=model,
            Radio_test=Radio_test,
            batch_size=batch_size,
            test_dir=test_dir,
            skip_images=skip_images,
            pred_steps=config.get('pred_steps', 1)
        )
        
        return True
        
    except Exception as e:
        print(f"Failed to process {run_dir}: {str(e)}")
        return False


def main(
        logs_dir: Path | str | None, 
        dry_run: bool, 
        run_dir: Path | str | None, 
        skip_images: bool, 
        pert: bool | str, 
        env_raster_subdir: str | None
    ) -> None:
    """
    Complete test evaluations for training runs.
    
    Args:
        logs_dir: Base logs directory to search for incomplete runs
        dry_run: If True, only find incomplete runs without running tests
        run_dir: Process only a specific run directory (mutually exclusive with logs_dir)
        skip_images: Whether to skip saving prediction images
        pert: Perturbation setting (True/False/'tx'/'rasters')
        env_raster_subdir: Optional override for environment raster subdirectory
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    
    if logs_dir is not None:
        logs_base_dir = Path(logs_dir)
        if not logs_base_dir.exists():
            raise FileNotFoundError(f"Logs directory '{logs_base_dir}' does not exist")
    else:
        if run_dir is None:
            raise ValueError("Either logs_dir or run_dir must be provided")

    
    test_dir_name = f'test_{pert=}'

    if env_raster_subdir is not None:
        test_dir_name += env_raster_subdir.replace('rasterized_projects_256x256', '')


    if run_dir:
        run_dir_path = Path(run_dir)
        if not run_dir_path.exists():
            raise FileNotFoundError(f"Run directory '{run_dir_path}' does not exist")
        incomplete_runs = [run_dir_path]
    else:
        incomplete_runs = find_incomplete_runs(logs_base_dir, test_dir_name=test_dir_name)
    
    if not incomplete_runs:
        print("No incomplete runs found")
        return
    
    print(f"Found {len(incomplete_runs)} incomplete runs")
    
    if dry_run:
        return
    
    successful = 0
    failed = 0
    
    for run_dir_item in incomplete_runs:
        if complete_test_run(run_dir_item, device, pert=pert, test_dir_name=test_dir_name, skip_images=skip_images, env_raster_subdir=env_raster_subdir):
            successful += 1
        else:
            failed += 1

    print(f"Successful: {successful}, Failed: {failed}, Total: {len(incomplete_runs)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete test evaluations for training runs")
    parser.add_argument("-l", "--logs-dir", type=str, default="logs", help="Base logs directory")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Only find incomplete runs")
    parser.add_argument("-r", "--run-dir", type=str, help="Process only a specific run directory")
    parser.add_argument("-s", "--skip-images", action='store_true', help="Skip saving images")
    parser.add_argument("-p", "--pert", type=str, default=False, help="Perturbation mode: True/False/'tx'/'rasters'")
    parser.add_argument("-e", "--env-raster-subdir", type=str, help="Override environment raster subdirectory")
    
    args = parser.parse_args()
    
    # Convert pert string to appropriate type
    pert_val = args.pert
    if isinstance(pert_val, str):
        if pert_val.lower() in ['true', '1']:
            pert_val = True
        elif pert_val.lower() in ['false', '0']:
            pert_val = False
        elif pert_val not in ['tx', 'rasters']:
            raise ValueError(f"Invalid pert value: {pert_val}. Must be True/False/'tx'/'rasters'")
    
    main(
        logs_dir=args.logs_dir,
        dry_run=args.dry_run,
        run_dir=args.run_dir,
        skip_images=args.skip_images,
        pert=pert_val,
        env_raster_subdir=args.env_raster_subdir,
    )
