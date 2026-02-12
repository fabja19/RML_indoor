import torch
from pathlib import Path

from lib import  utils
from script_complete_test_runs import find_incomplete_runs, load_config_and_create_dataset, create_model_from_config





def complete_test_run(
        run_dir: Path, 
        device: torch.device, 
        pert: bool | str, 
        test_dir_name: str, 
        skip_images: bool, 
        observation_percentage: float, 
        env_raster_subdir : str | None
    ) -> bool:
    """
    Complete the test evaluation for a single run with varying observation percentages.
    
    Args:
        run_dir: Path to the run directory
        device: PyTorch device to use
        pert: Perturbation setting (True/False/'tx'/'rasters')
        test_dir_name: Name of test directory to create
        skip_images: Whether to skip saving prediction images
        observation_percentage: Percentage of observations to use
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

        params_file = run_dir / 'parameters.json'
        config, Radio_test = load_config_and_create_dataset(params_file, pert=pert, env_raster_subdir=env_raster_subdir, observation_percentage=observation_percentage)
        
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
        env_raster_subdir: str | None = None
    ) -> None:
    """
    Complete test evaluations with varying observation percentages.
    
    Args:
        logs_dir: Base logs directory to search for incomplete runs
        dry_run: If True, only find incomplete runs without running tests
        run_dir: Process only a specific run directory (mutually exclusive with logs_dir)
        skip_images: Whether to skip saving prediction images
        L_shaped: Whether to use L-shaped room test split
        pert: Perturbation setting (True/False/'tx'/'rasters')
        env_raster_subdir: Optional override for environment raster subdirectory
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    
    if logs_dir is not None:
        logs_base_dir = Path(logs_dir)
        if not logs_base_dir.exists():
            raise FileNotFoundError(f"Error: Logs directory '{logs_base_dir}' does not exist")

    for obs_percentage in [0.25, 0.5, 1, 2, 4, 8, 16]:
        test_dir_name = f'test_{pert=}_{obs_percentage=}'

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
            continue
        
        print(f"Found {len(incomplete_runs)} incomplete runs for {obs_percentage=}")
        
        if dry_run:
            continue
        
        successful = 0
        failed = 0
        
        for run_dir_item in incomplete_runs:
            if complete_test_run(run_dir_item, device, pert=pert, env_raster_subdir=env_raster_subdir, test_dir_name=test_dir_name, skip_images=skip_images, observation_percentage=float(obs_percentage)):
                successful += 1
            else:
                failed += 1

        print(f"Successful: {successful}, Failed: {failed}, Total: {len(incomplete_runs)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete test evaluations with varying observation percentages")
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
