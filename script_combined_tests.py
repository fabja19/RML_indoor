"""Script to run multiple test configurations efficiently by calling different test scripts.

This script systematically runs tests across different configurations:
- Different environment raster subdirectories (standard, bin_025, bin_1)
- Different perturbation modes (True, False, 'tx', 'rasters')
- Different observation percentages (via script_obs_perc)
- Different frequencies (via script_obs_perc_freqs)
"""
from pathlib import Path

from script_obs_perc import main as main_obs
from script_complete_test_runs import main as main_standard


def main(
        run_dir: Path, 
        skip_images: bool,
        bin : bool,
        no_obs : bool
    ) -> None:
    """
    Run comprehensive test suite on a single trained model.
    
    Args:
        run_dir: Path to the run directory containing BestModel.pt and parameters.json
        skip_images: Whether to skip saving prediction images
    """
    raster_list = [None, 'rasterized_projects_256x256_bin_025_fixed', 'rasterized_projects_256x256_bin_1_fixed'] if bin else [None, 'rasterized_projects_256x256_025', 'rasterized_projects_256x256_1']
    for env_raster_subdir in raster_list:
        for pert in [True, False, 'tx', 'rasters']:
            main_standard(logs_dir=None, dry_run=False, run_dir=run_dir, skip_images=skip_images, pert=pert, env_raster_subdir=env_raster_subdir)


    
    main_obs(logs_dir=None, dry_run=False, run_dir=run_dir, skip_images=skip_images, pert=False)
    ### standard pert is with 0.5m, add also with 0.25 amd 1m
    for env_raster_subdir in raster_list:
        ### perturb both, only Tx and only rasters
        for pert in [True, 'tx', 'rasters']:
            main_obs(logs_dir=None, dry_run=False, run_dir=run_dir, skip_images=skip_images, pert=pert, env_raster_subdir=env_raster_subdir)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run comprehensive test suite on a trained model across multiple configurations"
    )
    parser.add_argument("-r", "--run-dir", type=str, required=True, 
                        help="Path to run directory containing model checkpoint")
    parser.add_argument("-i", "--images", action='store_true', 
                        help="Save prediction images (default: skip images)")
    parser.add_argument("-b", "--bin", action='store_true', 
                        help="For models trained with binary inputs.")
    parser.add_argument("-no", "--no-obs", action='store_true', 
                        help="Run without observations used")
    
    args = parser.parse_args()
    
    run_dir_path = Path(args.run_dir)
    if not run_dir_path.exists():
        raise FileNotFoundError(f"Run directory '{run_dir_path}' does not exist")
    
    main(
        run_dir=run_dir_path,
        skip_images=not args.images,
        bin=args.bin,
        no_obs=args.no_obs
    )
