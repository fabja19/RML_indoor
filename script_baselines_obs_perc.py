#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import json
import argparse
from typing import Callable
from collections import defaultdict
import time
from warnings import warn
from datetime import datetime

from lib import data_loading, utils
from lib.utils import PL_MAX, PL_TRNC, calculate_fspl, calculate_fspl_with_gain_offset, calculate_fspl_with_offset, rbf_interpolate, tps_interpolate

def run_test(interpolation_function : Callable, fun_param : float | None, dataloader : DataLoader, test_dir_out : Path, test_params : dict) -> None:
    """
    Run testing phase for an interpolation function on a given dataset.
    This function evaluates an interpolation function on test data, computing metrics,
    saving visualizations, and logging results.
    Args:
        interpolation_function (Callable): The interpolation function to test.
        fun_param (float | None): Parameter to pass to the interpolation function.
        dataloader (DataLoader): PyTorch DataLoader containing test batches with format
            (inputs, targets, mask, observation_mask, tx_coords).
        test_dir_out (Path): Output directory path where test results will be saved.
        test_params (dict): Dictionary of test parameters to save as JSON.
    Returns:
        None
    Side Effects:
        - Creates output directory if it doesn't exist
        - Saves visualization plot for first batch sample
        - Writes test metrics and timing to TestLog.txt
        - Saves test parameters as test_parameters.json
        - Prints timing information and current timestamp
        - Issues warnings if NaN losses are detected during testing
    """
    test_dir_out.mkdir(exist_ok=True, parents=True)
    
    metrics = defaultdict(float)
    epoch_samples = 0

    since = time.time()

    for idb, batch_data in enumerate(dataloader):
        inputs, targets, mask, observation_mask, tx_coords = batch_data  


        inputs = inputs.to(torch.float64)[...,::8,::8]
        targets = targets.to(torch.float64)


        with torch.set_grad_enabled(False):
            outputs1 = interpolation_function(inputs=inputs, observation_mask=observation_mask, tx_coords=tx_coords, pl_max=PL_MAX, pl_trnc=PL_TRNC, param=fun_param)
                    
            # Loss computed against full dense ground truth
            loss = utils.calc_loss_with_mask(outputs1, targets, mask, observation_mask, 1, metrics, reduction='none')
            if torch.any(torch.isnan(loss)):
                warn(f'\n\n{interpolation_function}\t{fun_param=}\nNaN loss during test for {idb=}\n{torch.any(torch.isnan(inputs))=}\n{torch.any(torch.isnan(targets))=}\n{torch.amax(inputs)=}\t{torch.amin(inputs)=}')
            epoch_samples += inputs.size(0)
            
            if idb == 0:
                utils.plot_dict({'inputs' : inputs.squeeze(), 'targets' : targets.squeeze(), 'outputs1' : outputs1.squeeze()}, save_path=test_dir_out / f'test_sample_{idb}.png', suptitle=f'{idb}', in_batch_id=None)

    log_name = test_dir_out / 'TestLog.txt'
    utils.print_metricsTest(metrics, epoch_samples, phase='test', pl_max=PL_MAX, pl_trnc=PL_TRNC, log_name=log_name)


    time_elapsed = time.time() - since
    with open(log_name, 'a') as f:
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=f)
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        now = datetime.now()
        print("now =", now, file=f)
        print("now =", now)

    
    with open(test_dir_out / 'test_parameters.json', 'w') as f:
        json.dump(test_params, f)
    print(f'############\n\n\n#########')


def main(logs_dir : Path, pert : bool) -> None:
    """
    Run baseline interpolation method tests with varying observation percentages.
    This function evaluates multiple interpolation methods (FSPL, RBF, TPS) across different
    observation percentages to assess their performance on radio map prediction tasks.
    Args:
        logs_dir (Path): Directory path where test results and logs will be saved
        pert (bool): Whether to include perturbation/augmentation with multiple copies
    Returns:
        None
    Notes:
        - Tests observation percentages: [0.25, 0.5, 1, 2, 4, 8, 16, 32]
        - Methods tested:
            * FSPL: Free Space Path Loss
            * FSPL with offset
            * FSPL with gain and offset  
            * RBF: Radial Basis Function interpolation with various smoothing parameters
            * TPS: Thin Plate Spline interpolation with various regularization parameters
        - Results are saved to structured directories under logs_dir/interpolation/
        - Skips tests if log file already exists to avoid recomputation
    """
    methods = {
        'fspl' : (calculate_fspl, [None]), 
        'fspl_offset' : (calculate_fspl_with_offset, [None]), 
        'fspl_gain_offset' : (calculate_fspl_with_gain_offset, [None]), 
        'rbf' : (rbf_interpolate, [None, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10]), 
        'tps' : (tps_interpolate, [0, 0.1, 0.5, 1, 2, 5, 10])
    }
    for obs_percentage in [0.25, 0.5, 1, 2, 4,8, 16, 32]:
        
        test_params = dict(
            n_copies=10 if pert else 0,
            observation_percentage=float(obs_percentage),
            sample_id_file="split_rectangular_L-shaped_val=0.1_test=0.1.json" 
            )
        
        dataset = data_loading.DatasetSketch(
            phase='test',
            dataset_dir="dataset/indoor_projects_filtered_250922",
            env_raster_subdir="rasterized_projects_256x256",
            env_radiomap_subdir="radio_maps_-12_-71",
            shape_target=(32, 32),
            use_tx_one_hot=False,
            use_tx_distance=False,
            use_Tx_distToRx=False,
            use_log_distance=False,
            use_fspl=False,
            rx_height=1.0,
            frequency_hz=5.82e9,
            observationFS=False,
            use_augmentation=False,
            use_observation_mask_as_input=False,
            pl_max=PL_MAX,
            pl_trnc=PL_TRNC,
            use_material_classes=False,
            use_material_properties=False,
            return_tx_coords=True,
            **test_params  # type: ignore
        )
        dataloader = DataLoader(dataset=dataset, batch_size=32)
        
        for name, (fun, param_list) in methods.items():
            for param in param_list:
                print(f'{name}\t{param=}\t{obs_percentage=}')
                test_dir_out = logs_dir / 'interpolation' / (f'{name}_{param}' if param is not None else name) / f'test_{pert=}_{obs_percentage=}'
                log_file = test_dir_out / 'TestLog.txt'
                if log_file.exists():
                    continue
                else:
                    run_test(interpolation_function=fun, fun_param=param, dataloader=dataloader, test_dir_out=test_dir_out, test_params=test_params)
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete missing test evaluations for training runs")
    parser.add_argument("-l", "--logs-dir", type=str, default="logs", help="Base logs directory (default: logs)")
    parser.add_argument("-p", "--perturbations", action='store_true', help="Use perturbations")
    args = parser.parse_args()
    
    main(logs_dir=Path(args.logs_dir), pert=args.perturbations)

