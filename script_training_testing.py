'''
This script runs the training loop and runs testing on the synthetic data afterwards.
For testing on different configurations (e.g. measurement data), use the testing script.
'''

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from datetime import datetime
import json
import numpy as np

from lib import data_loading, utils, modulesIPPNet
from lib.utils import RX_HEIGHT, FREQUENCY, SHAPE_TARGET

def main(
    observation_percentage: float,
    observation_percentage2: None | float,
    n_copies: int,
    use_tx_one_hot: bool,
    use_tx_distance: bool,
    use_Tx_distToRx: bool,
    use_log_distance: bool,
    use_fspl: bool,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    scheduler_mode: str,
    scheduler_factor: float,
    scheduler_patience: int,
    scheduler_threshold: float,
    scheduler_min_lr: float,
    dataset_dir: str,
    env_raster_subdir: str,
    env_radiomap_subdir : str,
    sample_id_file: str,
    model_class_name: str,
    use_data_augmentation : bool,
    use_observation_mask_as_input : bool,
    alpha : float,
    use_material_properties : bool,
    use_material_classes : bool,
    pred_steps : bool,
    momentum : float,
    warmup : int,
    perturb_tx : bool | None,
    perturb_rasters : bool | None
) -> None:
    
    model_class = getattr(modulesIPPNet, model_class_name, None)
    if model_class is None:
        raise ValueError(f"Model class '{model_class_name}' not found in modulesIPPNet.")
    if pred_steps > 1:
        use_fspl = True
    
    observation_percentage_final : float | tuple[float,float] = observation_percentage if (observation_percentage2 is None or observation_percentage2 <= observation_percentage) else (observation_percentage, observation_percentage2)
    

    ### directories
    str_dir = Path(f'logs/{model_class_name}/{datetime.now().strftime("%Y%m%d-%H:%M:%S")}')
    stringerI = str_dir / 'BestModel.pt'

    try: 
        Path(str_dir).mkdir(parents=True)
    except OSError as error: 
        print(error) 

    print(f'saving to {str(str_dir)}')


    ### save parameters, start log
    with open(Path(dataset_dir) / env_raster_subdir / 'rasterization_parameters.json', 'r') as f:
        raster_size = json.load(f)['x_steps']

    config = {
        "observation_percentage": observation_percentage_final,
        "raster_size": raster_size,
        "n_copies": n_copies,
        "use_tx_one_hot": use_tx_one_hot,
        "use_tx_distance": use_tx_distance,
        "use_Tx_distToRx": use_Tx_distToRx,
        "use_log_distance": use_log_distance,
        "use_fspl": use_fspl,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "scheduler_params": dict(
            mode=scheduler_mode,
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=scheduler_threshold,
            min_lr=scheduler_min_lr
        ),
        "dataset_dir": dataset_dir,
        "env_raster_subdir": env_raster_subdir,
        "env_radiomap_subdir":  env_radiomap_subdir,
        "sample_id_file": sample_id_file,
        "model_class_name": model_class_name,
        "use_observation_mask_as_input": use_observation_mask_as_input,
        "use_data_augmentation" : use_data_augmentation,
        "alpha" : alpha,
        "use_material_properties" : use_material_properties,
        "use_material_classes" : use_material_classes,
        "pred_steps" : pred_steps,
        "warmup" : warmup, 
        "momentum" : momentum,
        "perturb_tx" : perturb_tx,
        "perturb_rasters" : perturb_rasters
    }

    with open(str_dir / 'parameters.json', 'w') as f:
        json.dump(config, f)

    with open(str_dir / 'Log.txt', 'w') as f:
        for k, v in config.items():
            print(f'{k}:\t{v}', file=f)
        
        print('\n\nTraining Accuracy', file=f)
        print('-' * 20, file=f)

    ### torch setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.enabled

    ### initiate data
    Radio_train = data_loading.DatasetSketch(
        phase='train',
        dataset_dir=dataset_dir,  # Update path
        env_raster_subdir=env_raster_subdir,
        env_radiomap_subdir=env_radiomap_subdir,
        sample_id_file=sample_id_file,
        shape_target=SHAPE_TARGET,  # Adjust based on your target size
        n_copies=n_copies,  #  environment perturbation
        use_tx_one_hot=use_tx_one_hot,
        use_tx_distance=use_tx_distance,
        use_Tx_distToRx=use_Tx_distToRx,       # 3D distance
        use_log_distance=use_log_distance,      # 20 * log10(distance) in dB
        use_fspl=use_fspl,              # Free space path loss in dB
        rx_height=RX_HEIGHT,              # Receiver at 1m height
        frequency_hz=FREQUENCY,         # 5.82 GHz frequency for FSPL
        observation_percentage=observation_percentage_final,    # Sample 5% of pixels
        use_material_properties=use_material_properties,
        use_material_classes=use_material_classes,
        use_augmentation=use_data_augmentation,
        use_observation_mask_as_input=use_observation_mask_as_input,
        return_sample_id=momentum<1,
        perturb_tx=perturb_tx,
        perturb_rasters=perturb_rasters
    )

    Radio_val = data_loading.DatasetSketch(
        phase='val',
        dataset_dir=dataset_dir,
        env_raster_subdir=env_raster_subdir, 
        env_radiomap_subdir=env_radiomap_subdir,
        sample_id_file=sample_id_file,
        shape_target=SHAPE_TARGET,
        n_copies=n_copies,  
        use_tx_one_hot=use_tx_one_hot,
        use_tx_distance=use_tx_distance,
        use_Tx_distToRx=use_Tx_distToRx,  
        use_log_distance=use_log_distance,
        use_fspl=use_fspl,              
        rx_height=RX_HEIGHT,            
        frequency_hz=FREQUENCY,         
        observation_percentage=observation_percentage_final,    
        use_material_properties=use_material_properties,
        use_material_classes=use_material_classes,
        use_augmentation=use_data_augmentation,
        use_observation_mask_as_input=use_observation_mask_as_input,
        perturb_tx=perturb_tx,
        perturb_rasters=perturb_rasters
    )

    Radio_test = data_loading.DatasetSketch(
        phase='test',
        dataset_dir=dataset_dir,
        env_raster_subdir=env_raster_subdir,
        env_radiomap_subdir=env_radiomap_subdir,
        sample_id_file=sample_id_file, 
        shape_target=SHAPE_TARGET,
        n_copies=n_copies,  
        use_tx_one_hot=use_tx_one_hot,
        use_tx_distance=use_tx_distance,
        use_Tx_distToRx=use_Tx_distToRx,       
        use_log_distance=use_log_distance,     
        use_fspl=use_fspl,              
        rx_height=RX_HEIGHT,            
        frequency_hz=FREQUENCY,         
        observation_percentage=observation_percentage_final,
        use_material_properties=use_material_properties,
        use_material_classes=use_material_classes,
        use_observation_mask_as_input=use_observation_mask_as_input,
        use_augmentation=False,
        perturb_tx=perturb_tx,
        perturb_rasters=perturb_rasters
    )

    print(f"Training dataset size: {len(Radio_train)}")
    print(f"Validation dataset size: {len(Radio_val)}")
    print(f"Test dataset size: {len(Radio_test)}")

    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(Radio_val, batch_size=batch_size, shuffle=True, num_workers=4),
        'test': DataLoader(Radio_test, batch_size=batch_size, shuffle=False, num_workers=1)
    }

    print(f"Batches per epoch - Train: {len(dataloaders['train'])}, Val: {len(dataloaders['val'])}")

    ### model and optimizers
    model = model_class(
        inputs=data_loading.calculate_num_input(Radio_train), 
        initial_downsampling=int(raster_size / 32),
    ).to(device)

    optimizer_ft = optim.Adam(\
                              filter(lambda p: p.requires_grad,list(model.parameters())),\
                              lr=learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft,
        **config['scheduler_params'] # type: ignore
    )

    ### start training
    model = utils.train_model(
        model=model, 
        optimizer=optimizer_ft, 
        scheduler=scheduler, 
        str_dir=str_dir,
        stringerI=stringerI,
        dataloaders=dataloaders,
        num_epochs=num_epochs,
        alpha=alpha,
        pred_steps=pred_steps
    )
    
    ### testing with the parameters also used during training
    utils.test_model(
        model=model,
        Radio_test=Radio_test,
        batch_size=batch_size,
        test_dir= str_dir / 'test_default',
        skip_images=False,
        pred_steps=pred_steps
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and test RadioUNet or RadioNetAdapt model. " \
            "For binary inputs, give -r rasterized_projects_256x256_bin. For material property inputs give -nc -mp.")
    parser.add_argument("-o", "--observation-percentage", type=float, default=0., help="Fraction of pixels to sample as given observations from ground truth RM.")
    parser.add_argument("-o2", "--observation-percentage2", type=float, default=20., help="Give a second obsdervation percentage value o2 to draw observation_percentage randomly for each sample from the interval [o, o2].\
                        If o2<=o, we ignore it and we use o as a fixed value.")
    parser.add_argument("-c", "--n-copies", type=int, default=10, help="Number of environment perturbation copies to draw from. In the paper, SNDA corresponds to setting this to 10, whereas no SNDA corresponds to 0.")
    parser.add_argument("--no-tx-one-hot", action="store_true", help="Disable transmitter one-hot encoding.")
    parser.add_argument("--no-tx-distance", action="store_true", help="Disable transmitter distance feature.")
    parser.add_argument("--use-Tx-distToRx", action="store_true", help="Enable transmitter-to-receiver 3D distance feature.")
    parser.add_argument("--use-log-distance", action="store_true", help="Enable log-distance feature.")
    parser.add_argument("--use-fspl", action="store_true", help="Enable free space path loss feature (in dB).")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size for training, testing.")
    parser.add_argument("-e", "--num-epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--scheduler-mode", type=str, default='min', help="Scheduler mode for ReduceLROnPlateau.")
    parser.add_argument("--scheduler-factor", type=float, default=0.2, help="Scheduler factor for ReduceLROnPlateau.")
    parser.add_argument("-p", "--scheduler-patience", type=int, default=10, help="Scheduler patience for ReduceLROnPlateau.")
    parser.add_argument("--scheduler-threshold", type=float, default=5e-5, help="Scheduler threshold for ReduceLROnPlateau.")
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6, help="Minimum learning rate for scheduler. EarlyStopping is applied at this LR.")
    parser.add_argument("-d", "--dataset-dir", type=str, default='dataset/indoor_projects_filtered_250922', help="Path to dataset directory.")
    parser.add_argument("-r", "--env-raster-subdir", type=str, default='rasterized_projects_256x256', help="Subdirectory for rasterized environment data.")
    parser.add_argument("-rm", "--env-radiomap-subdir", type=str, default='radio_maps_-12_-71', help="Subdirectory for radio map data.")
    parser.add_argument("-m", "--model-class-name", type=str, default='RadioNetAnySize', help="Model class name in modulesIPPNet.")
    parser.add_argument("-a", "--deactivate-data-augmentation", action='store_true', help="Disable data augmentation: random flips/rotations.")
    parser.add_argument("-om", "--use_observation_mask_as_input", action='store_true', help="Add the observation mask to the input tensor (only if observations are activated).")
    parser.add_argument("-al", "--alpha", type=float, default=1, help="Alpha to weigh loss at given observation locations additionally.")
    parser.add_argument("-nc", "--no-material-classes", action="store_true", help="Dont give class rasters as input.")
    parser.add_argument("-mp", "--use-material-properties", action="store_true", help="Use material property rasters as input.")
    parser.add_argument("-ps", "--pred-steps", type=int, default=1, help="If >1, model can improve output further by running again.")
    parser.add_argument("-t", "--env-tx-subdir", type=str, default=None, help="Optional dir for tx locations, otherwise uses ")
    parser.add_argument("-pt", "--perturb-tx", action="store_true", default=None, help="Enable only transmitter position perturbation.")
    parser.add_argument("-pr", "--perturb-rasters", action="store_true", default=None, help="Enable only raster perturbation.")
    args = parser.parse_args()

    sample_id_file = "split_rectangular_L-shaped_val=0.1_test=0.1.json"

    main(
        observation_percentage=args.observation_percentage,
        observation_percentage2=args.observation_percentage2 if args.observation_percentage2 > args.observation_percentage else None,
        n_copies=args.n_copies,
        use_tx_one_hot=not args.no_tx_one_hot,
        use_tx_distance=not args.no_tx_distance,
        use_Tx_distToRx=args.use_Tx_distToRx,
        use_log_distance=args.use_log_distance,
        use_fspl=args.use_fspl,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        scheduler_mode=args.scheduler_mode,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        scheduler_threshold=args.scheduler_threshold,
        scheduler_min_lr=args.scheduler_min_lr,
        dataset_dir=args.dataset_dir,
        env_raster_subdir=args.env_raster_subdir,
        env_radiomap_subdir=args.env_radiomap_subdir,
        sample_id_file=sample_id_file,
        model_class_name=args.model_class_name,
        use_data_augmentation=not args.deactivate_data_augmentation,
        use_observation_mask_as_input=args.use_observation_mask_as_input,
        alpha=args.alpha,
        use_material_properties=args.use_material_properties,
        use_material_classes=not args.no_material_classes,
        pred_steps=args.pred_steps,
        momentum=args.momentum,
        warmup=args.warmup,
        perturb_tx=args.perturb_tx,
        perturb_rasters=args.perturb_rasters,
    )