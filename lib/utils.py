from scipy.interpolate import Rbf
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
import time
import copy
from collections import defaultdict
from datetime import datetime
import numpy as np
from typing import Any, Sequence, TypeVar
from torchvision.transforms import v2 as T

SHAPE_TARGET = (32, 32)
RX_HEIGHT = 1.0
FREQUENCY = 5.82e9
PL_MAX = -12
PL_TRNC = -71
SAMPLING_THRESHOLD = 50
PIXEL_SIZE = 9.6 / 32

def CustomTransformFunctional(inpt: Any, params: tuple[bool,int|float]) -> Any:
    """
    Applies custom transforms (horizontal flip and rotation) to input data.
    
    Args:
        inpt (Any): Input tensor to transform.
        params (tuple[bool, int|float]): Transformation parameters where params[0] is a boolean 
            for horizontal flip and params[1] is rotation angle in degrees.
    
    Returns:
        Any: Transformed input tensor.
    """
    if params[0]:
        inpt = T.functional.horizontal_flip(inpt)
    if params[1] != 0:
        inpt = T.functional.rotate(inpt, params[1])
    return inpt
    
def CustomTransformReverse(inpt : Any, params: tuple[bool,int|float]) -> Any:
    """
    Reverses custom transforms (rotation and horizontal flip) applied to input data.
    
    Args:
        inpt (Any): Input tensor to reverse-transform.
        params (tuple[bool, int|float]): Transformation parameters where params[0] is a boolean 
            for horizontal flip and params[1] is rotation angle in degrees.
    
    Returns:
        Any: Reverse-transformed input tensor.
    """
    if params[1] != 0:
        inpt = T.functional.rotate(inpt, -1 * params[1])
    if params[0]:
        inpt = T.functional.horizontal_flip(inpt)
    return inpt

def calc_loss_with_mask(
        pred : torch.Tensor, 
        target : torch.Tensor, 
        mask : torch.Tensor, 
        observation_mask : torch.Tensor | None,
        alpha : float,
        metrics : dict | None,
        reduction : str = 'mean',
    ) -> torch.Tensor:
    """
    Calculates the masked mean squared error (MSE) loss between predictions and targets, 
    considering only the valid pixels specified by the mask. Updates the provided metrics dictionary 
    with the computed loss. May weigh given observations stronger.

    Args:
        pred (torch.Tensor): Predicted tensor of shape (batch_size, 1, H, W) or (batch_size, H, W).
        target (torch.Tensor): Ground truth tensor of shape (batch_size, 1, H, W) or (batch_size, H, W).
        alpha (float): Weight for full map loss (0 <= alpha <= 1)
               Final loss = alpha * (full_map_loss) + (1-alpha) * (observation_loss)
        mask (torch.Tensor): Binary mask tensor indicating pixels considered for calculation of the loss, same shape as pred/target.
        observation_mask (torch.Tensor): Binary mask tensor indicating pixels with given observations, same shape as pred/target.
        metrics (dict|None): Dictionary to accumulate loss values under the key 'loss'.
        reduction (str, optional): Specifies the reduction to apply to the output: 
            'mean' for average loss over all valid pixels, 
            'none' for per-sample loss. Default is 'mean'.
            Note that for each sample in the batch, we always average over the spatial dimensions. The reduction
            argument only defines how/whether we combine the losses we obtain for each sample.
    Returns:
        torch.Tensor: The computed masked loss value (scalar for 'mean', tensor for 'none').
    Raises:
        ValueError: If an unsupported reduction type is provided.
    """
    assert 0 <= alpha <=1
    batch_size = pred.shape[0]
    
    # pred, target, mask, observation_mask  = pred.squeeze(), target.squeeze(), mask.squeeze(), observation_mask.squeeze() if observation_mask is not None else None
    pred_valid = mask * pred
    target_valid = mask * target
    
    ### loss per sample in the batch
    total_loss_per_sample = ((pred_valid - target_valid)**2).sum((-1, -2))
    valid_pixels_per_sample = mask.sum((-1, -2))
    total_loss = total_loss_per_sample.sum()
    total_pixels = valid_pixels_per_sample.sum()
    if metrics is not None:
        metrics['squared_errors_summed'] += total_loss.sum().item()
        metrics['valid_pixels_summed'] += total_pixels.sum().item()
    loss_full = total_loss / total_pixels if total_pixels > 0 else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    ### write into metrics and calculate value that will be retured
    if alpha == 1:
        if metrics is not None:
            metrics['loss'] += loss_full.item() * batch_size
        if reduction=='mean':
            loss_combined = loss_full
        elif reduction=='none':
            loss_combined = torch.where(valid_pixels_per_sample > 0, total_loss_per_sample / valid_pixels_per_sample, torch.tensor([0], dtype=pred.dtype, device=pred.device))
    else:
        assert observation_mask is not None
        loss_observations_per_sample = ((pred_valid - target_valid)**2 * observation_mask).sum((-1, -2))
        valix_pixels_observations_per_sample = observation_mask.sum((-1, -2))
        loss_observations_full = loss_observations_per_sample.sum() / valix_pixels_observations_per_sample.sum() if valix_pixels_observations_per_sample.sum() > 0 else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        loss_combined = alpha * loss_full + (1 - alpha) * loss_observations_full
        if metrics is not None:
            metrics['loss'] += loss_combined.item() * batch_size
            metrics['loss_full_map'] += loss_full.item() * batch_size
            metrics['loss_observations'] += loss_observations_full.item() * batch_size
        if reduction=='none':
            loss_combined = alpha * torch.where(valid_pixels_per_sample > 0, total_loss_per_sample / valid_pixels_per_sample, torch.tensor([0], dtype=pred.dtype, device=pred.device)) \
                    + (1 - alpha) * torch.where(valix_pixels_observations_per_sample > 0, loss_observations_per_sample / valix_pixels_observations_per_sample, torch.tensor(0.0, device=pred.device, dtype=pred.dtype))
    
    return loss_combined

def calc_loss_with_mask_exp_avg(        
        pred : torch.Tensor, 
        target : torch.Tensor, 
        mask : torch.Tensor, 
        metrics : dict | None,
        sample_id : list[torch.Tensor|list[torch.Tensor]],
        labels_old : dict[tuple[int,int], torch.Tensor],
        exp_avg_momentum : float, 
        exp_avg_warmup : int,
        epoch : int,
        reduction : str = 'mean',
    ) -> tuple[torch.Tensor,torch.Tensor]:
    if epoch == 0:
        ### fill dict of ground truths
        for k in range(len(sample_id[0])):
            key = (sample_id[0][k].item(), sample_id[1][k].item())
            transform_params = (sample_id[2][0][k].item(), sample_id[2][1][k].item())
            assert not key in labels_old.keys(), f'{key=} already exists in labels_old!'
            labels_old[key] = CustomTransformReverse(target[k], transform_params)
    if epoch <= exp_avg_warmup:
        target_here = target
    else:
        target_new_list = []
        for k in range(len(sample_id[0])):
            key = (sample_id[0][k].item(), sample_id[1][k].item())
            transform_params = sample_id[2][0][k].item(), sample_id[2][1][k].item()
            labels_old[key] = labels_old[key] * exp_avg_momentum + (1 - exp_avg_momentum) * CustomTransformReverse(pred[k].detach(),  transform_params)
            target_new_list.append(CustomTransformFunctional(labels_old[key], params=transform_params))

        target_here = torch.stack(target_new_list, 0)
    loss = calc_loss_with_mask(pred=pred, target=target_here, mask=mask, observation_mask=None, alpha=1, metrics=metrics, reduction=reduction)
    return loss, target_here


def print_metrics(
        metrics : dict[str,float], 
        epoch_samples : int, 
        phase : str, 
        str_dir : Path, 
    ) -> None:
    """
    Logs and prints normalized metric values for a given phase.

    Args:
        metrics (dict[str, float]): Dictionary containing metric names and their accumulated values.
        epoch_samples (int): Number of samples in the current epoch, used to normalize metric values.
        phase (str): Name of the current phase (e.g., 'train', 'val', 'test').
        str_dir (Path): Directory path where the log file ('Log.txt') will be appended.

    Returns:
        None
    """
    outputs1 = []
    for k in metrics.keys():
        outputs1.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    with open(str_dir / 'Log.txt', 'a') as f:
        print("{}: {}".format(phase, ", ".join(outputs1)), file=f)
        print("{}: {}".format(phase, ", ".join(outputs1)))        

def train_model(
        model : nn.Module, 
        optimizer : torch.optim.Optimizer, 
        scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau, 
        alpha : float,
        str_dir : Path, 
        stringerI : Path,
        dataloaders : dict[str,DataLoader], 
        num_epochs : int,
        pred_steps : int = 1,
        exp_avg_momentum : float = 1,
        exp_avg_warmup : int = 5
    ) -> nn.Module:
    """
    Trains a PyTorch model using the provided optimizer, scheduler, and dataloaders for a specified number of epochs.
    Tracks training and validation losses, saves the best model based on validation loss, and logs progress to a file.
    Args:
        model (nn.Module): The PyTorch model to train.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler.
        str_dir (Path): Directory path for saving logs and plots.
        stringerI (Path): File path to save the best model state dict.
        dataloaders (dict[str, DataLoader]): Dictionary containing 'train' and 'val' DataLoaders.
        num_epochs (int, optional): Number of training epochs.
        pred_steps (int=1): Give value >1 to feed the output again into the model to enhance the RM.
    Returns:
        nn.Module: The trained model with the best validation loss.
    """
    if exp_avg_momentum < 1:
        assert getattr(dataloaders['train'].dataset, 'return_sample_id', False)
    best_model = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    device = next(model.parameters()).device
    pl_max, pl_trnc = getattr(dataloaders['val'].dataset, 'pl_max'), getattr(dataloaders['val'].dataset, 'pl_trnc')
    if pred_steps > 1:
        assert getattr(dataloaders['val'].dataset, 'use_fspl', False), f'{pred_steps=} > 1 requires input fspl!'

    train_losses = []
    val_losses = []
    lr_reduced_epochs = []
    labels_old = {}

    for epoch in range(num_epochs):
        with open(str_dir / 'Log.txt', 'a') as f:
            print('-' * 10, file=f)
            print('Epoch {}/{}'.format(epoch, num_epochs - 1), file=f)
            print('-' * 10)
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        since = time.time()
        epoch_losses = {}

        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    with open(str_dir / 'Log.txt', 'a') as f:
                        print("learning rate", param_group['lr'], file=f)
                        print("learning rate", param_group['lr'])
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for batch_data in dataloaders[phase]:
                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()


                    # Expecting 4 outputs: inputs, targets, valid_shapes, observation_mask

                    if len(batch_data) == 5:
                        inputs, targets, mask, observation_mask, sample_id = batch_data  
                    elif len(batch_data) == 4:
                        inputs, targets, mask, observation_mask = batch_data  
                    elif len(batch_data) == 3:
                        inputs, targets, mask = batch_data  
                        observation_mask = None
                    

                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    mask = mask.to(device)
                    observation_mask = observation_mask.to(device) if observation_mask is not None else None

                    outputs1 = model(inputs)
                    
                    # Loss is computed against full dense ground truth
                    if exp_avg_momentum == 1 or phase=='val':
                        loss = calc_loss_with_mask(pred=outputs1, target=targets, mask=mask, observation_mask=observation_mask, alpha=alpha, metrics=metrics)
                        target_smooth = None
                    else:
                        loss, target_smooth = calc_loss_with_mask_exp_avg(pred=outputs1, target=targets, mask=mask, metrics=metrics, exp_avg_momentum=exp_avg_momentum, exp_avg_warmup=exp_avg_warmup, sample_id=sample_id, epoch=epoch, labels_old=labels_old)


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_samples += inputs.size(0)

                    if pred_steps > 1:
                        if exp_avg_momentum == 1:
                            raise NotImplementedError(f'We havent implemented pred_steps > 1 combined with exponential moving average GT.')
                        for step in range(pred_steps - 1):
                            inputs[:,-1,...] = torch.repeat_interleave(torch.repeat_interleave(outputs1.detach().squeeze(1), repeats=inputs.shape[-2] // outputs1.shape[-2], dim=-2), repeats=inputs.shape[-1] // outputs1.shape[-1], dim=-1)
                            outputs1 = model(inputs)
                            # Loss is computed against full dense ground truth
                            loss = calc_loss_with_mask(outputs1, targets, mask, observation_mask, alpha, metrics)
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                            epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase, str_dir)
            epoch_loss = metrics['loss'] / epoch_samples
            epoch_losses[phase] = epoch_loss

            ### save model state, images
            if phase == 'val' and epoch_loss < best_loss:
                with open(str_dir / 'Log.txt', 'a') as f:
                    print("saving best model", file=f)
                    print("saving best model")
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())
                
                torch.save(best_model, stringerI)

            if  10 * epoch % num_epochs == 0: 
            
                with torch.no_grad():
                    loss_per_sample = calc_loss_with_mask(outputs1, targets, mask, observation_mask, 1, metrics, reduction='none')
                
                tensor_dict = {}
                for k in range(inputs.shape[0]):
                    tens_to_check = [tens for tens in [inputs[k, -3:-2], targets[k]] if tens[tens > 0].numel() > 0]
                    if len(tens_to_check) == 0:
                        pl_min_plot, pl_max_plot = 0, 1
                    else:
                        pl_min_plot = min(torch.amin(tens[tens > 0]) for tens in tens_to_check)
                        pl_max_plot = max(torch.amax(tens[tens > 0]) for tens in tens_to_check)
                    try:
                            tensor_dict.update({
                            f'samples{k}' : (inputs[k,-3:-2], pl_min_plot, pl_max_plot),
                            f'dist/fspl{k}' : (inputs[k,-1:], 0, 1),
                            f'targets{k}, RMSE={float(torch.sqrt(loss_per_sample[k].squeeze()).numpy(force=True)) * (pl_max - pl_trnc):.1f}dB' : (targets[k], pl_min_plot, pl_max_plot),
                            f'outputs{k} (clipped)'  : (outputs1[k] * mask[k], pl_min_plot, pl_max_plot)
                        })
                    except:
                        raise ValueError(f'{loss_per_sample.shape=}')
                    if target_smooth is not None:
                        tensor_dict.update({
                            f'target smooth{k}' : (target_smooth[k] * mask[k], pl_min_plot, pl_max_plot)
                        })
                plot_dict(tensor_dict=tensor_dict,
                    in_batch_id=None,
                    save_path=str_dir / f'{phase}_sample_{epoch=}',
                    suptitle=f'Batch RMSE = {float(torch.sqrt(loss_per_sample.sum() / inputs.size(0)).numpy(force=True)) * (pl_max - pl_trnc):.1f}dB',
                    n_cols=4 if target_smooth is None else 5
                )


        train_losses.append(epoch_losses.get('train', None))
        val_losses.append(epoch_losses.get('val', None))
        
        ### logging and plotting
        time_elapsed = time.time() - since
        with open(str_dir / 'Log.txt', 'a') as f:
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=f)
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            now = datetime.now()
            print("now =", now, file=f)
            print("now =", now)
        
        with open(str_dir / 'Log.txt', 'a') as f:
            print('Best val loss: {:4f}\n\n\n'.format(best_loss), file=f)
            print('Best val loss: {:4f}'.format(best_loss))

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_losses['val'])
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr == scheduler.min_lrs[0]:
            break
        if new_lr < prev_lr:
            lr_reduced_epochs.append(epoch)


        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        if lr_reduced_epochs:
            plt.scatter(lr_reduced_epochs, [val_losses[i] for i in lr_reduced_epochs], color='red', marker='x', label='LR Reduced')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, max(0.003, 2*min(val_losses)))
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str_dir / 'loss_curve.png')
        plt.close()

    model.load_state_dict(best_model)
    return model

def train_AE_model(
        model : nn.Module, 
        optimizer : torch.optim.Optimizer, 
        scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau, 
        str_dir : Path, 
        stringerI : Path,
        dataloaders : dict[str,DataLoader], 
        num_epochs : int
    ) -> nn.Module:
    """
    Trains a PyTorch autoencoder model using the provided optimizer, scheduler, and dataloaders for a specified number of epochs.
    Tracks training and validation losses, saves the best model based on validation loss, and logs progress to a file.
    This is an adapted version of train_model.
    Args:
        model (nn.Module): The PyTorch AE model to train.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler.
        str_dir (Path): Directory path for saving logs and plots.
        stringerI (Path): File path to save the best model state dict.
        dataloaders (dict[str, DataLoader]): Dictionary containing 'train' and 'val' DataLoaders.
        num_epochs (int, optional): Number of training epochs.
    Returns:
        nn.Module: The trained model with the best validation loss.
    """
    best_model = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    device = next(model.parameters()).device
    pl_max, pl_trnc = getattr(dataloaders['val'].dataset, 'pl_max'), getattr(dataloaders['val'].dataset, 'pl_trnc')

    train_losses = []
    val_losses = []
    lr_reduced_epochs = []

    for epoch in range(num_epochs):
        with open(str_dir / 'Log.txt', 'a') as f:
            print('-' * 10, file=f)
            print('Epoch {}/{}'.format(epoch, num_epochs - 1), file=f)
            print('-' * 10)
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        since = time.time()
        epoch_losses = {}

        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    with open(str_dir / 'Log.txt', 'a') as f:
                        print("learning rate", param_group['lr'], file=f)
                        print("learning rate", param_group['lr'])
                model.train()
            else:
                model.eval()

            metrics = defaultdict(float)
            epoch_samples = 0

            for batch_data in dataloaders[phase]:
                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()


                    # Expecting 4 outputs: inputs, targets, valid_shapes, observation_mask
                    if len(batch_data) == 4:
                        inputs, targets, mask, observation_mask = batch_data  
                    else:
                        inputs, targets, mask = batch_data  
                        observation_mask = None
                    

                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    mask = torch.repeat_interleave(torch.repeat_interleave(mask.to(device), int(inputs.shape[-1] // mask.shape[-1]), -1), int(inputs.shape[-2] // mask.shape[-2]), -2)
                    observation_mask = observation_mask.to(device) if observation_mask is not None else None

                    outputs1 = model(inputs)
                    
                    loss = calc_loss_with_mask(outputs1, inputs, mask, observation_mask, 1, metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase, str_dir)
            epoch_loss = metrics['loss'] / epoch_samples
            epoch_losses[phase] = epoch_loss

            if phase == 'val' and epoch_loss < best_loss:
                with open(str_dir / 'Log.txt', 'a') as f:
                    print("saving best model", file=f)
                    print("saving best model")
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())
                
                torch.save(best_model, stringerI)
            
            if phase == 'val' and 10 * epoch % num_epochs == 0: 
                with torch.no_grad():
                    loss_per_sample = calc_loss_with_mask(outputs1, inputs, mask, observation_mask, 1, metrics, reduction='none').sum(1)
                

                for k in range(inputs.shape[0]):
                    pl_min_plot, pl_max_plot = 0, 1
                    tensor_dict = ({
                        f'targets{k}' : (inputs[k] * mask[k], pl_min_plot, pl_max_plot),
                        f'outputs{k} (clipped)'  : (outputs1[k] * mask[k], pl_min_plot, pl_max_plot)
                    })
                    plot_dict(tensor_dict=tensor_dict,
                        in_batch_id=None,
                        save_path=str_dir / f'val_sample_{epoch=}_{k}',
                        suptitle=f'Batch RMSE={float(torch.sqrt(loss_per_sample[k]).numpy(force=True)) * (pl_max - pl_trnc):.1f}dB',
                        n_cols=4
                    )



        train_losses.append(epoch_losses.get('train', None))
        val_losses.append(epoch_losses.get('val', None))

        time_elapsed = time.time() - since
        with open(str_dir / 'Log.txt', 'a') as f:
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=f)
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            now = datetime.now()
            print("now =", now, file=f)
            print("now =", now)
        
        with open(str_dir / 'Log.txt', 'a') as f:
            print('Best val loss: {:4f}\n\n\n'.format(best_loss), file=f)
            print('Best val loss: {:4f}'.format(best_loss))

        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_losses['val'])
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr == scheduler.min_lrs[0]:
            break
        if new_lr < prev_lr:
            lr_reduced_epochs.append(epoch)


        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        if lr_reduced_epochs:
            plt.scatter(lr_reduced_epochs, [val_losses[i] for i in lr_reduced_epochs], color='red', marker='x', label='LR Reduced')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, max(0.003, 2*min(val_losses)))
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(str_dir / 'loss_curve.png')
        plt.close()

    model.load_state_dict(best_model)
    return model




def print_metricsTest(
        metrics: dict[str, float], 
        epoch_samples: int, 
        phase: str, 
        pl_max : float,
        pl_trnc : float,
        log_name : Path | None
    ) -> None:
    """
    Logs and prints formatted metric values for a given training/testing phase.

    For each metric in `metrics`, computes the average value per sample and a derived root value in dB
    (i.e. for MSE in grayscale, calculates RMSE in dB), then writes the results to a log file and prints them to stdout.

    Args:
        metrics (dict[str, float]): Dictionary of metric names and their accumulated values.
        epoch_samples (int): Number of samples in the current epoch.
        phase (str): Name of the current phase (e.g., 'train', 'val', 'test').
        pl_max (float): Maximum path loss value used for dB calculation.
        pl_trnc (float): Lower threshold path loss value used for dB calculation.
        log_name (Path | None): Path to the log file to append results to, or None to skip logging.

    Returns:
        None
    """
    outputs1 = []
    for k in metrics.keys():
        outputs1.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
    if 'squared_errors_summed' in metrics.keys() and 'valid_pixels_summed' in  metrics.keys():
        outputs1.append("{}: {:4f}".format(f"RMSE in dB", (pl_max - pl_trnc) * (np.sqrt(metrics["squared_errors_summed"] / metrics["valid_pixels_summed"]))))

    if log_name is not None:
        with open(log_name, 'a') as f:
            print("{}: {}".format(phase, ", ".join(outputs1)), file=f)
    print("{}: {}".format(phase, ", ".join(outputs1)))

def test_model(
        model : nn.Module, 
        Radio_test : torch.utils.data.Dataset, 
        batch_size : int, 
        test_dir : Path,
        skip_images : bool,
        pred_steps : int
    ) -> None:
    """
    Evaluates the given model on a test dataset and logs performance metrics and visualizations.
    Args:
        model (nn.Module): The PyTorch model to be evaluated.
        Radio_test (torch.utils.data.Dataset): The test dataset containing input samples and ground truth.
        batch_size (int): Number of samples per batch for evaluation.
        test_dir (Path): Directory path where log and plots will be saved.
        skip_images (bool): Do not save images to disk, should speed up testing significantly.
        pred_steps (int): Give value >1 to feed the output again into the model to enhance the RM.
    Returns:
        None
    Raises:
        RuntimeError: If NaN values are detected in the loss or input/target tensors.
    Notes:
        - The function computes loss for each batch and generates plots for inputs, targets, and model outputs.
        - Metrics are accumulated and printed at the end of evaluation.
        - Execution time and timestamp are logged.
    """
    pl_max, pl_trnc = getattr(Radio_test, 'pl_max'), getattr(Radio_test, 'pl_trnc')
    test_dir.mkdir(exist_ok=True, parents=True)
    log_name = test_dir / 'TestLog.txt'

    since = time.time()
    model.eval()
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    epoch_samples = 0

    if pred_steps > 1:
        assert getattr(Radio_test, 'use_fspl', False), f'{pred_steps=} > 1 requires input fspl!'

    test_loader = DataLoader(Radio_test, batch_size=batch_size, shuffle=False, num_workers=1)
    for idb, batch_data in enumerate(test_loader):
        if len(batch_data) == 4:
            inputs, targets, mask, observation_mask = batch_data  
        else:
            inputs, targets, mask = batch_data  
            observation_mask = None
    

        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        observation_mask = observation_mask.to(device) if observation_mask is not None else None

        with torch.set_grad_enabled(False):
            # Feed all inputs to model (including sparse GT if present)
            outputs1 = model(inputs)
            
            # Loss computed against full dense ground truth
            loss = calc_loss_with_mask(outputs1, targets, mask, observation_mask, 1, metrics, reduction='none')
            if torch.any(torch.isnan(loss)):
                raise RuntimeError(f'NaN loss during test for {idb=}\n{torch.any(torch.isnan(inputs))=}\n{torch.any(torch.isnan(targets))=}\n{torch.amax(inputs)=}\t{torch.amin(inputs)=}')

        
            epoch_samples += inputs.size(0)

            outputs_to_plot = []
            if pred_steps > 1:
                for step in range(pred_steps - 1):
                    outputs_to_plot.append(outputs1.detach().clone())
                    inputs[:,-1,...] = torch.repeat_interleave(torch.repeat_interleave(outputs1.detach().squeeze(1), repeats=inputs.shape[-2] // outputs1.shape[-2], dim=-2), repeats=inputs.shape[-1] // outputs1.shape[-1], dim=-1)
                    outputs1 = model(inputs)
                    # Loss is computed against full dense ground truth
                    loss = calc_loss_with_mask(outputs1, targets, mask, observation_mask, 1, metrics, reduction='none')
                    
        
        if not skip_images:
            tensor_dict = {}
            for k in range(inputs.shape[0]):
                tens_to_check = [tens for tens in [inputs[k, -3:-2], targets[k]] if tens[tens > 0].numel() > 0]
                if len(tens_to_check) == 0:
                    pl_min_plot, pl_max_plot = 0, 1
                else:
                    pl_min_plot = min(torch.amin(tens[tens > 0]) for tens in tens_to_check)
                    pl_max_plot = max(torch.amax(tens[tens > 0]) for tens in tens_to_check)
                tensor_dict.update({
                    f'samples{k}' : (inputs[k,-3:-2], pl_min_plot, pl_max_plot),
                    f'dist{k}' : (inputs[k,-1:], 0, 1),
                    f'targets{k}, RMSE={float(torch.sqrt(loss[k].squeeze()).numpy(force=True)) * (pl_max - pl_trnc):.1f}dB' : (targets[k], pl_min_plot, pl_max_plot),
                    f'outputs{k} (masked)'  : (outputs1[k] * mask[k], pl_min_plot, pl_max_plot),
                    f'outputs{k} (full)'  : outputs1[k]
                })
                for outid, outint in enumerate(outputs_to_plot):
                    tensor_dict.update({
                        f'interm{outid} {k}' : (outint[k] * mask[k], pl_min_plot, pl_max_plot),
                    })
            plot_dict(tensor_dict=tensor_dict,
                in_batch_id=None,
                save_path=test_dir / f'test_sample_{idb}',
                suptitle=f'Batch RMSE = {float(torch.sqrt(loss.sum() / inputs.size(0)).numpy(force=True)) * (pl_max - pl_trnc):.1f}dB',
                n_cols=5 + len(outputs_to_plot)
            )



    print_metricsTest(
        metrics, 
        epoch_samples, 
        phase='test', 
        pl_max=pl_max, 
        pl_trnc=pl_trnc, log_name=log_name)


    time_elapsed = time.time() - since
    with open(log_name, 'a') as f:
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=f)
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        now = datetime.now()
        print("now =", now, file=f)
        print("now =", now)


def test_AE_model(
        model : nn.Module, 
        Radio_test : torch.utils.data.Dataset, 
        batch_size : int, 
        test_dir : Path,
        skip_images : bool
    ) -> None:
    """
    Evaluates the given autoencoder model on a test dataset and logs performance metrics and visualizations.
    Args:
        model (nn.Module): The PyTorch model to be evaluated.
        Radio_test (torch.utils.data.Dataset): The test dataset containing input samples and ground truth.
        batch_size (int): Number of samples per batch for evaluation.
        test_dir (Path): Directory path where log and plots will be saved.
        skip_images (bool): If True, skip saving images to disk for faster evaluation.
    Returns:
        None
    Raises:
        RuntimeError: If NaN values are detected in the loss or input/target tensors.
    Notes:
        - The function computes loss for each batch and generates plots for inputs, targets, and model outputs.
        - Metrics are accumulated and printed at the end of evaluation.
        - Execution time and timestamp are logged.
    """
    pl_max, pl_trnc = getattr(Radio_test, 'pl_max'), getattr(Radio_test, 'pl_trnc')
    log_name = test_dir / 'TestLog.txt'

    since = time.time()
    model.eval()
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    epoch_samples = 0

    test_loader = DataLoader(Radio_test, batch_size=batch_size, shuffle=False, num_workers=1)
    for idb, batch_data in enumerate(test_loader):
        if len(batch_data) == 4:
            inputs, targets, mask, observation_mask = batch_data  
        else:
            inputs, targets, mask = batch_data  
            observation_mask = None
    

        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = torch.repeat_interleave(torch.repeat_interleave(mask.to(device), int(inputs.shape[-1] // mask.shape[-1]), -1), int(inputs.shape[-2] // mask.shape[-2]), -2)
        observation_mask = observation_mask.to(device) if observation_mask is not None else None

        with torch.set_grad_enabled(False):
            # Feed all inputs to model (including sparse GT if present)
            outputs1 = model(inputs)
            
            # Loss computed against full dense ground truth
            loss = calc_loss_with_mask(outputs1, inputs, mask, observation_mask, 1, metrics, reduction='none')
            if torch.any(torch.isnan(loss)):
                raise RuntimeError(f'NaN loss during test for {idb=}\n{torch.any(torch.isnan(inputs))=}\n{torch.any(torch.isnan(targets))=}\n{torch.amax(inputs)=}\t{torch.amin(inputs)=}')


        
        epoch_samples += inputs.size(0)

        
        if not skip_images:
            tensor_dict = {}
            for k in range(inputs.shape[0]):
                pl_min_plot, pl_max_plot = 0, 1
                tensor_dict.update({
                    f'inputs{k} (full)'  : (inputs[k]* mask[k], pl_min_plot, pl_max_plot),
                    f'outputs{k} (masked)'  : (outputs1[k] * mask[k], pl_min_plot, pl_max_plot),
                })

            plot_dict(tensor_dict=tensor_dict,
                in_batch_id=None,
                save_path=test_dir / f'test_sample_{idb}',
                suptitle=f'Batch RMSE = {float(torch.sqrt(loss.sum() / inputs.size(0)).numpy(force=True)) * (pl_max - pl_trnc):.1f}dB',
                n_cols=5
            )



    print_metricsTest(
        metrics, 
        epoch_samples, 
        phase='test', 
        pl_max=pl_max, 
        pl_trnc=pl_trnc, log_name=log_name)


    time_elapsed = time.time() - since
    with open(log_name, 'a') as f:
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60), file=f)
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        now = datetime.now()
        print("now =", now, file=f)
        print("now =", now)

def plot_dict(tensor_dict: dict, in_batch_id: int | None, save_path: Path | None = None, suptitle : str | None = None, n_cols : int | None = None, close_fig : bool = True) -> None:
    """
    Plots all channels of tensors from tensor_dict into subplots with colorbar and key as title, then saves to save_path.
    
    Args:
        tensor_dict (dict[str, torch.Tensor | tuple]): Dictionary of tensors or tuples (tensor, vmin, vmax).
            Tensor shape is [B, H, W] or [B, C, H, W] if in_batch_id is not None, or [H, W] or [C, H, W] if in_batch_id is None.
        in_batch_id (int | None): Index in batch to plot. If None, tensors are assumed to have no batch dimension.
        save_path (Path | None, optional): Path to save the figure. If None, figure is not saved. Default is None.
        suptitle (str | None, optional): Super title for the figure. Default is None.
        n_cols (int | None, optional): Number of columns in subplot grid. Default is min(4, n_plots).
        close_fig (bool, optional): If True, close the figure after saving. Default is True.
    
    Returns:
        None
    """
    subplot_infos = []
    for key, entry in tensor_dict.items():
        if isinstance(entry, torch.Tensor):
            tensor = entry
            vmin = None
            vmax = None
        elif len(entry) == 3:
            tensor, vmin, vmax = entry
        else:
            raise ValueError(f'Expected either tensor or (tensor, vmin, vmax), but \n{entry=}')

        t = tensor[in_batch_id] if in_batch_id is not None else tensor
        arr = t.detach().cpu().numpy()
        if arr.ndim == 2:
            subplot_infos.append((key, arr, vmin, vmax))
        elif arr.ndim == 3:
            for c in range(arr.shape[0]):
                subplot_infos.append((f"{key} [ch {c}]", arr[c], vmin, vmax))
        else:
            raise ValueError(f"Tensor for key '{key}' must be 2D or 3D, got shape {arr.shape}")

    n_plots = len(subplot_infos)
    n_cols = min(4, n_plots) if n_cols is None else n_cols
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for ax, (title, img, vmin, vmax) in zip(axes, subplot_infos):
        im = ax.imshow(np.where(img==0, np.nan, img), cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        ax.axis('off')

    for ax in axes[n_plots:]:
        ax.axis('off')

    plt.tight_layout()
    if suptitle is not None:
        plt.suptitle(suptitle)
    if save_path is not None:
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path)
    if close_fig:
        plt.close(fig)


### conversion between Watts, dBm, gray
def W_dBm(x : np.ndarray) -> np.ndarray:
    """
    Converts power values from Watts to decibel-milliwatts (dBm).

    Parameters
    ----------
    x : np.ndarray
        Array of power values in Watts.

    Returns
    -------
    np.ndarray
        Array of power values in dBm.

    Notes
    -----
    Ignores divide-by-zero warnings; returns -inf for zero input values.
    """
    with np.errstate(divide='ignore'):
        return 10 * np.log10(x) + 30
    
def dBm_W(x : np.ndarray) -> np.ndarray:
    """
    Converts power values from decibel-milliwatts (dBm) to watts (W).

    Parameters
    ----------
    x : np.ndarray
        Input array containing power values in dBm.

    Returns
    -------
    np.ndarray
        Array of power values converted to watts (W).
    """
    return np.power(10, (x - 30)/10)


ArrayLike = TypeVar('ArrayLike', np.ndarray, torch.Tensor)

def gray_dBm(x : ArrayLike, pl_trnc : float, pl_max : float) -> ArrayLike:
    """
    Applies a linear transformation to an input array to scale its values between a lower threshold path loss and a maximum path loss in dBm.

    Parameters:
        x (np.ndarray): Input array of normalized values (typically in [0, 1]).
        pl_trnc (float): Lower threshold path loss value in dBm (lower bound).
        pl_max (float): Maximum path loss value in dBm (upper bound).

    Returns:
        np.ndarray: Array of scaled values in dBm.
    """
    return x * (pl_max - pl_trnc) + pl_trnc

def dBm_gray(x : ArrayLike, pl_trnc : float, pl_max : float, clip : bool) -> ArrayLike:
    """
    Normalizes dBm values to a grayscale range [0, 1].

    Parameters:
        x (np.ndarray): Input array of dBm values.
        pl_trnc (float): Lower threshold path loss value; values below this are mapped to 0.
        pl_max (float): Maximum value; values above this are mapped to 1.
        clip (bool): If True, output values are clipped to [0, 1]. If False, values may fall outside this range.

    Returns:
        np.ndarray: Array of normalized grayscale values.
    """
    y = (x - pl_trnc) / (pl_max - pl_trnc) 
    if isinstance(y, Tensor):
        return torch.clip(y, 0, 1) if clip else y
    else:
        return np.clip(y, 0, 1) if clip else y
    
h, w = SHAPE_TARGET
y = torch.repeat_interleave(torch.arange(h).reshape((1,1,-1)), w, 1) * PIXEL_SIZE
x = torch.repeat_interleave(torch.arange(w).reshape((1,-1,1)), h, 2) * PIXEL_SIZE

def calculate_fspl(tx_coords: Tensor, pl_trnc: float, pl_max: float, **kwargs) -> Tensor:
    """
    Calculate Free Space Path Loss (FSPL) for given transmitter coordinates.
    
    Args:
        tx_coords: Tensor of shape (batch_size, 3) containing transmitter coordinates (x, y, z)
        pl_trnc: Lower threshold path loss value in dBm
        pl_max: Maximum path loss value in dBm
        **kwargs: Additional keyword arguments (unused)
        
    Returns:
        Tensor of normalized FSPL values in range [0, 1]
    """
    tx_x, tx_y, tx_z = tx_coords[:,0].reshape((-1,1,1)), tx_coords[:,1].reshape((-1,1,1)), tx_coords[:,2].reshape((-1,1,1))
    # x,y need to be changed due to CS change!
    dist = torch.sqrt((tx_x - y) ** 2 + (tx_y - x) ** 2 + (tx_z - RX_HEIGHT) ** 2)
    eps = 1e-10
    fspl = 20 * torch.log10(torch.clip(dist, eps, None)) + 20 * np.log10(FREQUENCY) - 147.55
    # path loss/gain convention...
    fspl = -1 * fspl
    return dBm_gray(fspl, pl_trnc=pl_trnc, pl_max=pl_max, clip=True)


def calculate_fspl_with_offset(
    inputs: Tensor, 
    observation_mask: Tensor, 
    tx_coords: Tensor, 
    pl_trnc: float, 
    pl_max: float, 
    **kwargs
) -> Tensor:
    """
    Calculate FSPL with a constant offset fitted to observed measurements.
    
    Args:
        inputs: Input tensor containing observed measurements
        observation_mask: Boolean mask indicating which pixels have observations
        tx_coords: Tensor of transmitter coordinates (x, y, z)
        pl_trnc: Lower threshold path loss value in dBm
        pl_max: Maximum path loss value in dBm
        **kwargs: Additional keyword arguments (unused)
        
    Returns:
        Tensor of FSPL values with offset correction, clipped to [0, 1]
    """
    fspl_no_offset = calculate_fspl(tx_coords=tx_coords, pl_trnc=pl_trnc, pl_max=pl_max).unsqueeze(1)
    offset = 1 / torch.sum(observation_mask, (-1, -2), keepdim=True) * torch.sum(observation_mask * (inputs - fspl_no_offset), (-1, -2), keepdim=True)
    return torch.clip(fspl_no_offset + offset, 0, 1)
    

def calculate_fspl_with_gain_offset(
    inputs: Tensor, 
    observation_mask: Tensor, 
    tx_coords: Tensor, 
    pl_trnc: float, 
    pl_max: float, 
    **kwargs
) -> Tensor:
    """
    Calculate FSPL with both gain and offset fitted to observed measurements using least squares.
    
    Args:
        inputs: Input tensor containing observed measurements
        observation_mask: Boolean mask indicating which pixels have observations
        tx_coords: Tensor of transmitter coordinates (x, y, z)
        pl_trnc: Lower threshold path loss value in dBm
        pl_max: Maximum path loss value in dBm
        **kwargs: Additional keyword arguments (unused)
        
    Returns:
        Tensor of FSPL values with gain and offset correction, clipped to [0, 1]
    """
    fspl_no_offset = calculate_fspl(tx_coords=tx_coords, pl_trnc=pl_trnc, pl_max=pl_max)
    inputs, observation_mask = inputs.squeeze(1), observation_mask.squeeze(1)
    gains, offsets = [], []
    for b in range(inputs.shape[0]):
        observations_here = inputs[b][observation_mask[b]]
        if observations_here.numel() < 2:
            gain, offset = torch.tensor(1.0), torch.tensor(0.0)
        else:
            fspl_vals_here = fspl_no_offset[b][observation_mask[b]]
            A = torch.stack([fspl_vals_here, torch.ones_like(fspl_vals_here)], 1)
            gain, offset = torch.linalg.lstsq(A.to(torch.float64), observations_here, rcond=None)[0]
        gains.append(gain.item())
        offsets.append(offset.item())
    gains = torch.tensor(gains).reshape((-1, 1, 1, 1))
    offsets = torch.tensor(offsets).reshape((-1, 1, 1, 1))
    return torch.clip(gains * fspl_no_offset.unsqueeze(1) + offset, 0, 1)


def rbf_interpolate(
    inputs: Tensor, 
    observation_mask: Tensor, 
    param: float | None, 
    **kwargs
) -> Tensor:
    """
    Perform Radial Basis Function interpolation using multiquadric basis functions.
    
    Args:
        inputs: Input tensor containing observed measurements
        observation_mask: Boolean mask indicating which pixels have observations
        param: Epsilon parameter for RBF (shape parameter)
        **kwargs: Additional keyword arguments (unused)
        
    Returns:
        Tensor of interpolated values, clipped to [0, 1]
    """
    interpolated = []
    for b in range(inputs.shape[0]):
        observations_here = inputs[b][observation_mask[b]]
        x_here = x[observation_mask[b]]
        y_here = y[observation_mask[b]]
        # if observations_here.numel() < 3:
        #     interpolated.append(torch.ones(SHAPE_TARGET) * torch.nan)
        try:
            rbf = Rbf(x_here.numpy(), y_here.numpy(), observations_here.numpy(),
                function='gaussian', epsilon=param)
            # grid_x, grid_y = np.meshgrid(np.arange(SHAPE_TARGET[1]), np.arange(SHAPE_TARGET[0]))
            interp = rbf(x.flatten().numpy(), y.flatten().numpy()).reshape(SHAPE_TARGET)
            interpolated.append(torch.tensor(np.clip(interp, 0, 1)))
        except ZeroDivisionError:
            interpolated.append(torch.ones(SHAPE_TARGET) * np.nan)
    return torch.stack(interpolated).unsqueeze(1)
    

def tps_interpolate(
    inputs: Tensor, 
    observation_mask: Tensor, 
    param: float | None, 
    **kwargs
) -> Tensor:
    """
    Perform Thin Plate Spline interpolation.
    
    Args:
        inputs: Input tensor containing observed measurements
        observation_mask: Boolean mask indicating which pixels have observations
        param: Smoothing parameter for TPS
        **kwargs: Additional keyword arguments (unused)
        
    Returns:
        Tensor of interpolated values, clipped to [0, 1]
    """
    interpolated = []
    for b in range(inputs.shape[0]):
        observations_here = inputs[b][observation_mask[b]]
        x_here = x[observation_mask[b]]
        y_here = y[observation_mask[b]]
        try:
            rbf = Rbf(x_here.numpy(), y_here.numpy(), observations_here.numpy(),
                function='thin_plate', smooth=param)
            interp = rbf(x.flatten().numpy(), y.flatten().numpy()).reshape(SHAPE_TARGET)
            interpolated.append(torch.tensor(np.clip(interp, 0, 1)))
        except ZeroDivisionError:
            interpolated.append(torch.ones(SHAPE_TARGET) * np.nan)
    return torch.stack(interpolated).unsqueeze(1)


def get_rm_adjusted(inputs : torch.Tensor, observation_mask : torch.Tensor, rm : torch.Tensor) -> torch.Tensor:
    """
    Adjusts ray-tracing radio map by computing offset from sparse measurements.

    Calculates the mean difference between observed measurements and ray-tracing predictions,
    then applies this offset to the entire radio map for calibration. (MMSE)

    Args:
        inputs (torch.Tensor): Sparse measurement observations with shape (B, C, H, W).
        observation_mask (torch.Tensor): Binary mask indicating measurement locations with shape (B, 1, H, W).
        rm (torch.Tensor): Ray-tracing radio map predictions with shape (B, C, H, W).

    Returns:
        torch.Tensor: Adjusted radio map with the same shape as rm.

    Raises:
        Exception: If tensor shapes are incompatible or computation fails.
    """
    offset = None
    try:
        offset = 1 / torch.sum(observation_mask, (-1, -2), keepdim=True) * torch.sum(torch.where(observation_mask, inputs - rm, 0), (-1, -2), keepdim=True)
        return rm + offset
    except Exception as e:
        if offset is None:
            print(f'offset is None')
        else:
            print(f'{offset.shape=}')
        raise Exception(f'{inputs.shape=}\t{observation_mask.shape=}\t{rm.shape=}\n{e}')