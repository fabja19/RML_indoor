from typing import Sequence, Any
from itertools import combinations
from pathlib import Path
import torch
from torch import Tensor
from torch.utils.data import Dataset
import json
from PIL import Image
import numpy as np
from torchvision.transforms import v2 as T
from warnings import warn

from torchvision.tv_tensors import Image as tv_image
import h5py
from math import comb

from .utils import dBm_gray

### some parameters we will likely not change
rm_file = 'project{}-{}.png'
rm_file2 = 'projects-project{}-x3d3_3_1-t001_0{}.r006.h5'

env_raster_file_classes = 'project{}_height{}_classes.png'
env_raster_file_props = 'project{}_height{}_{}.png'
raster_params_file = 'rasterization_parameters.json'

material_properties_considered = ['conductivity', 'permittivity', 'thickness']

tx_file = 'project{}_tx.json'
room_height = 2.76 
room_length_max = 9.6
room_diam_max = np.sqrt(2 * room_length_max**2)

### custom augmentation including rotation and flips at once - needed for exponential moving average labels
class CustomTransform(T.Transform):
    def __init__(self, prob_flip : float = 0.5, angles : Sequence[int|float] = (0, 90, 180, 270)) -> None:
        self.prob_flip = prob_flip
        self.angles = angles
        self.last_params = {}
        super().__init__()

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, bool|int|float]:
        apply_flip = (torch.rand(size=(1,)) < self.prob_flip).item()
        rot_angle = self.angles[int(torch.randint(0, len(self.angles), size=(1,)).item())]
        params = dict(apply_flip=apply_flip, rot_angle=rot_angle)
        self.last_params = params
        return params


    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if params['apply_flip']:
            inpt = T.functional.horizontal_flip(inpt)
        if params['rot_angle'] != 0:
            inpt = T.functional.rotate(inpt, params['rot_angle'])
        return inpt



### data loading functions shared between our different datasets
def load_env_inputs(
        inputs : Tensor,
        env_id_here : str | int, 
        env_raster_subdir : Path, 
        raster_heights : list[float], 
        n_classes : int,
        use_material_classes : bool,
        use_material_properties : bool
    ) -> Tensor:
    """
    Loads environment raster data as tensors for a given environment ID and raster heights.

    Args:
        inputs (Tensor): Input tensor to concatenate env data to.
        env_id_here (str | int): Identifier for the environment (may include copy ID).
        env_raster_subdir (Path): Directory containing the raster files.
        raster_heights (list[float]): List of heights for which raster files should be loaded.
        n_classes (int): Number of classes used for normalization of raster values.
        use_material_classes (bool): Load raster files of material classes as input.
        use_material_properties (bool): Load raster files of electromagnetic properties as input.

    Returns:
        Tensor: A concatenated tensor containing the loaded and normalized raster data for each specified height.
    
    Raises:
        AssertionError: If both use_material_classes and use_material_properties are False.
    """
    assert use_material_classes or use_material_properties
    if use_material_classes:
        env_raster_files_classes = [env_raster_subdir / env_raster_file_classes.format(env_id_here, h) for h in raster_heights]
        raster_arrays = []
        for file in env_raster_files_classes:
            with Image.open(file) as f:
                raster_arrays.append(Tensor(np.array(f) / (n_classes - 1)).unsqueeze(0))
        inputs = torch.cat([inputs, *raster_arrays])
    if use_material_properties:
        env_raster_files_props = [env_raster_subdir / env_raster_file_props.format(env_id_here, h, prop) for h in raster_heights for prop in material_properties_considered]
        raster_arrays = []
        for file in env_raster_files_props:
            with Image.open(file) as f:
                raster_arrays.append(Tensor(np.array(f) / (n_classes - 1)).unsqueeze(0))
        inputs = torch.cat([inputs, *raster_arrays])
    return inputs
    

def load_gt_image(
    shape_target: tuple[int, int],
    env_radiomap_subdir: Path,
    env_radiomap_mask_subdir: Path | None,
    env_id: int | str,
    tx_id: int,
    pl_trnc: float,
    pl_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads the ground truth image for a given environment and transmitter.
    Args:
        shape_target (tuple[int, int]): Target shape of the output ground truth image.
        env_radiomap_subdir (Path): Path to the directory containing radiomap files for the environment.
        env_radiomap_mask_subdir (Path | None): Path to optional radio maps to be used as mask, or None.
        env_id (int | str): Identifier for the environment.
        tx_id (int): Transmitter ID.
        pl_trnc (float): Path loss truncation value for processing dBm data.
        pl_max (float): Maximum path loss value for processing dBm data.
    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - Ground truth image as a NumPy array with the specified target shape.
            - Validity mask indicating where ground truth values are available.
    """
    gt_np = np.zeros(shape_target)
    gt_valid = np.full(shape_target, False)
    with Image.open(env_radiomap_subdir / rm_file.format(env_id, f"{tx_id}")) as f:
        gt_small_np = np.array(f) / 255
    if env_radiomap_mask_subdir is not None:
        with Image.open(env_radiomap_mask_subdir / rm_file.format(env_id, f"{tx_id}")) as f:
            arr_for_mask = np.array(f) 
    else:
        arr_for_mask = gt_small_np

    
    gt_small_valid = np.isfinite(arr_for_mask) & (arr_for_mask > 0)
    gt_small_np[~gt_small_valid] = 0
    
    gt_np[:gt_small_np.shape[0], :gt_small_np.shape[1]] = gt_small_np
    gt_valid[:gt_small_valid.shape[0], :gt_small_valid.shape[1]] = gt_small_valid
    return gt_np, gt_valid

def load_tx_inputs(
    env_tx_subdir: Path,
    env_id_here: int | str,
    tx_id: int,
    use_tx_one_hot: bool,
    inputs: Tensor,
    x_step_size: float,
    y_step_size: float,
    use_tx_distance: bool,
    use_Tx_distToRx: bool,
    rx_height: float,
    use_log_distance: bool,
    use_fspl: bool,
    frequency_hz: float,
    pl_max : float,
    pl_trnc : float,
    return_coords: bool = False
) -> Tensor | None | tuple[Tensor,tuple[float]]:
    """
    Loads and encodes transmitter (Tx) information for a given environment and transmitter ID, 
    augmenting the input tensor with various Tx-related features as specified by the flags.
    Parameters:
        env_tx_subdir (Path): Directory containing environment tx location files.
        env_id_here (int | str): Identifier for the environment.
        tx_id (int): Identifier for the transmitter.
        use_tx_one_hot (bool): If True, adds a one-hot encoded Tx location channel to inputs.
        inputs (Tensor): Input tensor to be augmented.
        x_step_size (float): Step size in the x-direction for grid discretization.
        y_step_size (float): Step size in the y-direction for grid discretization.
        use_tx_distance (bool): If True, adds a normalized 2D distance channel from Tx to each grid point.
        use_Tx_distToRx (bool): If True, adds a normalized 3D distance channel from Tx to receiver plane at rx_height.
        rx_height (float): Height of the receiver plane for 3D distance calculations.
        use_log_distance (bool): If True, adds a normalized log-distance (in dB) channel from Tx to receiver plane.
        use_fspl (bool): If True, adds a normalized Free Space Path Loss (FSPL) channel for the given frequency.
        frequency_hz (float): Frequency in Hz for FSPL calculation.
        pl_max (float): Maximum path loss value for normalization.
        pl_trnc (float): Path loss truncation value for normalization.
        return_coords (bool): If True, returns tuple of (inputs, tx_coords), else returns inputs only.
    Returns:
        Tensor | tuple[Tensor, tuple[float]]: Augmented input tensor, or tuple of (inputs, tx_coords) if return_coords=True.
    """
    with open(env_tx_subdir / tx_file.format(env_id_here), 'r') as f:
        tx_coords = json.load(f)[str(tx_id)]
    if use_tx_one_hot:
        tx_one_hot = np.zeros(inputs.shape[1:])
        tx_one_hot[int(tx_coords[1] / x_step_size), int(tx_coords[0] / y_step_size)] = tx_coords[2] / room_height
        inputs = torch.concat([inputs, Tensor(tx_one_hot).unsqueeze(0)])
    
    if use_tx_distance:
        xrange, yrange = np.expand_dims(np.arange(inputs.shape[1]), 1) * x_step_size, np.expand_dims(np.arange(inputs.shape[2]), 0) * y_step_size
        distance = np.sqrt((tx_coords[1] - xrange)**2 + (tx_coords[0] - yrange)**2)
        inputs = torch.concat([inputs, Tensor(distance).unsqueeze(0) / room_diam_max])
        
    if use_Tx_distToRx:
        xrange = np.expand_dims(np.arange(inputs.shape[1]), 1) * x_step_size
        yrange = np.expand_dims(np.arange(inputs.shape[2]), 0) * y_step_size

        tx_x, tx_y, tx_z = tx_coords[0], tx_coords[1], tx_coords[2]
        distance_3d = np.sqrt((tx_x - yrange)**2 + (tx_y - xrange)**2 + (tx_z - rx_height)**2)
        max_3d_distance = np.sqrt(room_diam_max**2 + room_height**2)

        inputs = torch.concat([inputs, Tensor(distance_3d).unsqueeze(0) / max_3d_distance])
        
    if use_log_distance:
        xrange = np.expand_dims(np.arange(inputs.shape[1]), 1) * x_step_size
        yrange = np.expand_dims(np.arange(inputs.shape[2]), 0) * y_step_size

        tx_x, tx_y, tx_z = tx_coords[0], tx_coords[1], tx_coords[2]
        distance_3d = np.sqrt((tx_x - yrange)**2 + (tx_y - xrange)**2 + (tx_z - rx_height)**2)
        distance_tensor = torch.tensor(distance_3d, dtype=torch.float32)

        epsilon = 1e-10
        log_distance = 20 * torch.log10(torch.clamp(distance_tensor, min=epsilon))
        log_distance = (log_distance - log_distance.min()) / (log_distance.max() - log_distance.min())

        inputs = torch.concat([inputs, log_distance.unsqueeze(0)])

    if use_fspl:
        xrange = np.expand_dims(np.arange(inputs.shape[1]), 1) * x_step_size
        yrange = np.expand_dims(np.arange(inputs.shape[2]), 0) * y_step_size

        tx_x, tx_y, tx_z = tx_coords[0], tx_coords[1], tx_coords[2]
        distance_3d = np.sqrt((tx_x - yrange)**2 + (tx_y - xrange)**2 + (tx_z - rx_height)**2)
        distance_tensor = torch.tensor(distance_3d, dtype=torch.float32)

        epsilon = 1e-10
        fspl = (20 * torch.log10(torch.clamp(distance_tensor, min=epsilon)) + 
                20 * torch.log10(torch.tensor(frequency_hz)) - 147.55)
        fspl = torch.clip((fspl - pl_trnc) / (pl_max - pl_trnc), 0, 1)

        inputs = torch.concat([inputs, fspl.unsqueeze(0)])
    
    return inputs if not return_coords else (inputs, tx_coords)

def load_observation_data(
    gt: np.ndarray, 
    gt_mask: np.ndarray,
    observation_percentage: float | tuple[float,float],
    inputs: Tensor,
    use_observation_mask_as_input: bool,
    random_seed : int | None = None
) -> tuple[Tensor, Tensor]:
    """
    Loads and processes observation data for sparse ground truth generation.
    Observations are "samples" from the ground truth, added to the input tensor.
    Also generates a mask indicating observation locations, which is optionally added to inputs.

    Args:
        gt (np.ndarray): Ground truth as numpy array.
        gt_mask (np.ndarray): Ground truth mask indicating valid pixels.
        observation_percentage (float | tuple[float, float]): Percentage of valid pixels to sample as observations. 
            If tuple, draws random percentage within the range.
        inputs (Tensor): Input tensor to augment with observation data.
        use_observation_mask_as_input (bool): Whether to add observation mask as input channel.
        random_seed (int, optional): Random seed for reproducible sampling.
    Returns:
        tuple[Tensor, Tensor]: Updated inputs tensor and observation mask tensor.
    """
    if isinstance(observation_percentage, Sequence):
        assert len(observation_percentage) == 2
        assert 0<= observation_percentage[0] < observation_percentage[1] <= 100
        if random_seed is not None:
            rng = np.random.RandomState(random_seed)
            observation_percentage = float(rng.rand()) * (observation_percentage[1] - observation_percentage[0]) + observation_percentage[0]
        else:
            observation_percentage = float(np.random.rand()) * (observation_percentage[1] - observation_percentage[0]) + observation_percentage[0]

    allowed_indices = np.flatnonzero(gt_mask)
    num_allowed = allowed_indices.size
    num_observations = int(np.ceil(num_allowed * (observation_percentage / 100)))
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
        sampled_indices = rng.choice(allowed_indices, size=num_observations, replace=False)
    else:
        sampled_indices = np.random.choice(allowed_indices, size=num_observations, replace=False)

    observation_mask = np.zeros_like(gt_mask, dtype=bool)
    if len(sampled_indices) > 0:
        observation_mask.flat[sampled_indices] = 1

    observation_image = np.zeros_like(gt)
    if len(sampled_indices) > 0:
        observation_image.flat[sampled_indices] = gt.flat[sampled_indices]

    sparse_gt_tensor = torch.tensor(observation_image, dtype=torch.float32)
    observation_mask_tensor = torch.tensor(observation_mask, dtype=torch.bool)

    repeats_h, repeats_w = int(inputs.shape[1] // sparse_gt_tensor.shape[0]), int(inputs.shape[2] // sparse_gt_tensor.shape[1])
    sparse_gt_resized = torch.repeat_interleave(torch.repeat_interleave(sparse_gt_tensor, repeats_h, 0), repeats_w, 1)
    
    if use_observation_mask_as_input:
        observation_mask_resized = torch.repeat_interleave(torch.repeat_interleave(observation_mask_tensor, repeats_h, 0), repeats_w, 1)
        inputs = torch.concat([inputs, observation_mask_resized.unsqueeze(0)])

    inputs = torch.concat([inputs, sparse_gt_resized.unsqueeze(0)])


    return inputs, observation_mask_tensor

def get_measurement_points(
    idx: int, 
    measurements: dict, 
    n_clusters_test: int, 
    per_cluster_split: int
) -> tuple[str, list, list]:
    """
    Maps a linear sample index to observation points and GT points from measured data.
    
    Args:
        idx (int): Linear sample index
        measurements (dict): Dictionary of measurements with structure:
                                {Tx_id: {MeasurementArea_id: {point_id: {coordinates, pl, percentage_samples_available}}}}
        n_clusters_test (int): Number of measurement areas used for GT
        per_cluster_split (int): Number of splits per combination (determines subset size)
    
    Returns:
        tuple[str, list, list]: (selected_tx, observation_points, gt_points) where:
            - selected_tx: The transmitter ID used
            - observation_points: List of points from non-GT measurement areas
            - gt_points: List of points from GT measurement areas
    """
    
    tx_keys = list(measurements.keys())
    measurement_area_keys = list(measurements[tx_keys[0]].keys())
    n_measurement_combinations = comb(len(measurement_area_keys), n_clusters_test)
    total_combinations_per_tx = n_measurement_combinations * per_cluster_split
    
    # Determine which Tx, measurement area combination, and split
    tx_idx = idx // total_combinations_per_tx
    remaining_idx = idx % total_combinations_per_tx
    combination_idx = remaining_idx // per_cluster_split
    split_idx = remaining_idx % per_cluster_split
    
    selected_tx = tx_keys[tx_idx]
    measurement_combinations = list(combinations(measurement_area_keys, n_clusters_test))
    gt_measurement_areas = measurement_combinations[combination_idx]
    observation_measurement_areas = [area for area in measurement_area_keys if area not in gt_measurement_areas]
    
    # Collect points from GT measurement areas
    gt_points = []
    for area in gt_measurement_areas:
        area_data = measurements[selected_tx][area]
        points = [(data['coordinates'], data['pl']) 
                 for data in area_data.values()]
        
        subset_size = max(1, len(points) // per_cluster_split)
        start_idx = split_idx * subset_size
        end_idx = min((split_idx + 1) * subset_size, len(points))
        selected_points = points[start_idx:end_idx]
        gt_points.extend(selected_points)
    
    # Collect points from observation measurement areas
    observation_points = []
    for area in observation_measurement_areas:
        area_data = measurements[selected_tx][area]
        points = [(data['coordinates'], data['pl']) 
                 for data in area_data.values()]
        
        subset_size = max(1, len(points) // per_cluster_split)
        start_idx = split_idx * subset_size
        end_idx = min((split_idx + 1) * subset_size, len(points))
        selected_points = points[start_idx:end_idx]
        observation_points.extend(selected_points)

    return selected_tx, observation_points, gt_points

def load_gt_observations_from_dict(
        use_observations : bool,
        idx : int, 
        measurements: dict, 
        n_clusters_test: int, 
        per_cluster_split: int, 
        inputs : Tensor, 
        shape_target : tuple[int,int],
        pl_trnc : float,
        pl_max : float,
        use_observation_mask_as_input : bool
    ) -> tuple[int, Tensor, Tensor, Tensor, Tensor]:
    """
    Load ground truth and observation data from measured observations dictionary. 
    Observations and ground truth are mapped to grid coordinates and added to inputs as channels.
    A mask indicating the locations of observations is also returned and optionally added to inputs.
    
    Args:
        use_observations (bool): Whether to include observation data in the inputs tensor.
        idx (int): Sample index used to determine which measurement points to load.
        measurements (dict): Dictionary of measurements with structure:
                            {Tx_id: {MeasurementArea_id: {point_id: {coordinates, pl, percentage_samples_available}}}}
        n_clusters_test (int): Number of clusters to use for ground truth.
        per_cluster_split (int): Number of splits per combination for subset selection.
        inputs (Tensor): Input tensor to augment with observation channels.
        shape_target (tuple[int,int]): Target shape for ground truth and observation grids.
        pl_trnc (float): Path loss truncation value for dBm to gray conversion.
        pl_max (float): Maximum path loss value for dBm to gray conversion.
        use_observation_mask_as_input (bool): Whether to add observation mask as input channel.
        
    Returns:
        tuple[int, Tensor, Tensor, Tensor, Tensor]:
            - tx_id (int): Transmitter ID as integer.
            - inputs (Tensor): Input tensor with observation and optionally GT channels added.
            - sparse_gt_tensor (Tensor): Ground truth values on measurement grid.
            - gt_mask_tensor (Tensor): Boolean mask indicating GT measurement locations.
            - observation_mask_tensor (Tensor): Boolean mask indicating observation locations.
    """
    selected_tx, observation_points, gt_points = get_measurement_points(
        idx=idx, 
        measurements=measurements, 
        n_clusters_test=n_clusters_test, 
        per_cluster_split=per_cluster_split
    )
    
    gt_points_np = np.zeros(shape_target, dtype=np.float32)
    gt_mask_np = np.zeros(shape_target, dtype=bool)
    observation_points_np = np.zeros(shape_target, dtype=np.float32)
    observation_mask_np = np.zeros(shape_target, dtype=bool)
    
    for coordinates, pl_value in gt_points:
        gt_points_np[*coordinates] = dBm_gray(pl_value, pl_trnc=pl_trnc, pl_max=pl_max, clip=True)
        gt_mask_np[*coordinates] = 1
    
    for coordinates, pl_value in observation_points:
        observation_points_np[*coordinates] = dBm_gray(pl_value, pl_trnc=pl_trnc, pl_max=pl_max, clip=True)
        observation_mask_np[*coordinates] = 1
    sparse_gt_tensor = torch.tensor(gt_points_np, dtype=torch.float32)
    gt_mask_tensor = torch.tensor(gt_mask_np, dtype=torch.bool)
    sparse_observation_tensor = torch.tensor(observation_points_np, dtype=torch.float32)
    observation_mask_tensor = torch.tensor(observation_mask_np, dtype=torch.bool)
    
    repeats_h, repeats_w = int(inputs.shape[1] // sparse_observation_tensor.shape[0]), int(inputs.shape[2] // sparse_observation_tensor.shape[1])
    sparse_observation_resized = torch.repeat_interleave(torch.repeat_interleave(sparse_observation_tensor, repeats_h, 0), repeats_w, 1)

    if use_observation_mask_as_input:
        observation_mask_resized = torch.repeat_interleave(torch.repeat_interleave(observation_mask_tensor, repeats_h, 0), repeats_w, 1)
        inputs = torch.concat([inputs, observation_mask_resized.unsqueeze(0)])

    if use_observations:
        inputs = torch.concat([inputs, sparse_observation_resized.unsqueeze(0)])
    
    return int(selected_tx.replace('Tx', '')) - 1, inputs, sparse_gt_tensor, gt_mask_tensor, observation_mask_tensor


class DatasetSketch(Dataset):
    '''Draft for a dataset class loading observations from a list stored in a json file and incoporating project copies.'''  
    def __init__(self,
            phase : str,
            dataset_dir : str | Path,
            env_raster_subdir : str,
            env_radiomap_subdir : str,
            sample_id_file : str | None,
            shape_target : tuple[int,int],
            n_copies : int,
            use_tx_one_hot : bool,
            use_tx_distance : bool,
            use_Tx_distToRx : bool,
            use_log_distance : bool,
            use_fspl : bool,
            observation_percentage : float | tuple[float,float],  # NEW PARAMETER
            use_material_properties : bool,
            use_material_classes : bool,
            rx_height : float,
            frequency_hz : float,
            use_augmentation : bool,
            use_observation_mask_as_input : bool,
            pl_max : float | None = None,
            pl_trnc : float | None = None,
            return_tx_coords : bool = False,
            return_sample_id : bool = False,
            perturb_tx : bool | None = None,
            perturb_rasters : bool | None = None,
            env_radiomap_mask_subdir : str | None = None,
            **kwargs
            ) -> None:
        """
        Initializes the data loading class with configuration parameters for dataset handling, preprocessing, and augmentation.

        Args:
            phase (str): The phase of data usage ('train', 'val', 'test').
            dataset_dir (str | Path): Path to the root directory of the dataset.
            env_raster_subdir (str): Subdirectory name for environment raster data.
            env_radiomap_subdir (str): Subdirectory name for environment radiomap data.
            sample_id_file (str | None): Filename for sample IDs (JSON), or None if not used.
            shape_target (tuple[int, int]): Target shape for samples (height, width), to expand ground truth RMs.
            n_copies (int): Number of copies to draw from per sample (for augmentation).
            use_tx_one_hot (bool): Whether to use transmitter one-hot encoding as input.
            use_tx_distance (bool): Whether to use transmitter distance as input.
            use_Tx_distToRx (bool): Whether to use transmitter-to-receiver distance as input.
            use_log_distance (bool): Whether to use logarithmic distance as input.
            use_fspl (bool): Whether to use free-space path loss as input.
            observation_percentage (float): Percentage of observations to use (0 disables observation).
            use_material_properties (bool): Load raster files of electromagnetic properties as input.
            use_material_classes (bool): Load raster files of classes as input.
            rx_height (float): Receiver height in meters.
            frequency_hz (float): Frequency in Hz.
            use_augmentation (bool): Whether to apply data augmentation transforms.
            use_observation_mask_as_input (bool): Whether to use the observation mask as an input channel.
            pl_max (float | None, optional): Maximum path loss value (used if not found in JSON).
            pl_trnc (float | None, optional): Path loss truncation value (used if not found in JSON).
                (These are important when calculating loss on test set for data on full scale, like measurements,
                loaded from h5 files or jsons isntead of png images).
            return_tx_coords (bool=False): For interpolation methods
            env_tx_subdir (str | None = None): subdir with tx location files. If not given explicitely, same as for rasters.
            perturb_tx (bool | None = None): Whether to use perturbed Tx locations. If not given, determined by n_copies as before.
            perturb_rasters (bool | None = None): Whether to use perturbed raster env files. If not given, determined by n_copies as before.

        Raises:
            AssertionError: If required path loss parameters are missing and not provided.
        """
        if isinstance(observation_percentage, (float, int)):
            if use_observation_mask_as_input:
                assert observation_percentage > 0
        elif isinstance(observation_percentage, Sequence):
            assert len(observation_percentage) == 2
            assert 0 <= observation_percentage[0] < observation_percentage[1] <= 100
        else:
            raise ValueError(f'{type(observation_percentage)=}')


        self.dataset_dir = Path(dataset_dir)
        self.env_raster_subdir = self.dataset_dir / env_raster_subdir
        self.env_radiomap_subdir = self.dataset_dir / env_radiomap_subdir
        self.env_radiomap_mask_subdir = self.dataset_dir / env_radiomap_mask_subdir if env_radiomap_mask_subdir is not None else None
        try:
            with open(self.env_radiomap_subdir / 'rm_processing_parameters.json', 'r') as f:
                rm_params = json.load(f)
            self.pl_max, self.pl_trnc = rm_params['PL_max'], rm_params['PL_trnc']
        except:
            if pl_max is None or pl_trnc is None:
                raise ValueError(f'Could not load pl_max, pl_trnc from {str(self.env_radiomap_subdir)} (no json file there),\
                                 need to give them explicitly.')
            self.pl_max, self.pl_trnc = pl_max, pl_trnc

        if sample_id_file is not None:
            with open(self.dataset_dir / sample_id_file, 'r') as f:
                self.sample_ids = json.load(f)[f'sample_ids_{phase}']
        else:
            self.sample_ids = None
        self.phase = phase
        self.shape_target = shape_target
        self.n_copies = n_copies
        self.use_tx_one_hot = use_tx_one_hot
        self.use_tx_distance = use_tx_distance
        self.use_Tx_distToRx = use_Tx_distToRx
        self.use_log_distance = use_log_distance
        self.use_fspl = use_fspl
        self.rx_height = rx_height
        self.frequency_hz = frequency_hz
        self.observation_percentage = observation_percentage
        self.use_material_properties = use_material_properties
        self.use_material_classes = use_material_classes
        self.return_tx_coords = return_tx_coords
        self.return_sample_id = return_sample_id

        ### this is a weird construction, because initially we didnt differentiate between perturbed rasters and Tx locations, we only set n_copies
        if any([perturb_tx, perturb_rasters]):
            assert n_copies > 0, f'{n_copies=}\tbut {perturb_tx=}, {perturb_rasters=}'

        if perturb_tx is None and perturb_rasters is None:
            if n_copies > 0:
                self.perturb_tx = self.perturb_rasters = True
            else:
                self.perturb_tx = self.perturb_rasters = False
        else:
            self.perturb_tx = perturb_tx if perturb_tx is not None else False
            self.perturb_rasters = perturb_rasters if perturb_rasters is not None else False
        
        print(f'\n{self.perturb_rasters=}\t{self.perturb_tx=}')


        ### obtain number of classes and heights
        with open(self.env_raster_subdir / raster_params_file, 'r') as f:
            raster_params = json.load(f)
        self.n_classes = len(raster_params['materials']) + 1
        self.raster_heights = raster_params['heights']
        self.x_steps = raster_params['x_steps']
        self.y_steps = raster_params['y_steps']
        self.x_step_size = raster_params['x_max'] / raster_params['x_steps']
        self.y_step_size = raster_params['y_max'] / raster_params['y_steps']
        self.use_observations = isinstance(observation_percentage, Sequence) or observation_percentage > 0
        if not self.use_observations:
            print(f'\n\n\n{self.use_observations=}\nbecause {observation_percentage=}\n\n\n')
        self.use_observation_mask_as_input = use_observation_mask_as_input


        self.transform = CustomTransform() if use_augmentation else CustomTransform(prob_flip=0, angles=[0])
        

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of sample IDs.
        """
        if self.sample_ids is None:
            raise RuntimeError(f'For the base class DatasetSketch, {self.sample_ids=} should not happen.\
                               For subclasses, this is allowed but need to overwrite __len__.')
        return len(self.sample_ids)
    
    def get_ids(self, idx : int) -> tuple[int,int]:
        return self.sample_ids[idx]
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor | None]:
        """
        Retrieves a data sample for the given index.
        
        This method loads and processes input data, ground truth, and optional observation masks for a specific 
        environment and transmitter pair. It supports loading data from multiple copies of environments, applies 
        transformations, and handles observation if requested.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            Depending on configuration flags, returns tuple containing:
                - inputs (Tensor): Processed input tensor with shape (C, H, W).
                - gt (Tensor): Ground truth radio map with shape (H, W).
                - gt_mask (Tensor): Boolean mask indicating valid GT pixels with shape (H, W).
                - observation_mask (Tensor, optional): Mask showing observation locations with shape (H, W).
                  Included if use_observations=True.
                - tx_coords (Tensor, optional): Transmitter coordinates [x, y, z].
                  Included if return_tx_coords=True.
                - sample_info (tuple, optional): Tuple of (env_id, tx_id, transform_params).
                  Included if return_sample_id=True.
        
        Raises:
            RuntimeError: If self.sample_ids is None in the base class.
        """

        if self.sample_ids is None:
            raise RuntimeError(f'For the base class DatasetSketch, {self.sample_ids=} should not happen.\
                               For subclasses, this is ok but need to overwrite __len__ and __getitem__.')
        env_id, tx_id = self.sample_ids[idx]
        
        if self.n_copies > 0:
            if self.phase == 'test':
                rng = np.random.RandomState(idx * 1000) 
                copy_id = rng.randint(0, self.n_copies)
            else:
                copy_id = np.random.randint(0, self.n_copies)
            env_id_copy = f'{env_id}_copy{copy_id}'
            env_id_tx = env_id_copy if self.perturb_tx else env_id
            env_id_rasters = env_id_copy if self.perturb_rasters else env_id
        else:
            env_id_tx = env_id_rasters = env_id
        
        inputs = torch.empty((0, self.x_steps, self.y_steps))
        if self.use_material_properties or self.use_material_classes:
            inputs = load_env_inputs(
                inputs=inputs,
                env_id_here=env_id_rasters, 
                env_raster_subdir=self.env_raster_subdir, 
                raster_heights=self.raster_heights, 
                n_classes=self.n_classes,
                use_material_classes=self.use_material_classes,
                use_material_properties=self.use_material_properties
            )

        gt, gt_mask = load_gt_image(
            shape_target=self.shape_target,
            env_radiomap_subdir=self.env_radiomap_subdir,
            env_radiomap_mask_subdir=self.env_radiomap_mask_subdir,
            env_id=env_id,
            tx_id=tx_id,
            pl_trnc=self.pl_trnc,
            pl_max=self.pl_max,
        )

        observation_mask = None
        if self.use_observations:
            inputs, observation_mask = load_observation_data(
                gt=gt,
                gt_mask=gt_mask,
                observation_percentage=self.observation_percentage,
                inputs=inputs,
                use_observation_mask_as_input=self.use_observation_mask_as_input,
                random_seed=idx if self.phase=='test' else None
            )

        gt = Tensor(gt)
        inputs, tx_coords = load_tx_inputs(
            env_tx_subdir=self.env_raster_subdir,
            env_id_here=env_id_tx,
            tx_id=tx_id,
            use_tx_one_hot=self.use_tx_one_hot,
            inputs=inputs,
            x_step_size=self.x_step_size,
            y_step_size=self.y_step_size,
            use_tx_distance=self.use_tx_distance,
            use_Tx_distToRx=self.use_Tx_distToRx,
            rx_height=self.rx_height,
            use_log_distance=self.use_log_distance,
            use_fspl=self.use_fspl,
            frequency_hz=self.frequency_hz,
            pl_max=self.pl_max,
            pl_trnc=self.pl_trnc,
            return_coords=True
        ) # type: ignore

        return_vals = list(self.transform(tv_image(inputs), tv_image(gt), tv_image(gt_mask), tv_image(observation_mask)) if observation_mask is not None else self.transform(tv_image(inputs), tv_image(gt), tv_image(gt_mask)))
        if self.return_tx_coords:
            return_vals.append(torch.tensor(tx_coords))
        if self.return_sample_id:
            return_vals.append((env_id, tx_id, tuple(self.transform.last_params.values())))

        return tuple(return_vals)
        

class DatasetMeasurements(DatasetSketch):
    def __init__(self,
            sample_id_file : str,
            percentage_measurements_threshold : float,
            per_cluster_split : int,
            n_clusters_test : int,
            phase : str,
            dataset_dir : str | Path,
            env_raster_subdir : str,
            env_radiomap_subdir : str,
            shape_target : tuple[int,int],
            n_copies : int,
            use_tx_one_hot : bool,
            use_tx_distance : bool,
            use_Tx_distToRx : bool,
            use_log_distance : bool,
            use_fspl : bool,
            observation_percentage : float,
            use_material_properties : bool,
            use_material_classes : bool,
            rx_height : float,
            frequency_hz : float,
            use_augmentation : bool,
            use_observation_mask_as_input : bool,
            pl_max : float | None = None,
            pl_trnc : float | None = None,
            return_tx_id : bool = False,
            return_tx_coords : bool = False,
            **kwargs
        ) -> None:
        """
        Initializes the data loading class with configuration parameters for sample selection, 
        clustering, dataset paths, input features, and augmentation options.

        Args:
            sample_file (str): Path to the file containing measured samples in JSON format.
            percentage_measurements_threshold (float): Minimum percentage of measurements required to include a pixel.
            per_cluster_split (int): Number of parts to split the data in each cluster into.
            n_clusters_test (int): Number of clusters used for testing (others are input).
            phase (str): Phase of the dataset (e.g., 'train', 'val', 'test').
            dataset_dir (str | Path): Directory containing the dataset.
            env_raster_subdir (str): Subdirectory for environment raster data.
            env_radiomap_subdir (str): Subdirectory for environment radiomap data.
            shape_target (tuple[int, int]): Target shape for the data samples.
            n_copies (int): Number of copies to generate for each sample.
            use_tx_one_hot (bool): Whether to use transmitter one-hot encoding as input.
            use_tx_distance (bool): Whether to use transmitter distance as input.
            use_Tx_distToRx (bool): Whether to use transmitter-to-receiver distance as input.
            use_log_distance (bool): Whether to use logarithmic distance as input.
            use_fspl (bool): Whether to use free-space path loss as input.
            use_material_properties (bool): Load raster files of electromagnetic properties as input.
            use_material_classes (bool): Load raster files of classes as input.
            observation_percentage (float): observation_percentage used for training. For testing with measurements, it is
                only relevant whether this quantity is equal to or larger than 0. The number of given measurements is determined
                by the parameters percentage_measurements_threshold, per_clusters_split and n_clisters_test.
            rx_height (float): Receiver height in meters.
            frequency_hz (float): Frequency in Hertz.
            use_augmentation (bool): Whether to apply data augmentation.
            use_observation_mask_as_input (bool): Whether to use observation mask as input.
            pl_max (float | None, optional): Maximum path loss value for HDF5/measurement data. Defaults to None. 
            pl_trnc (float | None, optional): Truncated path loss value for HDF5/measurement data. Defaults to None.
            return_tx_id (bool): Return the Tx ID additionally to other data. This is useful in order to use this
                class together with the RT data. Defaults to False.
            # separate_clusters (bool): Use separate clusters as training/testing clusters
        Raises:
            FileNotFoundError: If the sample file does not exist.
            json.JSONDecodeError: If the sample file is not valid JSON.
        """
        with open(Path(dataset_dir) / sample_id_file, 'r') as f:
            self.measurements = json.load(f)
        self.measurements = {
            t : {
                m : {
                    p : p_dict for p, p_dict in m_dict.items() if p_dict['percentage_samples_available'] >= percentage_measurements_threshold
                } for m, m_dict in t_dict.items()
            } for t, t_dict in self.measurements.items()
        }
        for t, td in self.measurements.items():
            for m, md in td.items():
                if len(md) == 0:
                    print(f'for Tx {t} and measurement area {m}, there are not measurement points included!')


        if not len(list(self.measurements.values())[0]) > n_clusters_test:
            raise ValueError(f'{len(list(self.measurements.values())[0])=} <= {n_clusters_test=}')
        
        super().__init__(
            phase=phase,
            dataset_dir=dataset_dir,
            env_raster_subdir=env_raster_subdir,
            env_radiomap_subdir=env_radiomap_subdir,
            sample_id_file=None,
            shape_target=shape_target,
            n_copies=n_copies,
            use_tx_one_hot=use_tx_one_hot,
            use_tx_distance=use_tx_distance,
            use_Tx_distToRx=use_Tx_distToRx,
            use_log_distance=use_log_distance,
            use_fspl=use_fspl,
            observation_percentage=observation_percentage,
            use_material_properties=use_material_properties,
            use_material_classes=use_material_classes,
            rx_height=rx_height,
            frequency_hz=frequency_hz,
            use_augmentation=use_augmentation,
            use_observation_mask_as_input=use_observation_mask_as_input,
            pl_max=pl_max,
            pl_trnc=pl_trnc,
        )
        self.per_cluster_split = per_cluster_split
        self.n_clusters_test = n_clusters_test
        self.return_tx_id = return_tx_id
        self.return_tx_coords = return_tx_coords

        print(f'Created DatasetMeasurements with length {self.__len__()}')
        
    def __len__(self) -> int:
        return len(self.measurements) * comb(len(self.measurements[list(self.measurements.keys())[0]]), self.n_clusters_test) * self.per_cluster_split


    def __getitem__(self, idx: int) -> tuple[Tensor,...]:
        """
        Retrieves a data sample for the given index from measurement data.
        
        This method loads and processes input data, ground truth, and observation data from measured points 
        for a specific transmitter. It supports loading data from multiple copies of environments and applies 
        transformations.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            Tuple containing:
        - inputs (Tensor): Processed input tensor with shape (C, H, W).
        - gt (Tensor): Ground truth radio map constructed from measured points with shape (H, W).
        - gt_mask (Tensor): Boolean mask indicating GT measurement locations with shape (H, W).
        - observation_mask (Tensor): Mask showing observation measurement locations with shape (H, W).
        - tx_id (Tensor, optional): Transmitter ID as single-element tensor.
            Included if return_tx_id=True.
        - tx_coords (Tensor, optional): Transmitter coordinates [x, y, z].
            Included if return_tx_coords=True.
        Raises:
            RuntimeError: If `self.sample_ids` is None in the base class.
        """

        env_id = 'wi_files'
        
        if self.n_copies > 0:
            ### draw a random copy ID, which is used for loading the data
            copy_id = np.random.randint(0, self.n_copies)
            env_id_here = f'{env_id}_copy{copy_id}'
        else:
            ### use the raster images describing the original env used in the simulations
            env_id_here = env_id
        
        # Load environment inputs
        inputs = torch.empty((0, self.x_steps, self.y_steps))
        if self.use_material_properties or self.use_material_classes:
            inputs = load_env_inputs(
                inputs=inputs, 
                env_id_here=env_id_here, 
                env_raster_subdir=self.env_raster_subdir, 
                raster_heights=self.raster_heights, 
                n_classes=self.n_classes, 
                use_material_classes=self.use_material_classes, 
                use_material_properties=self.use_material_properties
            )
        
        # Load measured data points and add to inputs
        tx_id, inputs, gt, gt_mask, observation_mask = load_gt_observations_from_dict(
            use_observations=self.use_observations,
            idx=idx,
            measurements=self.measurements,
            n_clusters_test=self.n_clusters_test,
            per_cluster_split=self.per_cluster_split,
            inputs=inputs,
            shape_target=self.shape_target,
            pl_max=self.pl_max,
            pl_trnc=self.pl_trnc,
            use_observation_mask_as_input=self.use_observation_mask_as_input
        )

        # Load transmitter inputs
        inputs, tx_coords = load_tx_inputs(
            env_tx_subdir=self.env_raster_subdir,
            env_id_here=env_id_here,
            tx_id=tx_id,
            use_tx_one_hot=self.use_tx_one_hot,
            inputs=inputs,
            x_step_size=self.x_step_size,
            y_step_size=self.y_step_size,
            use_tx_distance=self.use_tx_distance,
            use_Tx_distToRx=self.use_Tx_distToRx,
            rx_height=self.rx_height,
            use_log_distance=self.use_log_distance,
            use_fspl=self.use_fspl,
            frequency_hz=self.frequency_hz,
            return_coords=True,
            pl_max=self.pl_max,
            pl_trnc=self.pl_trnc,
        ) # type: ignore
        
        return_vals = list(self.transforms(tv_image(inputs), tv_image(gt), tv_image(gt_mask), tv_image(observation_mask)) if observation_mask is not None else self.transforms(tv_image(inputs), tv_image(gt), tv_image(gt_mask)))
        if self.return_tx_id:
            return_vals.append(torch.tensor([tx_id]))
        if self.return_tx_coords:
            return_vals.append(torch.tensor(tx_coords))
        return tuple(return_vals)
        

def calculate_num_input(dataset : DatasetSketch) -> int:
    """Calculate the number of input channels based on dataset configuration"""
    # Base channels
    num_channels = 0
    if dataset.use_material_classes:
        num_channels += len(dataset.raster_heights)
    if dataset.use_material_properties:
        num_channels += 3 * len(dataset.raster_heights)
    
    # Add channels based on feature flags
    if getattr(dataset, 'use_tx_one_hot'):
        num_channels += 1
    if getattr(dataset, 'use_tx_distance'):
        num_channels += 1
    if getattr(dataset, 'use_Tx_distToRx'):
        num_channels += 1
    if getattr(dataset, 'use_log_distance'):
        num_channels += 1
    if getattr(dataset, 'use_fspl'):
        num_channels += 1
    op = getattr(dataset, 'observation_percentage')
    if (isinstance(op, float) and op > 0) or isinstance(op, Sequence):
        if getattr(dataset, 'use_observation_mask_as_input'):
            num_channels += 2
        else:
            num_channels += 1  # Sparse GT as input channel
    
    return num_channels

def visualize_dataset_samples(
        dataset : DatasetSketch, 
        output_dir : str | Path, 
        num_samples : int = 3
    ) -> None:
    """Visualize dataset samples and save to disk"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for sample_id in range(num_samples):
        data = dataset.__getitem__(sample_id)
        assert len(data) == 4
        inputs, gt, mask_shape, observation_mask = data
        
        
        n_plots = 1 + inputs.shape[0] + (1 if dataset.use_observations else 0)
        n_cols = 5
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(4*n_cols, 3*n_rows))
        plot_idx = 1
        
        # Ground truth
        fig.add_subplot(n_rows, n_cols, plot_idx)
        plt.imshow(torch.where(gt==0, torch.nan, gt).cpu().numpy())
        plt.colorbar()
        plt.title('Ground Truth')
        plot_idx += 1
        
        # Input channels
        channel_names = ['Env Rasters', 'Tx One-Hot', 'Tx Distance', '3D Distance', 'Log Distance', 'FSPL', 'Sparse GT']
        for k in range(inputs.shape[0]):
            fig.add_subplot(n_rows, n_cols, plot_idx)
            plt.imshow(inputs[k,...].cpu().numpy())
            plt.colorbar()
            title = channel_names[k] if k < len(channel_names) else f'Input {k}'
            plt.title(title)
            plot_idx += 1
        
        # observation mask
        if dataset.use_observations:
            fig.add_subplot(n_rows, n_cols, plot_idx)
            plt.imshow(observation_mask.cpu().numpy())
            plt.colorbar()
            plt.title('observation Mask')
        
        save_path = output_dir / f'sample_{sample_id}.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved sample {sample_id} to {save_path}')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Create dataset with default parameters from script_training_testing
    dataset = DatasetSketch(
        phase='train',
        dataset_dir='dataset/indoor_projects_filtered_250922',
        env_raster_subdir='rasterized_projects_256x256',
        env_radiomap_subdir='radio_maps_-12_-71',
        # env_radiomap_subdir='simulation_data',
        sample_id_file='split_rectangular_L-shaped_val=0.1_test=0.1.json',
        shape_target=(32, 32),
        n_copies=10,
        use_tx_one_hot=True,
        use_tx_distance=True,
        use_Tx_distToRx=False,
        use_log_distance=False,
        use_fspl=False,
        use_material_properties=False,
        use_material_classes=True,
        observation_percentage=1.0,
        rx_height=1.0,
        frequency_hz=5.82e9,
        use_augmentation=True,
        use_observation_mask_as_input=False,
        pl_max=-12,
        pl_trnc=-71,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get first element
    sample = dataset[0]
    
    if len(sample) == 4:  # With observation
        inputs, gt, gt_mask, observation_mask = sample
        print(f"Inputs shape: {inputs.shape}")
        print(f"GT shape: {gt.shape}")
        # print(f"GT original shape: {gt_mask}")
        print(f"observation mask shape: {observation_mask.shape}")
        
        # Plot results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot some input channels
        axes[0, 0].imshow(inputs[0], cmap='viridis')
        axes[0, 0].set_title('Input Channel 0 (Environment)')
        
        if inputs.shape[0] > 1:
            axes[0, 1].imshow(inputs[-2], cmap='viridis')
            axes[0, 1].set_title('Input Channel -2 (TX info)')
        
        if inputs.shape[0] > 2:
            axes[0, 2].imshow(inputs[-1], cmap='viridis')
            axes[0, 2].set_title('Input Channel -1 (Distance)')
        
        # Plot ground truth and observation
        axes[1, 0].imshow(gt.squeeze())#, cmap='plasma')
        axes[1, 0].set_title('Ground Truth')
        
        axes[1, 1].imshow(observation_mask.squeeze(), cmap='binary')
        axes[1, 1].set_title('observation Mask')
        
        
    else:  # Without observation
        inputs, gt, gt_shape = sample
        print(f"Inputs shape: {inputs.shape}")
        print(f"GT shape: {gt.shape}")
        print(f"GT original shape: {gt_shape}")
        
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot some input channels
        axes[0].imshow(inputs[0], cmap='viridis')
        axes[0].set_title('Input Channel 0 (Environment)')
        
        if inputs.shape[0] > 1:
            axes[1].imshow(inputs[-1], cmap='viridis')
            axes[1].set_title('Input Channel -1 (TX info)')
        
        # Plot ground truth
        axes[2].imshow(gt.squeeze(), cmap='plasma')
        axes[2].set_title('Ground Truth')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Number of input channels: {inputs.shape[0]}")
    print("Test completed successfully!")
