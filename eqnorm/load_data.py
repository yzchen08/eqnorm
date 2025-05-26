from functools import wraps
import os
import random
import time
from typing import List, Union
from ase import Atom
import numpy as np
import torch
from tqdm import tqdm


def time_count(func):
    """
    Decorator to count the time consumed by a function. 
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        print(
            f"{func.__name__} time consume: {round(time.perf_counter() - start, 4)} s."
        )
        return res
    return wrapper


@time_count
def get_data(
        data_path: str,  # data path recursively containing all pt files
        start: int = 0,  # start index of selection
        end: Union[int, None] = None,  # end index of selection
        ) -> List:
    """
    Get pt data from data_path recursively, which can be a directory or a file.
    """
    file_paths = []
    
    def recursive_collect(path):
        names = os.listdir(path)
        for name in names:
            full_path = os.path.join(path, name)
            if os.path.isdir(full_path):
                recursive_collect(full_path)
            else:
                file_paths.append(full_path)

    if os.path.isdir(data_path):
        recursive_collect(data_path)
    else:
        file_paths.append(data_path)

    results = []
    for filename in tqdm(file_paths[start: end]):
        results.append(torch.load(filename))
    
    return results


def load_data(args, logger):
    # dataset path
    train_dir = args.train_dir
    val_dir = args.val_dir
    logger.info(f"loading dataset from {train_dir}, {val_dir} ...")

    import sys
    sys.path.append(args.data_loader_path)
    from materials_models.data import BlendedGraphDatasetConfig, BlendedGraphDataset, find_blend_files # type: ignore

    # load train dataset
    if '.idx' in train_dir:
        blend_config = BlendedGraphDatasetConfig(
            datasets=find_blend_files(train_dir),
            random_seed=args.dataset_seed,
            shuffle=False,
            mmap_read=False,
            )
        samples_train = BlendedGraphDataset(blend_config)
        logger.info(f"total samples train: {len(samples_train)} ...")
    else:
        # a list of sublist, each sublist is a list of Data and loaded from one pt file
        samples_train = get_data(train_dir, start=0, end=args.num_train)

        # flatten samples_train and shuffle
        samples_train = [ii for sublist in samples_train for ii in sublist]
        random.seed(args.dataset_seed)
        random.shuffle(samples_train)

    # load val or test dataset
    if '.idx' in val_dir[0]:
        samples_val_loader = []
        for val_subdir in val_dir:
            blend_config = BlendedGraphDatasetConfig(
                datasets=find_blend_files(val_subdir),
                random_seed=args.dataset_seed,
                shuffle=False,
                mmap_read=False,
                )
            samples_val_loader.append(BlendedGraphDataset(blend_config))
        if not args.TEST:
            samples_val_loader = samples_val_loader[0][np.linspace(0, len(samples_val_loader[0]) - 1, 128).astype(int)]
            samples_val = [[ii for ii in samples_val_loader]]
            del samples_val_loader
        else:
            # samples_val = samples_val_loader
            samples_val_loader = samples_val_loader[0][np.linspace(0, len(samples_val_loader[0]) - 1, 5000).astype(int)]
            samples_val = [[ii for ii in samples_val_loader]]
            del samples_val_loader
    else:
        # a list of sublist, each sublist is a list of Data and loaded from one pt file
        samples_val = get_data(val_dir[0], start=0, end=args.num_val)
        if not args.TEST:
            samples_val = [samples_val[0][:128]]
    
    # samples_train is a dataloader or list of Data, samples_val is a list of dataloaders or Data sublists
    logger.info(f"total samples train: {len(samples_train)} ...")
    logger.info(f"selected systems val: {len(samples_val)} ...")
    logger.info(f"total samples val: {sum([len(ii) for ii in samples_val])} ...")

    # calculate or load energy shift and scale
    if args.TRAIN:
        try:
            unique_elements, energy_shift, energy_scale = load_stat_from_file(logger, args)
        except FileNotFoundError as e:
            # get unique elements, shift and scale from scratch
            logger.info(e)
            logger.info("calculate stat from scratch")
            species_total = torch.zeros(len(samples_train), 100).long()
            energy_total = torch.zeros(len(samples_train), 1)
            atomic_number_total = torch.zeros(len(samples_train), 1)
            force_total = torch.zeros(len(samples_train) * int(args.avg_atoms * 1.2), 3)
            force_idx = 0
            for ii, dd in enumerate(tqdm(samples_train, desc="stat dataset")):
                species_total[ii] = torch.bincount(dd.atomic_numbers, minlength=100)
                energy_total[ii][0] = dd.energy.item()
                atomic_number_total[ii][0] = len(dd.atomic_numbers)
                force_total[force_idx: force_idx + len(dd.forces)] = dd.forces
                force_idx += len(dd.forces)      
            force_total = force_total[:force_idx]
            per_atom_energy_total = energy_total / atomic_number_total
            unique_elements = torch.nonzero(species_total.sum(dim=0)).squeeze()  # the unique elements in dataset
            species_total = species_total[:, unique_elements]  # mapped atomic numbers

            # calculate energy shift and scale
            per_atom_mean, per_atom_std = per_atom_energy_total.mean(), per_atom_energy_total.std(unbiased=True)
            force_rms_std = (force_total.mean() ** 2 + force_total.std() ** 2) ** 0.5

            # per_species_mean = torch.linalg.lstsq(species_total.float(), energy_total).solution
            from sklearn.linear_model import Ridge
            per_species_mean = Ridge(alpha=0.1, fit_intercept=False).fit(species_total, energy_total).coef_
            per_species_mean = torch.tensor(per_species_mean).float().view(-1, 1)

            # from GP_solver import solver
            # per_species_mean, per_species_std = solver(species_total, energy_total)

            stat = {
                'per_atom_mean': per_atom_mean, 
                'per_atom_std': per_atom_std, 
                'per_species_mean': per_species_mean, 
                # 'per_species_std': per_species_std, 
                'force_rms_std': force_rms_std, 
                'unique': unique_elements
                }
            torch.save(stat, args.data_stat)
            logger.info(f"save stat: {args.data_stat}, please run this script again.")
            exit()
    else:
        unique_elements, energy_shift, energy_scale = load_stat_from_ckpt(logger, args)
        # unique_elements, energy_shift, energy_scale = load_stat_from_file(logger, args)
    
    return samples_train, samples_val, unique_elements, energy_shift, energy_scale


def load_stat_from_file(logger, args):
    logger.info(f"loading stat from {args.data_stat}")
    stat = torch.load(args.data_stat)
    logger.info(f"loaded stat: {stat.keys()}")
    unique_elements = stat['unique']
    logger.info(f"loaded unique elements: {unique_elements}")
    if args.shift is not None:
        energy_shift = stat[f"{args.shift}_mean"]
        if energy_shift.ndim > 0:
            map_mean = {Atom(i.item()).symbol: round(j.item(), 4) for i, j in zip(unique_elements, energy_shift)}
            logger.info(f"loaded energy shift: {map_mean}")
        else:
            logger.info(f"loaded energy shift: {energy_shift.item()}")
    else:
        energy_shift = None
    if args.scale is not None:
        energy_scale = stat[f"{args.scale}_std"]
        if energy_scale.ndim > 0:
            map_std = {Atom(i.item()).symbol: round(j.item(), 4) for i, j in zip(unique_elements, energy_scale)}
            logger.info(f"loaded energy scale: {map_std}")
        else:
            logger.info(f"loaded energy scale: {energy_scale.item()}")
    else:
        energy_scale = None
    return unique_elements, energy_shift, energy_scale


def load_stat_from_ckpt(logger, args):
    start_load = time.perf_counter()
    checkpoint = torch.load(args.load_name)
    logger.info(f"loading checkpoint from {args.load_name}, time: {(time.perf_counter() - start_load):.4f}")
    unique_elements = checkpoint['unique_elements']
    logger.info(f"loaded unique elements: {unique_elements}")

    # load energy shift and scale from checkpoint
    energy_shift, energy_scale = checkpoint['energy_shift'], checkpoint['energy_scale']
    if energy_shift is not None:
        if energy_shift.ndim > 0:
            map_mean = {Atom(i.item()).symbol: round(j.item(), 4) for i, j in zip(unique_elements, energy_shift)}
            logger.info(f"loaded energy shift: {map_mean}")
        else:
            logger.info(f"loaded energy shift: {energy_shift.item()}")
    if energy_scale is not None:
        if energy_scale.ndim > 0:
            map_std = {Atom(i.item()).symbol: round(j.item(), 4) for i, j in zip(unique_elements, energy_scale)}
            logger.info(f"loaded energy scale: {map_std}")
        else:
            logger.info(f"loaded energy scale: {energy_scale.item()}")
    return unique_elements, energy_shift, energy_scale
