import time
import wget
import importlib
import importlib.resources
import os
import torch
import numpy as np

from ase.calculators.calculator import Calculator, all_changes
from ase import Atom, Atoms, units
from torch_geometric.data import Data, Batch

import vesin

from .config import LoadConfig


class EqnormCalculator(Calculator):
    """
    Supported properties: 
    ['energy', 'free_energy', 'forces', 'stress']
    """
    implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

    url_dict = {
        'eqnorm': {
            'eqnorm-mptrj': {
                "0.9": "https://figshare.com/files/54821513", 
                "1.0": "https://figshare.com/files/54851735",
                },
            'eqnorm-pro-mptrj': {
                "0.3": "https://figshare.com/files/54876488", 
                "0.6": "https://figshare.com/files/55065911", 
                },
            'eqnorm-max-mptrj': {
                "0.3": "https://figshare.com/files/55065917", 
                },
            }
        }

    def __init__(self, 
                 model_name: str,
                 model_variant: str,
                 train_progress: str="1.0",
                 device: str="cuda",
                 compile: bool=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        self.model_name = model_name
        self.model_variant = model_variant
        self.train_progress = str(train_progress)
        self.device = torch.device(device)
        self.compile = compile
        if not torch.cuda.is_available() and self.device.type == "cuda":
            raise ValueError("CUDA is not available, switching device to cpu.")

        # 检查 model_name 是否在 url_dict 中
        if self.model_name not in self.url_dict:
            valid_model_names = list(self.url_dict.keys())
            raise KeyError(
                f"Model name '{self.model_name}' not found in url_dict. "
                f"Valid model names are: {valid_model_names}"
            )

        # 检查 model_variant 是否在 url_dict[self.model_name] 中
        if self.model_variant not in self.url_dict[self.model_name]:
            valid_model_variants = list(self.url_dict[self.model_name].keys())
            raise KeyError(
                f"Model variant '{self.model_variant}' not found under model name '{self.model_name}' in url_dict. "
                f"Valid model variants are: {valid_model_variants}"
            )

        # 检查 train_progress 是否在 url_dict[self.model_name][self.model_variant] 中
        if self.train_progress not in self.url_dict[self.model_name][self.model_variant]:
            valid_train_progress = list(self.url_dict[self.model_name][self.model_variant].keys())
            raise KeyError(
                f"Train progress '{self.train_progress}' not found under model variant '{self.model_variant}' in url_dict. "
                f"Valid train progress values are: {valid_train_progress}"
            )

        module = importlib.import_module(f"{self.model_name}.{self.model_variant}")
        HDNNP = getattr(module, "HDNNP")

        self.model_args = str(importlib.resources.files(f"{self.model_name}.model_settings") / f"{self.model_variant}.yaml")
        print(f"load model args file from {self.model_args}")
        self.model_args = LoadConfig(yaml_path=self.model_args).load_args()

        self.r_cutoff = self.model_args.r_cutoff

        os.makedirs(os.path.expanduser(f"~/.cache/{self.model_name}"), exist_ok=True)
        self.ckpt_file = os.path.expanduser(f"~/.cache/{self.model_name}/{self.model_variant}-{self.train_progress}.pt")
        if os.path.exists(self.ckpt_file):
            print(f"File {self.ckpt_file} already exists, skipping download.")
        else:
            url = self.url_dict[self.model_name][self.model_variant][self.train_progress]
            print(f"File {self.ckpt_file} not exists, downloading from {url}...")
            try:
                wget.download(url, self.ckpt_file)
                print(f"File downloaded successfully and saved as {self.ckpt_file}")
            except Exception as e:
                print(f"Error downloading file: {e}")
                print(f"you can manually download the file from {url} and save it to {self.ckpt_file}")
                raise RuntimeError(f"Failed to download file from {url}")

        start_load = time.perf_counter()
        checkpoint = torch.load(self.ckpt_file, map_location=self.device, weights_only=False)
        print(f"load ckpt from {self.ckpt_file}, time: {(time.perf_counter() - start_load):.4f}")

        unique_elements = checkpoint['unique_elements']
        # print(f"loaded resume or test unique elements: {unique_elements}")

        # load energy shift and scale from checkpoint
        energy_shift, energy_scale = checkpoint['energy_shift'], checkpoint['energy_scale']

        self.model = HDNNP(
            args=self.model_args, 
            unique_elements=unique_elements.to(self.device), 
            shift=energy_shift.to(self.device), 
            scale=energy_scale.to(self.device), 
            )
        self.model = self.model.to(self.device)

        if self.model_args.use_ema:
            self.model.load_state_dict(checkpoint['ema_model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"total parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        for param in self.model.parameters():
            param.requires_grad = False

        if self.compile:
            torch.set_float32_matmul_precision('high')
            self.model = torch.compile(self.model, mode='default')


    def calculate(self, atoms: Atoms = None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties

        super().calculate(atoms, properties, system_changes)

        calculator = vesin.NeighborList(cutoff=self.r_cutoff, full_list=True, sorted=False)
        idx_i, idx_j, shifts = calculator.compute(
            points=atoms.positions, 
            box=atoms.cell.array, 
            periodic=True, 
            quantities="ijS"
            )
        idx_i, idx_j = idx_i.astype(np.int64), idx_j.astype(np.int64)
        
        idx_i, idx_j, shifts = torch.tensor(idx_i).long(), torch.tensor(idx_j).long(), torch.tensor(shifts).long()

        data = Data(
            x=None, y=None, pos=torch.tensor(self.atoms.positions).float(),  # A
            edge_index=torch.vstack([idx_i, idx_j]).long(),
            atomic_numbers=torch.tensor(self.atoms.get_atomic_numbers()).long(),
            shifts=shifts,  # for pbc, D = pos_j - pos_i + shifts @ cell or D = pos_i - pos_j - shifts @ cell
            cell=torch.tensor(self.atoms.cell.array.reshape(-1, 3, 3)).float(),  # for pbc, A
        )

        self.model.eval()
        data = Batch.from_data_list([data]).to(self.device)
        energy, forces, stress, _, _ = self.model(data.to_dict(), training=False)

        # energy = energy * 0.0433641153087705  # kcal/mol to eV
        # force = force * 0.0433641153087705  # kcal/mol/A to eV/A
        self.results['energy'] = energy.item()
        self.results['free_energy'] = energy.item()
        self.results['forces'] = forces.cpu().detach().numpy()
        if self.model_args.STRESS:
            # self.results['stress'] = stress[0].cpu().detach().numpy().flatten()[[0, 4, 8, 5, 2, 1]]
            self.results['stress'] = -stress[0].cpu().detach().numpy()[[0, 1, 2, 4, 5, 3]]
        else:
            self.results['stress'] = np.zeros(6).astype(np.float32)


