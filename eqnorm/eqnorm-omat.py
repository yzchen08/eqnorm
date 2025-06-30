import argparse
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from torch_scatter import scatter

from e3nn import o3
from e3nn.o3 import FullyConnectedTensorProduct, TensorProduct, Linear
from e3nn.nn import FullyConnectedNet, Gate

from .grad_output import EdgewiseGrad, NodewiseGrad
from .layernorm import RMSLayerNorm as EquivariantLayerNorm


def tp_path_exists(
        irreps_in1: Union[str, o3.Irreps], 
        irreps_in2: Union[str, o3.Irreps], 
        ir_out: Union[str, o3.Irreps], 
        ) -> bool:
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


def get_path(
        irreps_in1: o3.Irreps, 
        irreps_in2: o3.Irreps, 
        irreps_out: o3.Irreps
        ) -> Tuple[o3.Irreps, list[Tuple[int, int, int, str, bool]]]:
    irreps_mid = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps_in1):
        for j, (_, ir_sh) in enumerate(irreps_in2):
            for ir_out in ir_in * ir_sh:
                if ir_out in irreps_out:
                    k = len(irreps_mid)
                    irreps_mid.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", True))
    irreps_mid = o3.Irreps(irreps_mid)
    irreps_mid, p, _ = irreps_mid.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, p[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]
    return irreps_mid, instructions


class E3NN(torch.nn.Module):
    def __init__(
            self,
            irreps_hidden: Union[str, o3.Irreps],  # node hidden representation
            irreps_sh: Union[str, o3.Irreps],  # edge spherical harmonics
            num_conv_layers: int = 5,  # number of convolution layers
            num_types: int = 4,  # number of node atom types
            num_features: int = 128,  # number of features for node embedding
            max_radius: int = 6.0,  # cutoff radius
            num_basis: int = 8,   # number of Bessel basis functions
            invariant_layers: int = 2,  # number of radial layers
            invariant_neurons: int = 64,  # number of hidden neurons in radial function
            poly_p: int = 6,  # polynomial cutoff
            nonlinearity_type: str = "gate",  # nonlinearity type for invariant and equivariant layers
            avg_num_neighbors: Optional[float] = None,  # average number of neighbors
            ) -> None:
        super().__init__()
        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps(irreps_sh)

        self.num_types = num_types
        self.num_features = num_features
        self.scalar_features = o3.Irreps(f"{self.num_features}x0e")

        self.max_radius = max_radius
        self.num_basis = num_basis
        self.invariant_layers = invariant_layers
        self.invariant_neurons = invariant_neurons
        self.poly_p = poly_p

        self.embedding = Embedding_layer(num_types=self.num_types, num_features=self.num_features)

        self.sph = o3.SphericalHarmonics(self.irreps_sh, normalize=True, normalization='component')

        self.besssel_basis = BesselBasis(r_max=self.max_radius, num_basis=self.num_basis)
        self.poly_cutoff = PolynomialCutoff(r_max=self.max_radius, p=self.poly_p)

        self.nonlinearity_scalars = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }
        self.nonlinearity_gates = {
            1: torch.nn.functional.silu,
            -1: torch.tanh,
        }

        self.num_conv_layers = num_conv_layers
        self.layers = torch.nn.ModuleList()
        self.irreps_in = self.scalar_features
        for num_conv_layer in range(self.num_conv_layers):
            irreps_scalars = o3.Irreps()
            irreps_gate_scalars = o3.Irreps()
            irreps_nonscalars = o3.Irreps()

            # get scalar target irreps
            for multiplicity, irrep in self.irreps_hidden:
                if o3.Irrep(irrep).l == 0 and tp_path_exists(self.irreps_in, self.irreps_sh, irrep):
                    irreps_scalars += [(multiplicity, irrep)]
                irreps_scalars = o3.Irreps(irreps_scalars)

            # get non-scalar target irreps
            for multiplicity, irrep in self.irreps_hidden:
                if o3.Irrep(irrep).l > 0 and tp_path_exists(
                    self.irreps_in, self.irreps_sh, irrep
                ):
                    irreps_nonscalars += [(multiplicity, irrep)]
                irreps_nonscalars = o3.Irreps(irreps_nonscalars)

            # get gate scalar irreps
            if tp_path_exists(self.irreps_in, self.irreps_sh, '0e'):
                gate_scalar_irreps_type = '0e'
            else:
                gate_scalar_irreps_type = '0o'

            for multiplicity, _ in irreps_nonscalars:
                irreps_gate_scalars += [(multiplicity, gate_scalar_irreps_type)]
            irreps_gate_scalars = o3.Irreps(irreps_gate_scalars).simplify()

            # final layer output irreps are all three
            self.irreps_out = irreps_scalars + irreps_gate_scalars + irreps_nonscalars
            self.irreps_out = self.irreps_out.sort().irreps.simplify()

            self.layers.append(E3Conv(
                self.irreps_in, 
                self.irreps_sh, 
                self.irreps_out, 
                num_basis=self.num_basis,
                invariant_layers=self.invariant_layers,
                invariant_neurons=self.invariant_neurons,
                avg_num_neighbors=avg_num_neighbors,
                mode="nonscalar",
                ))
            self.irreps_in = irreps_scalars + irreps_nonscalars

            if nonlinearity_type == "gate":
                self.layers.append(EquivariantGate(
                    irreps_scalars=irreps_scalars,
                    act_scalars=[self.nonlinearity_scalars[ir.p] for _, ir in irreps_scalars],
                    irreps_gates=irreps_gate_scalars,
                    act_gates=[self.nonlinearity_gates[ir.p] for _, ir in irreps_gate_scalars],
                    irreps_gated=irreps_nonscalars,
                ))
            else:
                raise NotImplementedError(f"nonlinearity_type {nonlinearity_type} not implemented.")

            self.layers.append(E3Conv(
                self.irreps_in, 
                self.irreps_sh, 
                self.scalar_features, 
                num_basis=self.num_basis,
                invariant_layers=self.invariant_layers,
                invariant_neurons=self.invariant_neurons,
                avg_num_neighbors=avg_num_neighbors,
                mode="scalar",
                ))
            
            self.layers.append(NonlinearAndAdd())
           
        self.output_block = FullyConnectedNet(
            [self.num_features]
            + [self.num_features // 2]
            + [1],
            self.nonlinearity_scalars[1],
        )

    def forward(
            self, 
            data: dict[str, torch.Tensor],
            ) -> torch.Tensor:
        data['node_hiddens'] = self.embedding(data['atomic_numbers'])
        data['output'] = data['node_hiddens']
        
        distance = torch.norm(data['edge_vec'], p=2, dim=-1)
        # data['edge_sh'] = o3.spherical_harmonics(self.irreps_sh, data['edge_vec'], normalize=True, normalization='component')
        data['edge_sh'] = self.sph(data['edge_vec'])

        data['edge_attr'] = self.besssel_basis(distance)
        cutoff = self.poly_cutoff(distance).unsqueeze(-1)
        data['edge_attr'] = data['edge_attr'] * cutoff

        # print("otuput", -1, data['output'][:, :128].pow(2).mean(dim=-1, keepdim=True).pow(0.5).mean())
        for layer_idx, layer in enumerate(self.layers):
            layer(data)
            # print("otuput", layer_idx, data['output'][:, :128].pow(2).mean(dim=-1, keepdim=True).pow(0.5).mean())
            # print("hidden", layer_idx, data['node_hiddens'][:, :128].pow(2).mean(dim=-1, keepdim=True).pow(0.5).mean())
            # print("hidden", layer_idx, data['node_hiddens'][:, 128:].pow(2).mean(dim=-1, keepdim=True).pow(0.5).mean())

        data['output'] = self.output_block(data['output'])
        # print("otuput", 100, data['output'][:, :128].pow(2).mean(dim=-1, keepdim=True).pow(0.5).mean())

        return data['output']


class EquivariantGate(torch.nn.Module):
    def __init__(
            self, 
            irreps_scalars: Union[str, o3.Irreps],
            act_scalars: List[Callable],
            irreps_gates: Union[str, o3.Irreps],
            act_gates: List[Callable],
            irreps_gated: Union[str, o3.Irreps],
            ) -> None:
        super().__init__()
        self.gate = Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=act_scalars,
            irreps_gates=irreps_gates,
            act_gates=act_gates,
            irreps_gated=irreps_gated,
        )

    def forward(
            self, 
            data: dict[str, torch.Tensor], 
            ) -> None:
        data['node_hiddens'] = self.gate(data['node_hiddens'])
        return None


class NonlinearAndAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nonlinear = torch.nn.SiLU()

    def forward(
            self, 
            data: dict[str, torch.Tensor], 
            ) -> None:
        data['output'] = data['output'] + self.nonlinear(data['node_scalars'])
        return None


class Embedding_layer(torch.nn.Module):
    def __init__(
            self, 
            num_types: int, 
            num_features: int, 
            ) -> None:
        super().__init__()
        self.num_types = num_types
        self.num_features = num_features
        self.linear = Linear(o3.Irreps(f"{self.num_types}x0e"), o3.Irreps(f"{self.num_features}x0e"))
    
    def forward(
            self, 
            atomic_numbers: torch.Tensor, 
            ) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(atomic_numbers, num_classes=self.num_types).float()
        return self.linear(one_hot)


class BesselBasis(torch.nn.Module):
    def __init__(
            self, 
            r_max: float, 
            num_basis: int = 8, 
            trainable: bool = True,
            ) -> None:
        super().__init__()

        self.trainable = trainable
        self.num_basis = num_basis
        self.r_max = float(r_max)
        self.prefactor = 2.0 / self.r_max

        bessel_weights = torch.linspace(start=1.0, end=num_basis, steps=num_basis) * torch.math.pi
        if self.trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

    def forward(
            self, 
            x: torch.Tensor
            ) -> torch.Tensor:
        numerator = torch.sin(self.bessel_weights * x.unsqueeze(-1) / self.r_max)
        return self.prefactor * (numerator / x.unsqueeze(-1))


class PolynomialCutoff(torch.nn.Module):
    def __init__(
            self, 
            r_max: float, 
            p: float = 6,
            ) -> None:
        super().__init__()
        assert p >= 2.0
        self.p = float(p)
        self._factor = 1.0 / float(r_max)

    def poly_cutoff(
            self, 
            x: torch.Tensor, 
            factor: float, 
            p: float = 6.0
            ) -> torch.Tensor:
        x = x * factor
        out = 1.0
        out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p))
        out = out + (p * (p + 2.0) * torch.pow(x, p + 1.0))
        out = out - ((p * (p + 1.0) / 2) * torch.pow(x, p + 2.0))
        return out * (x < 1.0)

    def forward(self, x):
        return self.poly_cutoff(x, self._factor, p=self.p)


class E3Conv(torch.nn.Module):
    def __init__(
            self,
            irreps_hidden: Union[str, o3.Irreps],
            irreps_sh: Union[str, o3.Irreps],
            irreps_out: Union[str, o3.Irreps],
            num_basis: int = 8,
            invariant_layers: int = 2,
            invariant_neurons: int = 64,
            avg_num_neighbors: Optional[float] = None,
            use_sc: bool = True,
            nonlinearity_scalars: Dict[str, Callable] = {"e": torch.nn.functional.silu}, 
            mode: Literal["nonscalar", "scalar"] = "nonscalar",
            ) -> None:
        super().__init__()

        self.irreps_hidden = o3.Irreps(irreps_hidden)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.irreps_out = o3.Irreps(irreps_out)

        self.avg_num_neighbors = avg_num_neighbors
        self.use_sc = use_sc
        self.mode = mode

        self.linear_1 = Linear(
            irreps_in=self.irreps_hidden,
            irreps_out=self.irreps_hidden,
        )

        irreps_mid, instructions = get_path(
            self.irreps_hidden, 
            self.irreps_sh, 
            self.irreps_out
            )
        self.tp = TensorProduct(
            self.irreps_hidden,
            self.irreps_sh,
            irreps_mid,
            instructions,
            internal_weights=False,
            shared_weights=False,
        )

        self.fc = FullyConnectedNet(
            [num_basis]
            + invariant_layers * [invariant_neurons]
            + [self.tp.weight_numel],
            nonlinearity_scalars['e'],
        )

        self.linear_2 = Linear(
            irreps_in=irreps_mid.simplify(),
            irreps_out=self.irreps_out,
        )

        self.sc = None
        if self.use_sc:
            self.sc = Linear(
                self.irreps_hidden,
                self.irreps_out,
            )

        self.ln = EquivariantLayerNorm(self.irreps_out, centering=False)

    def forward(
            self,
            data: dict[str, torch.Tensor],
            ) -> None:
        weight = self.fc(data['edge_attr'])
        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]

        if self.sc is not None:
            sc = self.sc(data['node_hiddens'])

        node_hiddens = self.linear_1(data['node_hiddens'])

        edge_features = self.tp(
            node_hiddens[edge_dst], data['edge_sh'], weight
        )

        # node_hiddens = torch.zeros(len(node_hiddens), edge_features.shape[-1], device=edge_features.device, dtype=edge_features.dtype)
        # node_hiddens.index_add_(0, edge_src, edge_features)
        node_hiddens = scatter(edge_features, edge_src, dim=0, dim_size=len(node_hiddens), reduce='sum')
        if self.avg_num_neighbors is not None:
            node_hiddens = node_hiddens.div(self.avg_num_neighbors ** 0.5)
        else:
            avg_num_neighbors = edge_src.bincount(minlength=len(node_hiddens)).unsqueeze(-1)
            avg_num_neighbors[avg_num_neighbors == 0] = 1  # avoid division by zero
            node_hiddens = node_hiddens.div(avg_num_neighbors.sqrt())

        node_hiddens = self.linear_2(node_hiddens)

        if self.sc is not None:
            node_hiddens = node_hiddens + sc
            node_hiddens = self.ln(node_hiddens)

        if self.mode == "nonscalar":
            data['node_hiddens'] = node_hiddens
        elif self.mode == "scalar":
            data['node_scalars'] = node_hiddens
        else:
            raise NotImplementedError(f"mode {self.mode} not implemented.")
        
        return None


class HDNNP(torch.nn.Module):
    def __init__(
            self, 
            args: argparse.Namespace, 
            unique_elements: torch.Tensor, 
            shift: Optional[torch.Tensor] = None, 
            scale: Optional[torch.Tensor] = None, 
            ) -> None:
        super().__init__()
        self.r_cutoff = args.r_cutoff  # A
        self.unique_elements = unique_elements
        self.num_types = len(self.unique_elements)
        self.shift = shift
        self.scale = scale
        self.grad_mode = args.grad_mode

        if args.shift_trainable and self.shift is not None:
            self.shift = torch.nn.Parameter(self.shift)
        if args.scale_trainable and self.scale is not None:
            self.scale = torch.nn.Parameter(self.scale)

        self.calc_stress = args.STRESS
        self.calc_dipole = args.DIPOLE
        self.calc_polar = args.POLAR

        self.e3nn_layer = E3NN(
            irreps_hidden=args.irreps_hidden, 
            irreps_sh=args.irreps_sh,
            num_conv_layers=args.num_convs, 
            num_types=self.num_types, 
            num_features=args.num_features,
            max_radius=self.r_cutoff, 
            num_basis=args.num_basis, 
            invariant_layers=args.invariant_layers,
            invariant_neurons=args.invariant_neurons,
            poly_p=args.poly_p,
            avg_num_neighbors=args.avg_nbr,
            )
        
        if self.grad_mode == 'edge':
            self.force_stress_output = EdgewiseGrad(calc_stress=self.calc_stress)
        elif self.grad_mode == 'node':
            self.force_stress_output = NodewiseGrad(calc_stress=self.calc_stress)
        else:
            raise NotImplementedError(f"grad_mode {self.grad_mode} not implemented.")

    def forward(
            self, 
            data: dict[str, torch.Tensor], 
            training: bool = True,  # determine whether to create graph for autograd
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], None, None]:
        data['pos'] = data['pos'].requires_grad_(True)
        # map atomic numbers to {0, 1, 2, ...} according to unique_elements
        data['atomic_numbers'] = torch.searchsorted(self.unique_elements, data['atomic_numbers'])

        if self.calc_stress and self.grad_mode == 'node':
            data['displacement'] = torch.zeros((3, 3), dtype=data['pos'].dtype, device=data['pos'].device)
            data['displacement'] = data['displacement'].view(-1, 3, 3).expand(data['batch'][-1] + 1, 3, 3)  # (N_batch, 3, 3)
            data['displacement'] = data['displacement'].requires_grad_(True)
            data['symmetric_displacement'] = 0.5 * (data['displacement'] + data['displacement'].transpose(-1, -2))
            data['pos'] = data['pos'] + torch.bmm(
                data['pos'].unsqueeze(-2), data['symmetric_displacement'][data['batch']]
            ).squeeze(-2)
            data['cell'] = data['cell'] + torch.bmm(data['cell'], data['symmetric_displacement'])

        # calc edge vectors depending on periodicity
        if 'shifts' in data and 'cell' in data:
            pbc_shift = torch.einsum("ni,nij->nj", data['shifts'].float(), data['cell'][data['batch']][data['edge_index'][0]])
            data['edge_vec'] = data['pos'][data['edge_index'][1]] - data['pos'][data['edge_index'][0]] + pbc_shift
        else:
            data['edge_vec'] = data['pos'][data['edge_index'][1]] - data['pos'][data['edge_index'][0]]

        energy = self.e3nn_layer(data)
        if self.scale is not None:
            if self.scale.dim() == 0:
                energy = energy * self.scale
            else:
                energy = energy * self.scale[data['atomic_numbers']]
        if self.shift is not None:
            if self.shift.dim() == 0:
                energy = energy + self.shift
            else:
                energy = energy + self.shift[data['atomic_numbers']]
        # reduced_energy = torch.zeros(data['batch'][-1] + 1, 1, device=energy.device, dtype=energy.dtype)
        # reduced_energy.index_add_(0, data['batch'], energy)
        # energy = reduced_energy
        energy = scatter(energy, data['batch'], dim=0, dim_size=data['batch'][-1] + 1)

        forces, stress = self.force_stress_output(energy, data, training)

        if self.calc_dipole:
            dipole = None
        else:
            dipole = None
        
        if self.calc_polar:
            polar = None
        else:
            polar = None

        return energy, forces, stress, dipole, polar


