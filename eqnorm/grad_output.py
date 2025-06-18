from typing import Optional
import torch
from torch_scatter import scatter


class NodewiseGrad(torch.nn.Module):
    def __init__(self, calc_stress: bool = False):
        super().__init__()
        self.calc_stress = calc_stress

    def forward(
        self,
        energy: torch.Tensor, 
        data: dict[str, torch.Tensor], 
        training: bool, 
        ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        forces = torch.jit.annotate(Optional[torch.Tensor], None)
        stress = torch.jit.annotate(Optional[torch.Tensor], None)
        if self.calc_stress:
            grad = torch.autograd.grad(
                [energy.sum()],
                [data['pos'], data['displacement']],
                create_graph=training,
                retain_graph=training,
                )
            forces = grad[0]
            if forces is not None:
                forces = torch.neg(forces)
            stress = grad[1]
            if stress is not None:
                volume = torch.linalg.det(data['cell']).abs().unsqueeze(-1)
                stress = stress / volume.view(len(data['cell']), 1, 1)
                stress = stress.flatten(1, 2)[:, [0, 4, 8, 5, 2, 1]]  # voigt notation
        else:
            grad = torch.autograd.grad(
                [energy.sum()],
                [data['pos']],
                create_graph=training,
                retain_graph=training,
                )
            forces = grad[0]
            if forces is not None:
                forces = torch.neg(forces)

        return forces, stress


class EdgewiseGrad(torch.nn.Module):
    """
    https://github.com/MDIL-SNU/SevenNet/blob/main/sevenn/nn/force_output.py
    """
    def __init__(self, calc_stress: bool = False) -> None:
        super().__init__()
        self.calc_stress = calc_stress

    def forward(
        self,
        energy: torch.Tensor,
        data: dict[str, torch.Tensor],
        training: bool,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        grad = torch.autograd.grad(
            [energy.sum()],
            [data['edge_vec']],
            create_graph=training,
            retain_graph=training,
        )
        fij = grad[0]

        forces = torch.jit.annotate(Optional[torch.Tensor], None)
        stress = torch.jit.annotate(Optional[torch.Tensor], None)
        if fij is not None:
            # pf = torch.zeros(len(data['pos']), 3, dtype=fij.dtype, device=fij.device)
            # nf = torch.zeros(len(data['pos']), 3, dtype=fij.dtype, device=fij.device)
            # pf.index_add_(0, data['edge_index'][0], fij)
            # nf.index_add_(0, data['edge_index'][1], fij)
            pf = scatter(fij, data['edge_index'][0], dim=0, dim_size=len(data['pos']))
            nf = scatter(fij, data['edge_index'][1], dim=0, dim_size=len(data['pos']))
            forces = pf - nf

            # compute stress
            if self.calc_stress:
                diag = data['edge_vec'] * fij
                s12 = data['edge_vec'][..., 0] * fij[..., 1]
                s23 = data['edge_vec'][..., 1] * fij[..., 2]
                s31 = data['edge_vec'][..., 2] * fij[..., 0]
                # cat last dimension
                _virial = torch.cat([
                    diag,
                    s23.unsqueeze(-1),
                    s31.unsqueeze(-1),
                    s12.unsqueeze(-1),
                ], dim=-1)  # voigt notation

                # _s = torch.zeros(len(data['pos']), 6, dtype=fij.dtype, device=fij.device)
                # _s.index_add_(0, data['edge_index'][1], _virial)
                _s = scatter(_virial, data['edge_index'][1], dim=0, dim_size=len(data['pos']))

                # sout = torch.zeros(data['batch'][-1] + 1, 6, dtype=_virial.dtype, device=_virial.device)
                # sout.index_add_(0, data['batch'], _s)
                sout = scatter(_s, data['batch'], dim=0, dim_size=data['batch'][-1] + 1)

                volume = torch.linalg.det(data['cell']).abs().unsqueeze(-1)
                stress = sout / volume

        return forces, stress

