from typing import List, Optional
import torch
import torch.nn as nn
from e3nn.o3 import Irreps
from e3nn.util.jit import compile, compile_mode


def get_l_to_all_m_expand_index(lmax: int):
    expand_index = torch.zeros([(lmax + 1) ** 2]).long()
    for lval in range(lmax + 1):
        start_idx = lval**2
        length = 2 * lval + 1
        expand_index[start_idx : (start_idx + length)] = lval
    return expand_index


@compile_mode("script")
class EquiformerRMSLayerNorm(nn.Module):
    '''
        Irreps should have same multiplicity, e.g., "16x0e + 16x1o + 16x2e".
        https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/models/uma/nn/layer_norm.py
    '''
    def __init__(
        self,
        irreps: Irreps,
        eps: float = 1e-12,
        affine: bool = True,
        normalization: str = "component",
        centering: bool = True,
        std_balance_degrees: bool = True,
    ):
        super().__init__()

        self.irreps = irreps
        self.lmax = irreps.lmax
        self.num_channels = irreps[0].mul
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees

        # for L >= 0
        if self.affine:
            self.affine_weight = nn.Parameter(
                torch.ones((self.lmax + 1), self.num_channels)
            )
            if self.centering:
                self.affine_bias = nn.Parameter(torch.zeros(self.num_channels))
            else:
                self.register_parameter("affine_bias", nn.Parameter(torch.zeros(1)))
        else:
            self.register_parameter("affine_weight", nn.Parameter(torch.zeros(1)))
            self.register_parameter("affine_bias", nn.Parameter(torch.zeros(1)))

        assert normalization in ["norm", "component"]
        self.normalization = normalization

        expand_index = get_l_to_all_m_expand_index(self.lmax)
        self.register_buffer("expand_index", expand_index)

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            for lval in range(self.lmax + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (
                    1.0 / length
                )
            balance_degree_weight = balance_degree_weight / (self.lmax + 1)
            self.register_buffer("balance_degree_weight", balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, centering={self.centering}, std_balance_degrees={self.std_balance_degrees})"
    
    def forward(self, node_input: torch.Tensor) -> torch.Tensor:
        '''
            Assume input is of shape [N, sphere_basis, C]
        '''
        node_input = self.reshape_irreps(node_input)

        feature = node_input

        if self.centering:
            feature_l0 = feature.narrow(1, 0, 1)
            feature_l0_mean = feature_l0.mean(dim=2, keepdim=True)  # [N, 1, 1]
            feature_l0 = feature_l0 - feature_l0_mean
            feature = torch.cat(
                (feature_l0, feature.narrow(1, 1, feature.shape[1] - 1)), dim=1
            )

        # for L >= 0
        if self.normalization == "norm":
            assert not self.std_balance_degrees
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
        elif self.normalization == "component":
            if self.std_balance_degrees:
                feature_norm = feature.pow(2)  # [N, (L_max + 1)**2, C]
                feature_norm = torch.einsum(
                    "nic, ia -> nac", feature_norm, self.balance_degree_weight
                )  # [N, 1, C]
            else:
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]
        else:
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]

        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)  # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        if self.affine:
            weight = self.affine_weight.view(
                1, (self.lmax + 1), self.num_channels
            )  # [1, L_max + 1, C]
            weight = torch.index_select(
                weight, dim=1, index=self.expand_index
            )  # [1, (L_max + 1)**2, C]
            feature_norm = feature_norm * weight  # [N, (L_max + 1)**2, C]

        out = feature * feature_norm

        if self.affine and self.centering:
            out[:, 0:1, :] = out.narrow(1, 0, 1) + self.affine_bias.view(
                1, 1, self.num_channels
            )

        out = self.recover_irreps(out)
        return out

    def reshape_irreps(self, input: torch.Tensor) -> torch.Tensor:
        '''
            shape [N, feature] -> [N, sphere_basis, C]
        '''
        out = []
        for l in range(self.lmax + 1):
            feature = input[:, (l ** 2) * self.num_channels : ((l + 1) ** 2) * self.num_channels]
            feature = feature.reshape(-1, self.num_channels, 2 * l + 1).transpose(1, 2)
            out.append(feature)
        out = torch.cat(out, dim=1)
        return out
    
    def recover_irreps(self, input: torch.Tensor) -> torch.Tensor:
        '''
            shape [N, sphere_basis, C] -> [N, feature]
        '''
        out = []
        for l in range(self.lmax + 1):
            feature = input[:, l ** 2 : (l + 1) ** 2]
            feature = feature.transpose(1, 2).flatten(1)
            out.append(feature)
        out = torch.cat(out, dim=1)
        return out


@compile_mode("script")
class RMSLayerNorm(nn.Module):
    '''
        Irreps can have different multiplicity, e.g., "16x0e + 8x1o + 4x2e".
    '''
    def __init__(
            self, 
            irreps: Irreps, 
            eps: float = 1e-12, 
            affine: bool = True, 
            centering: bool = True, 
            std_balance_degrees: bool = True
            ) -> None:
        super().__init__()

        self.irreps = irreps
        self.lmax: int = self.irreps.lmax
        self.max_dim: int = max([self.irreps[l].mul for l in range(self.lmax + 1)])
        # self.slices: List[slice] = self.irreps.slices()
        self.slices: List[int] = [0] + [i.stop for i in self.irreps.slices()]
        self.channels: List[int] = [self.irreps[l].mul for l in range(self.lmax + 1)]
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees
        
        self.affine_weight = torch.jit.annotate(Optional[nn.ParameterList], None)
        self.affine_bias = torch.jit.annotate(Optional[nn.Parameter], None)
        if self.affine:
            self.affine_weight = nn.ParameterList()
            for l in range(self.lmax + 1):
                self.affine_weight.append(nn.Parameter(torch.ones(self.irreps[l].mul)))  # [C_L]
            if self.centering:
                self.affine_bias = nn.Parameter(torch.zeros(self.irreps[0].mul))  # [C_0]

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            balance_channel_weight = torch.zeros(self.max_dim, 1)
            for lval in range(self.lmax + 1):
                start_idx = lval**2
                length = 2 * lval + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = 1.0 / length
                balance_channel_weight[:self.irreps[lval].mul, :] += 1
            balance_weight = torch.einsum("ai, bi -> ab", balance_degree_weight, 1 / balance_channel_weight)  # [(L + 1) ** 2, max_dim]
            self.register_buffer("balance_weight", balance_weight)
        else:
            self.balance_weight = None

    def __repr__(self):
        num_params = sum(p.numel() for p in self.parameters())
        return f"{self.__class__.__name__}(irreps={self.irreps}, eps={self.eps}, affine={self.affine}, centering={self.centering}, std_balance_degrees={self.std_balance_degrees}) | params: {num_params}"
    
    def forward(self, node_input: torch.Tensor) -> torch.Tensor:
        '''
            Assume input is of shape [N, sum(2L + 1 * C)]
        '''
        # for L = 0
        if self.centering:
            feature_l0 = node_input[:, self.slices[0] : self.slices[1]]  # [N, C_0]
            feature_l0_mean = feature_l0.mean(dim=1, keepdim=True)  # [N, 1]
            feature_l0 = feature_l0 - feature_l0_mean
            if self.lmax == 0:
                feature = feature_l0
            else:
                feature = torch.cat((feature_l0, node_input[:, self.slices[1]:]), dim=1)  # [N, sum(2L + 1 * C)]
        else:
            feature = node_input

        weights = torch.jit.annotate(Optional[torch.Tensor], None)
        if self.affine:
            weights_list = []
            for l, weight in enumerate(self.affine_weight):
                weight = weight.view(-1, 1).repeat(1, 2 * l + 1).view(1, -1)  # [1, (2L + 1) * C_L]
                weights_list.append(weight)
            weights = torch.cat(weights_list, dim=1)  # [1, sum((2L + 1) * C)]

        # for L >= 0
        feature_list = []
        # num_paddings = []
        for l in range(self.lmax + 1):
            feature_l = feature[:, self.slices[l] : self.slices[l+1]].view(-1, self.channels[l], 2 * l + 1).transpose(1, 2)  # [N, 2L + 1, C_L]
            feature_l = torch.nn.functional.pad(feature_l, (0, self.max_dim - feature_l.shape[2]))  # [N, 2L + 1, max_dim]
            # num_paddings.append(self.max_dim - feature_l.shape[2])
            feature_list.append(feature_l)
        feature_list = torch.cat(feature_list, dim=1)  # [N, (L + 1) ** 2, max_dim]

        if self.std_balance_degrees:
            feature_norm = feature_list.pow(2)  # [N, (L_max + 1)**2, max_dim]
            feature_norm = (feature_norm * self.balance_weight.unsqueeze(0)).sum(dim=1, keepdim=True)  # [N, 1, max_dim]
        else:
            raise NotImplementedError(f"set std_balance_degrees as True.")

        feature_norm = torch.mean(feature_norm, dim=2)  # [N, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        if weights is not None:
            feature = feature * feature_norm * weights  # [N, sum(2L + 1 * C)]

        if self.affine_bias is not None:
            feature[:, self.slices[0] : self.slices[1]] = feature[:, self.slices[0] : self.slices[1]] + self.affine_bias.view(1, -1)

        return feature
    

if __name__ == '__main__':
    # Example usage
    import time

    torch.set_printoptions(precision=4, sci_mode=False)
    num_channels = 128
    lmax = 2
    batch_size = 128

    # irreps = Irreps(f"{num_channels}x0e + {num_channels}x1o + {num_channels}x2e")
    irreps = Irreps("8x0e + 4x1o + 2x2e")
    x = torch.randn(batch_size, irreps.dim).requires_grad_(True) * 0.01  # Example input tensor
    print(x[0])

    # start = time.perf_counter()
    # sln = EquiformerRMSLayerNorm(irreps=irreps, affine=True, centering=False)
    # sln = compile(sln)
    # y_0 = sln(x)
    # print(y_0[0])
    # print(f"{sln} Elapsed time: {time.perf_counter() - start:.4f} seconds")
    # exit()

    start = time.perf_counter()
    sln = RMSLayerNorm(irreps=irreps, affine=True, centering=False)
    sln = compile(sln)
    y = sln(x)
    print(y[0])
    print(f"{sln} Elapsed time: {time.perf_counter() - start:.4f} seconds")
    print(y[0] / x[0])

    # print(torch.allclose(y_0, y, atol=1e-6))

