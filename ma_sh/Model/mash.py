import torch

from ma_sh.Config.degree import MAX_MASK_DEGREE, MAX_SH_DEGREE
from ma_sh.Method.kernel import toParams, toPreLoadDatas, toMashSamplePoints

try:
    from ma_sh.Method.render import renderPoints

    NO_RENDER = 0
except:
    print("[WARN][mash::import]")
    print("\t import open3d failed! all render functions will be disabled now!")
    NO_RENDER = 1


class Mash(object):
    def __init__(
        self,
        anchor_num: int,
        mask_degree_max: int,
        sh_degree_max: int,
        mask_boundary_sample_num: int,
        sample_polar_num: int,
        idx_dtype=torch.int64,
        dtype=torch.float64,
        device: str = "cpu",
    ) -> None:
        # Super Params
        self.anchor_num = anchor_num
        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.mask_boundary_sample_num = mask_boundary_sample_num
        self.sample_polar_num = sample_polar_num
        self.idx_dtype = idx_dtype
        self.dtype = dtype
        self.device = device

        # Diff Params
        self.mask_params = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.sh_params = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.rotate_vectors = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.positions = torch.tensor([0.0], dtype=dtype).to(self.device)

        # Pre Load Datas
        self.sample_phis = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.sample_thetas = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.mask_boundary_phi_idxs = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.mask_boundary_base_values = torch.tensor([0.0], dtype=dtype).to(
            self.device
        )
        self.sample_base_values = torch.tensor([0.0], dtype=dtype).to(self.device)
        self.sample_sh_directions = torch.tensor([0.0], dtype=dtype).to(self.device)

        self.reset()
        return

    def setGradState(self, need_grad: bool) -> bool:
        self.mask_params.requires_grad_(need_grad)
        self.sh_params.requires_grad_(need_grad)
        self.rotate_vectors.requires_grad_(need_grad)
        self.positions.requires_grad_(need_grad)
        return True

    def loadParams(
        self,
        mask_params: torch.Tensor,
        sh_params: torch.Tensor,
        rotate_vectors: torch.Tensor,
        positions: torch.Tensor,
    ) -> bool:
        self.mask_params.data = (
            mask_params.detach().clone().type(self.dtype).to(self.device)
        )
        self.sh_params.data = (
            sh_params.detach().clone().type(self.dtype).to(self.device)
        )
        self.rotate_vectors.data = (
            rotate_vectors.detach().clone().type(self.dtype).to(self.device)
        )
        self.positions.data = (
            positions.detach().clone().type(self.dtype).to(self.device)
        )
        return True

    def initParams(self) -> bool:
        self.mask_params, self.sh_params, self.rotate_vectors, self.positions = (
            toParams(
                self.anchor_num,
                self.mask_degree_max,
                self.sh_degree_max,
                self.dtype,
                self.device,
            )
        )
        return True

    def updatePreLoadDatas(self) -> bool:
        (
            self.sample_phis,
            self.sample_thetas,
            self.mask_boundary_phi_idxs,
            self.mask_boundary_base_values,
            self.sample_base_values,
            self.sample_sh_directions,
        ) = toPreLoadDatas(
            self.anchor_num,
            self.mask_degree_max,
            self.mask_boundary_sample_num,
            self.sample_polar_num,
            self.idx_dtype,
            self.dtype,
            self.device,
        )
        return True

    def reset(self) -> bool:
        self.initParams()
        self.updatePreLoadDatas()
        return True

    def updateMaskDegree(self, mask_degree_max: int) -> bool:
        if mask_degree_max == self.mask_degree_max:
            return True

        if mask_degree_max < 0 or mask_degree_max > MAX_MASK_DEGREE:
            print("[ERROR][Mash::updateMaskDegree]")
            print("\t mask degree max out of range!")
            print("\t mask_degree_max:", mask_degree_max)
            return False

        self.mask_degree_max = mask_degree_max

        new_dim = 2 * self.mask_degree_max + 1

        new_mask_params = torch.zeros(
            [self.mask_params.shape[0], new_dim], dtype=self.dtype
        ).to(self.device)

        copy_dim = min(new_dim, self.mask_params.shape[1])

        new_mask_params[:, :copy_dim] = self.mask_params.detach().clone()

        self.mask_params = new_mask_params

        self.updatePreLoadDatas()
        return True

    def updateSHDegree(self, sh_degree_max: int) -> bool:
        if sh_degree_max == self.sh_degree_max:
            return True

        if sh_degree_max < 0 or sh_degree_max > MAX_SH_DEGREE:
            print("[ERROR][Mash::updateSHDegree]")
            print("\t sh degree max out of range!")
            print("\t sh_degree_max:", sh_degree_max)
            return False

        self.sh_degree_max = sh_degree_max

        new_dim = (self.sh_degree_max + 1) ** 2

        new_sh_params = torch.zeros(
            [self.sh_params.shape[0], new_dim], dtype=self.dtype
        ).to(self.device)

        copy_dim = min(new_dim, self.sh_params.shape[1])

        new_sh_params[:, :copy_dim] = self.sh_params.detach().clone()

        self.sh_params = new_sh_params
        return True

    def toSamplePoints(self) -> torch.Tensor:
        sample_points = toMashSamplePoints(
            self.sh_degree_max,
            self.mask_params,
            self.sh_params,
            self.rotate_vectors,
            self.positions,
            self.sample_phis,
            self.sample_thetas,
            self.mask_boundary_phi_idxs,
            self.mask_boundary_base_values,
            self.sample_base_values,
            self.sample_sh_directions,
        )

        return sample_points

    def renderSamplePoints(self) -> bool:
        if NO_RENDER:
            return False

        sample_points = self.toSamplePoints().detach().clone().cpu().numpy()

        renderPoints(sample_points)
        return True
