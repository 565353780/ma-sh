import torch
from typing import Union


class SimpleMash(object):
    def __init__(
        self,
        anchor_num: int = 400,
        mask_degree_max: int = 3,
        sh_degree_max: int = 2,
        sample_phi_num: int = 10,
        sample_theta_num: int = 10,
        idx_dtype=torch.int64,
        dtype=torch.float32,
        device: str = "cpu",
    ) -> None:
        # Super Params
        self.anchor_num = anchor_num
        self.mask_degree_max = mask_degree_max
        self.sh_degree_max = sh_degree_max
        self.sample_phi_num = sample_phi_num
        self.sample_theta_num = sample_theta_num
        self.idx_dtype = idx_dtype
        self.dtype = dtype
        self.device = device

        # Processed Data
        self.sample_phis = torch.linspace(
            0, 2 * torch.pi, self.sample_phi_num, dtype=self.dtype, device=self.device
        )
        self.sample_thetas = torch.linspace(
            0, torch.pi, self.sample_theta_num, dtype=self.dtype, device=self.device
        )
        self.sample_phi_theta_mat = torch.stack(
            torch.meshgrid(self.sample_phis, self.sample_thetas, indexing="ij"), dim=-1
        )

        # Diff Params
        self.mask_params = torch.ones(
            [anchor_num, 2 * mask_degree_max + 1],
            dtype=self.dtype,
        ).to(self.device)
        self.sh_params = torch.zeros(
            [anchor_num, (sh_degree_max + 1) ** 2],
            dtype=self.dtype,
        ).to(self.device)
        self.ortho_poses = torch.zeros([anchor_num, 6], dtype=self.dtype).to(
            self.device
        )
        self.positions = torch.zeros([anchor_num, 3], dtype=self.dtype).to(self.device)
        return

    def setGradState(
        self, need_grad: bool, anchor_mask: Union[torch.Tensor, None] = None
    ) -> bool:
        if anchor_mask is None:
            self.mask_params.requires_grad_(need_grad)
            self.sh_params.requires_grad_(need_grad)
            self.ortho_poses.requires_grad_(need_grad)
            self.positions.requires_grad_(need_grad)
            return True

        self.mask_params[anchor_mask].requires_grad_(need_grad)
        self.sh_params[anchor_mask].requires_grad_(need_grad)
        self.ortho_poses[anchor_mask].requires_grad_(need_grad)
        self.positions[anchor_mask].requires_grad_(need_grad)
        return True

    def toMaskThetas(self) -> torch.Tensor:
        mask_thetas = torch.zeros(
            (self.anchor_num, self.sample_phi_num),
            dtype=self.dtype,
            device=self.device,
        )

        mask_thetas = mask_thetas + self.mask_params[:, 0].unsqueeze(1)

        for degree in range(1, self.mask_degree_max + 1):
            cos_term = torch.cos(degree * self.sample_phis)
            sin_term = torch.sin(degree * self.sample_phis)

            mask_thetas = (
                mask_thetas
                + self.mask_params[:, 2 * degree - 1].unsqueeze(1) * cos_term
            )
            mask_thetas = (
                mask_thetas + self.mask_params[:, 2 * degree].unsqueeze(1) * sin_term
            )

        mask_thetas = torch.sigmoid(mask_thetas)

        return mask_thetas

    def toWeightedSamplePhiThetaMat(self) -> torch.Tensor:
        weighted_sample_phi_theta_mat = self.sample_phi_theta_mat.unsqueeze(0).repeat(
            self.anchor_num, 1, 1, 1
        )

        theta_weights = self.toMaskThetas().unsqueeze(-1)

        weights = torch.ones_like(weighted_sample_phi_theta_mat)
        weights[:, :, :, 1] = theta_weights

        weighted_sample_phi_theta_mat = weighted_sample_phi_theta_mat * weights

        return weighted_sample_phi_theta_mat
