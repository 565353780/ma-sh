import torch
from typing import Tuple

from ma_sh.Lib.ChamferDistance.chamfer_python import distChamfer

if torch.cuda.is_available():
    from ma_sh.Lib.ChamferDistance.chamfer3D.dist_chamfer_3D import chamfer_3DDist


def chamferDistance(
    pts1: torch.Tensor, pts2: torch.Tensor, is_cpu: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if is_cpu:
        return distChamfer(pts1, pts2)
    else:
        return chamfer_3DDist()(pts1, pts2)
