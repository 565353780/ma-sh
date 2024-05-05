import torch

def toNormalTag(anchor_num: int, mask_boundary_points: torch.Tensor,
                mask_boundary_point_idxs: torch.Tensor,
                mask_boundary_normals: torch.Tensor) -> torch.Tensor:
    normal_tag = torch.ones([anchor_num], dtype=torch.bool).to(mask_boundary_points.device)

    tagged_anchor_idxs = torch.tensor([0]).type(torch.int)
    untagged_anchor_idxs = torch.range(1, anchor_num).type(torch.int)

    print(tagged_anchor_idxs)
    print(untagged_anchor_idxs)
    exit()

    mask_boundary_points_list = []
    mask_boundary_normals_list = []

    for i in range(anchor_num):
        anchor_point_mask = mask_boundary_point_idxs == i

        anchor_boundary_points = mask_boundary_points[anchor_point_mask]
        anchor_boundary_normals = mask_boundary_normals[anchor_point_mask]

        mask_boundary_points_list.append(anchor_boundary_points)
        mask_boundary_normals_list.append(anchor_boundary_normals)

    while len(tagged_anchor_idxs) < anchor_num:
        tagged_anchor_idxs

    exit()
    return normal_tag
