import sys
sys.path.append('../data-convert')
sys.path.append("../chamfer-distance")

import torch
from typing import Union

from data_convert.Module.pipeline_convertor import PipelineConvertor

from ma_sh.Module.Convertor.mesh_to_pcd import Convertor as MeshToPcdConvertor
from ma_sh.Module.Convertor.pcd_to_mash import Convertor as PcdToMashConvertor
