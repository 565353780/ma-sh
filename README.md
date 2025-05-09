# MASH: Masked Anchored SpHerical Distances for 3D Shape Representation and Generation (SIGGRAPH 2025 Conference)

### [Project Page](https://565353780.github.io/MASH/) | [Paper (arXiv)](https://arxiv.org/abs/2504.09149)

**This repository is the official implementation of *MASH Optimization (<https://arxiv.org/abs/2504.09149>)*.**

## Setup

```bash
conda create -n mash python=3.8
conda activate mash
./setup.sh
```

## Prepare Dataset

```bash
1. ./parallel_convert.sh convert_pipeline_pcd.py <parallel-run-num>
2. ./parallel_convert.sh convert_pipeline_sdf.py <parallel-run-num>
3. ./convert_mash.sh <gpu-id>
4. python convert_capture_image.py
5. python convert_encode_image.py
```

## Fit your custom data

```bash
python train.py
```

## Citation

```bibtex
@article{Li-2025-MASH,
  title = {MASH: Masked Anchored SpHerical Distances for 3D Shape Representation and Generation},
  author = {Changhao Li, Yu Xin, Xiaowei Zhou, Ariel Shamir, Hao Zhang, Ligang Liu, Ruizhen Hu},
  journal = {Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers (SIGGRAPH Conference Papers ’25)},
  volume = {ACM Tans. Graph. 1, 1 (April 2025)},
  pages = {11 pages},
  year = {2025}
}
```

## Enjoy it~
