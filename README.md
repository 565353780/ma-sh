# MASH

## Setup

```bash
conda create -n mash python=3.8
conda activate mash
./setup.sh
```

## Prepare Dataset

```bash
python convert_normalize_mesh.py
python convert_sample_pcd.py
python convert_to_manifold.py
python convert_sample_sdf.py
```

## Run

```bash
python demo.py
```

## Fit your mesh

```bash
python train.py
```

## Enjoy it~
