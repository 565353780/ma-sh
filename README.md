# MASH

## Setup

```bash
conda create -n mash python=3.8
conda activate mash
./setup.sh
```

## Prepare Dataset

```bash
# Step 1
python convert_normalize_mesh.py

# Step 2, can run parallel
python convert_sample_pcd.py
python convert_to_manifold.py

# Step 3, can run parallel
python convert_sample_sdf.py
python convert_mash.py
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
