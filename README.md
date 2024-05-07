# MASH

## Setup

```bash
conda create -n mash python=3.8
conda activate mash
./setup.sh
```

## Prepare Dataset

```bash
1. python convert_normalize_mesh.py
2. ./convert_sample_pcd.sh
3. python convert_to_manifold.py
4. python convert_sample_sdf.py
5. python convert_mash.py
6. python convert_capture_image.py
7. python convert_encode_image.py
8. python convert_encode_mash.py
```

You can run by this orders:

```bash
1 --> 2 --> 5 --> 8
  |-> 3 --> 4
        |-> 6 --> 7
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
