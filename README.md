# Geometric Algebra Based Embeddings on Point Clouds: A comparative study of geometric inductive bias in neural networks


#### MLGP resource
https://github.com/pavlo-melnyk/mlgp-embedme
#### PointNet resource
https://github.com/charlesq34/pointnet

#### PointNet++ resource
https://github.com/charlesq34/pointnet2

#### VN-PointNet resource
https://github.com/FlyingGiraffe/vnn


To install the requirements, run:

```setup
pip install -r requirements.txt
```


# Dataset
ModelNet10 is available upon installation of torch_geometric


## Training

To train the model(s) in canonical setting (for example run):

```
 python3 .\train.py --train_all --epochs 61 --step 2 --batch_size 128 --data_type canonical --learning_rate 5e-4
```

and in data augmented setting


```
 python3 .\train.py --train_all --epochs 151 --step 2 --batch_size 128 --rotate --data_type augmented --learning_rate 5e-4
```

the available arguments are:
PointNet training: --train_pn
CpointNet training: --train_cpn
POintNet++ training: --train_baseline
CPointNet++ training: --train_geom
VN-PointNet training: --train_vn
or train all with: --train_all

## Evaluation

To evaluate all of the trained models on the corresponding test dataset, run:


```
python3 .\eval.py --test_all --trials 30
```

## Pre-trained models

You can find the pre-trained models in the ```pretrained_models``` folder.



### Main Findings:

## Table 1: Models Trained on Canonical Orientation

| Model | Test Acc (%) | x-axis (%) | y-axis (%) | z-axis (%) | Arbitrary (%) |
|---|---|---|---|---|---|
| PointNet | 84.0 | 31.3 ± 0.9 (↓52.8) | 24.4 ± 1.3 (↓59.6) | 44.0 ± 1.4 (↓40.0) | 13.7 ± 0.7 (↓70.3) |
| CG-PointNet | 87.8 | 33.7 ± 1.5 (↓54.1) | 26.1 ± 1.3 (↓61.7) | 44.0 ± 1.2 (↓43.8) | 12.8 ± 1.1 (↓75.0) |
| PointNet++ | 90.2 | 34.7 ± 1.2 (↓55.5) | 29.6 ± 1.0 (↓60.6) | 53.6 ± 1.0 (↓36.6) | 15.0 ± 0.9 (↓75.2) |
| CGA-PointNet++ | 90.6 | 39.4 ± 1.4 (↓51.2) | 36.1 ± 0.9 (↓54.6) | 55.1 ± 1.1 (↓35.5) | 19.8 ± 1.3 (↓70.8) |
| VN-PointNet | 63.7 | 63.7 ± 0.0 (↓0.0) | 63.7 ± 0.0 (↓0.0) | 63.7 ± 0.0 (↓0.0) | 63.7 ± 0.0 (↓0.0) |

## Table 2: Models Trained with SO(3) Rotation Augmentation

| Model | Test Acc (%) | x-axis (%) | y-axis (%) | z-axis (%) | Arbitrary (%) |
|---|---|---|---|---|---|
| PointNet | 77.8 | 52.4 ± 0.8 (↓25.3) | 51.9 ± 1.0 (↓25.8) | 65.6 ± 0.7 (↓12.2) | 44.4 ± 1.1 (↓33.4) |
| CG-PointNet | 71.1 | 48.5 ± 1.1 (↓22.7) | 44.8 ± 1.4 (↓26.4) | 61.0 ± 0.7 (↓10.2) | 37.1 ± 1.2 (↓34.0) |
| PointNet++ | 64.3 | 67.6 ± 0.8 (↑3.3) | 64.0 ± 1.2 (↓0.4) | 64.2 ± 1.1 (↓0.1) | 64.8 ± 1.1 (↑0.5) |
| CGA-PointNet++ | 71.9 | 71.0 ± 0.6 (↓0.9) | 71.1 ± 1.0 (↓0.8) | 72.1 ± 0.7 (↑0.1) | 71.7 ± 0.5 (↓0.2) |
 