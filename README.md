# PointNet: Deep Learning on Point Cloud Data

<div align="center">

![PointNet Architecture](Figure/pointnet_teaser.png)

*A PyTorch implementation of PointNet for 3D point cloud processing*

</div>

## Overview

This project implements **PointNet**, a pioneering deep learning architecture designed to directly process unordered 3D point cloud data. Based on the seminal paper ["PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation"](https://arxiv.org/abs/1612.00593) (CVPR 2017), this implementation demonstrates the network's versatility across multiple computer vision tasks.

### Key Features

- 🎯 **Point Cloud Classification** - Object recognition from 3D point clouds
- 🔍 **Semantic Part Segmentation** - Per-point labeling for detailed shape analysis  
- 🔄 **Auto-Encoding** - Learning compact latent representations of 3D shapes
- ⚡ **Efficient Processing** - Direct consumption of raw point cloud data without voxelization
- 🏗️ **Modular Architecture** - Clean, extensible code structure built with PyTorch

## Architecture

PointNet's modular design enables multiple downstream tasks through a shared feature extraction backbone. Below are the four key architectures implemented in this project:

### 0. Global Feature Extraction

![Feature Extraction](Figure/feat.png)

The **PointNetFeat** module serves as the backbone for extracting global features from point clouds. It takes N×3 point coordinates as input and produces a 1024-dimensional global descriptor through:
- Input and feature transformation networks (T-Nets) for spatial alignment
- Shared MLPs for point-wise feature learning
- Max pooling for permutation-invariant global feature aggregation

This global feature vector captures the geometric essence of the entire point cloud and is used by all downstream task networks.

### 1. Point Cloud Classification

![Classification Network](Figure/cls.png)

The **PointNetCls** architecture extends the feature extraction backbone with classification heads. After obtaining the 1024-d global feature, it applies:
- Fully connected layers (512 → 256 → k classes)
- Batch normalization and ReLU activations
- Dropout for regularization
- Final softmax for category prediction

**Performance**: 88.6% accuracy on ModelNet40 (40 object categories)

### 2. Point Cloud Part Segmentation

![Segmentation Network](Figure/seg.png)

The **PointNetPartSeg** architecture combines local and global features for per-point semantic labeling:
- Concatenates point-wise features with the global descriptor
- Shared MLPs process the combined features
- Outputs m-dimensional logits for each point (m = number of part categories)
- Enables fine-grained understanding of object geometry

**Performance**: 83.6% instance mIoU on ShapeNet Parts dataset

### 3. Point Cloud Auto-Encoding

![Auto-Encoding Network](Figure/ae.png)

The **PointNetAutoEncoder** learns compressed representations through reconstruction:
- **Encoder**: PointNetFeat extracts 1024-d latent codes
- **Decoder**: MLPs progressively expand latent vector (1024 → 2048 → N×3)
- Reconstructs N points in 3D space
- Learns meaningful shape representations without labels

**Performance**: 0.0043 Chamfer distance on ModelNet40

---

**Common Design Principles:**
- All architectures use spatial transformer networks (T-Nets) for input alignment
- Max pooling ensures permutation invariance
- Shared MLPs enable efficient point-wise feature learning
- Modular design allows easy extension to new taskspermutations. The architecture employs spatial transformer networks (T-Nets) for canonical alignment of input data and features.

## Results

### Classification (ModelNet40)

| Metric | Performance |
|--------|-------------|
| Overall Accuracy | **88.6%** |

Achieved competitive performance on the challenging ModelNet40 3D shape classification benchmark.

### Part Segmentation (ShapeNet Parts)

| Metric | Performance |
|--------|-------------|
| Instance mIoU | **83.6%** |

High-quality semantic segmentation results on ShapeNet part annotation dataset.

### Auto-Encoding (ModelNet40)

| Metric | Performance |
|--------|-------------|
| Chamfer Distance | **0.0043** |

Low reconstruction error demonstrating effective learning of compact 3D shape representations.

## Installation

### Requirements

- Python 3.9+
- PyTorch 1.13.0
- CUDA 11.6 (for GPU acceleration)

### Setup

Create a conda environment and install dependencies:

```bash
conda create -n pointnet python=3.9
conda activate pointnet

# Install PyTorch with CUDA support
conda install pytorch=1.13.0 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia

# Install PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d

# Install additional dependencies
pip install tqdm h5py matplotlib
```

## Project Structure

```
pointnet/
├── model.py                      # PointNet architecture implementations
├── train_cls.py                  # Classification training script
├── train_seg.py                  # Segmentation training script  
├── train_ae.py                   # Auto-encoder training script
├── dataloaders/
│   ├── modelnet.py              # ModelNet40 dataset loader
│   └── shapenet_partseg.py      # ShapeNet parts dataset loader
├── utils/
│   ├── metrics.py               # Evaluation metrics
│   ├── misc.py                  # Utility functions
│   └── model_checkpoint.py      # Model checkpointing
└── visualization.ipynb           # Point cloud visualization tools
```

## Usage

### Classification

Train the classification network on ModelNet40:

```bash
python train_cls.py --batch_size 32 --lr 0.001 --epochs 50
```

### Part Segmentation

Train the segmentation network on ShapeNet Parts:

```bash
python train_seg.py --batch_size 32 --lr 0.001 --epochs 50
```

### Auto-Encoding

Train the auto-encoder on ModelNet40:

```bash
python train_ae.py --batch_size 32 --lr 0.001 --epochs 50
```

Model checkpoints are automatically saved during training when validation performance improves.

## Datasets

The project uses two benchmark datasets:

- **ModelNet40**: 12,311 CAD models from 40 categories (9,843 train / 2,468 test)
- **ShapeNet Parts**: 16,881 shapes from 16 categories with fine-grained part annotations

Datasets are automatically downloaded when running training scripts. Alternatively, download manually from [here](https://drive.google.com/drive/folders/1Ly2DbsBMBXp75CGCA4vnJ3uoyDxOJz84?usp=drive_link) and place in the `data/` directory.

## Implementation Details

### Model Components

- **PointNetFeat**: Feature extraction backbone that produces 1024-dimensional global descriptors
- **PointNetCls**: Classification head for object category prediction
- **PointNetPartSeg**: Segmentation head for per-point part labeling
- **PointNetAutoEncoder**: Encoder-decoder architecture for shape reconstruction
- **STNKd**: Spatial Transformer Network for input/feature alignment

### Training

- Optimizer: Adam
- Learning rate: 0.001 (default)
- Batch size: 32 (default)
- Data augmentation: Random rotation, jittering, scaling

## Visualization

The `visualization.ipynb` notebook provides tools for visualizing:
- Input point clouds
- Segmentation predictions  
- Auto-encoder reconstructions

## References

This implementation is based on:

```bibtex
@article{qi2017pointnet,
  title={PointNet: Deep learning on point sets for 3d classification and segmentation},
  author={Qi, Charles R and Su, Hao and Mo, Kaichun and Guibas, Leonidas J},
  journal={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2017}
}
```

### Related Work

For further exploration of point cloud deep learning:

- **PointNet++**: Learning Deep Hierarchical Features from Point Sets in a Metric Space (NeurIPS 2017)
- **DGCNN**: Dynamic Graph CNN for Learning on Point Clouds (TOG 2019)  
- **PointConv**: Deep Convolutional Networks on 3D Point Clouds (CVPR 2019)
- **PointNeXt**: Revisiting PointNet++ with Improved Training and Scaling Strategies (NeurIPS 2022)
- **PointMLP**: Rethinking Network Design and Local Geometry in Point Cloud (ICLR 2022)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*For questions or collaboration opportunities, feel free to open an issue or reach out!*


