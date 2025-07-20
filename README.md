# MIME-ViT: Medical Image Segmentation with Vision Transformer

MIME-ViT is a hybrid deep learning model that combines Vision Transformer (ViT) and U-Net architectures for medical image segmentation tasks, specifically designed for breast cancer detection and segmentation in mammography images.

## Model Architecture

The MIME-ViT model integrates:
- **Vision Transformer (ViT)** components for capturing global features at multiple scales
- **U-Net** decoder architecture for precise localization and segmentation
- **Residual blocks** for enhanced feature extraction
- **Multi-scale processing** with different patch sizes (512, 256, 64, 16)

### Key Features:
- Multi-scale Vision Transformer blocks for hierarchical feature extraction
- Skip connections for preserving spatial information
- Hybrid CNN-Transformer architecture
- Support for both UNet and MIMEViT models
- Configurable data augmentation pipeline
- Multiple loss functions including Dice Loss, Focal Loss, and custom FBeta Loss

## Requirements

```bash
torch
torchvision
numpy
pandas
opencv-python
albumentations
scikit-learn
matplotlib
tqdm
PIL
pycocotools
shapely
einops
torchmetrics
```

## Dataset Structure

The model expects data in CSV format with the following structure:
- `save_dir`: Path to the directory containing image and annotation files
- `annotation`: Boolean indicating whether the image has annotations
- `labels`: Labels for the annotations

Each data directory should contain:
- `image.png`: The input mammography image
- `AnnotationFile.Json`: JSON file with annotation information

## Configuration

The model configuration is defined in `config.py`. Key parameters include:

### General Settings
- `EPOCHS`: Number of training epochs (default: 10)
- `BATCH_SIZE`: Batch size for training (default: 16)
- `LR`: Learning rate (default: 1e-3)
- `NUM_CLASSES`: Number of output classes (default: 1)
- `DEVICE`: Computation device ('cuda:0' or 'cpu')

### Model Settings
- `NAME`: Model type ('UNet' or 'MIMEViT')
- `PRETRAINED`: Whether to use pretrained weights

### Optimization Settings
- `OPTIMIZER`: Optimizer type ('Adam' or 'AdamW')
- `SCHEDULER`: Learning rate scheduler ('StepLR', 'ReduceLROnPlateau', 'MultiStepLR', 'CosineAnnealingLR')

### Augmentation Settings
- `CLAHE`: Contrast Limited Adaptive Histogram Equalization
- `ROTATE`: Random rotation with configurable limits
- `HFLIP`/`VFLIP`: Horizontal/Vertical flipping
- `RANDOMSIZEDCROP`: Random sized cropping
- `BREASTSHIFT`: Custom breast-specific augmentation

## Usage

### Training

To train the model, use the following command:

```bash
python train.py --config path/to/config.py
```

**Example:**
```bash
python train.py --config config.py
```

The training script will:
1. Load the configuration from the specified config file
2. Create output directories in `result_dir/results/`
3. Initialize the model (UNet or MIMEViT) based on configuration
4. Train the model with the specified parameters
5. Save the best model weights and training logs
6. Generate training metrics and loss curves

**Training Output:**
- Model weights: `result_dir/results/result_{config}/fold1/weights/`
- Training logs: `result_dir/results/result_{config}/fold1/logs/`
- Metrics: CSV files with epoch-wise performance metrics

### Testing/Inference

To test the trained model, use:

```bash
python test.py --config path/to/config.py
```

**Example:**
```bash
python test.py --config config.py
```

The test script will:
1. Load the trained model weights
2. Evaluate on test dataset
3. Generate prediction images and metrics
4. Save results in pickle format for further analysis

**Test Output:**
- Prediction images: `{LOG_DIR}/test_images_cv/fold1/`
- Results pickle: `{LOG_DIR}/pickles/test_results_fold1.pkl`
- Performance metrics: Dice scores and IoU for different mask types

### Configuration Examples

#### Basic Training Configuration
```python
class Config:
    class GENERAL:
        EPOCHS = 50
        BATCH_SIZE = 8
        LR = 1e-4
        NUM_CLASSES = 3
        DEVICE = 'cuda:0'
    
    class MODEL:
        NAME = 'MIMEViT'
        PRETRAINED = False
    
    class OPTIMIZER:
        NAME = 'AdamW'
        
    class SCHEDULER:
        NAME = 'CosineAnnealingLR'
        T_MAX = 50
        ETA_MIN = 1e-6
```

#### Data Augmentation Configuration
```python
class AUGMENTATION:
    CLAHE = True
    ROTATE = True
    class ROTATE:
        LIMIT = 30
    HFLIP = True
    VFLIP = True
    BREASTSHIFT = True
```

## Model Performance

The model evaluates performance using:
- **Dice Coefficient**: Measures overlap between predicted and ground truth masks
- **IoU (Intersection over Union)**: Measures segmentation accuracy
- **Bounding Box IoU**: For cases with annotation masks
- **Background Accuracy**: For cases without annotation masks

## File Structure

```
MIME-ViT/
├── README.md
├── config.py                 # Configuration file
├── train.py                 # Training script
├── test.py                  # Testing script
├── models/
│   ├── __init__.py
│   ├── mimevit_model.py     # MIME-ViT model implementation
│   ├── mimevit_parts.py     # Model components (ViT, ResBlock, etc.)
│   ├── unet_model.py        # U-Net model implementation
│   └── unet_parts.py        # U-Net components
└── utils/
    ├── __init__.py
    ├── dataset_segmentation.py  # Dataset loading and preprocessing
    ├── loss.py                  # Loss functions
    ├── utils_csv.py            # Training utilities
    ├── utils.py                # General utilities
    ├── pcgrad.py               # PCGrad optimizer
    └── run_pretraining.sh      # Pretraining script
```

## Loss Functions

The framework supports multiple loss functions:

1. **Focal Loss**: Addresses class imbalance
2. **Weighted Focal Loss**: Focal loss with class weighting
3. **Dice Loss**: Optimizes for segmentation overlap
4. **Tversky Loss**: Generalizes Dice loss with FP/FN trade-off control
5. **FBeta Loss**: Balances precision and recall
6. **CrossEntropy FBeta Loss**: Combined cross-entropy and FBeta loss

## Advanced Features

### Multi-GPU Training
Set `DEVICE = 'cuda'` in config for automatic GPU detection.

### Gradient Accumulation
The training script supports gradient accumulation with configurable steps.

### Mixed Precision Training
Automatic mixed precision (AMP) is enabled for faster training.

### Custom Metrics
- Bounding box IoU for detection tasks
- Background dice score for negative samples
- Ensemble scoring for multiple model predictions

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch size in config
2. **Convergence issues**: Adjust learning rate or scheduler
3. **Poor segmentation**: Check data augmentation settings
4. **Training crashes**: Verify dataset paths and file formats

### Performance Tips:

1. Use mixed precision training for faster computation
2. Optimize batch size based on GPU memory
3. Use appropriate data augmentation for medical images
4. Monitor training metrics to detect overfitting

## Citation

If you use this code in your research, please cite the relevant papers and acknowledge the MIME-ViT architecture.

## License

Please refer to the license file for usage terms and conditions.