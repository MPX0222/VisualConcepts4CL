<div align="center">
<h1>Augmenting Continual Learning of Diseases with LLM-Generated Visual Concepts</h1>
</div>

<div align="center">
<p>
    <a href="">Jiantao Tan</a><sup>1</sup>&nbsp;&nbsp;
    <a href="">Peixian Ma</a><sup>2</sup>&nbsp;&nbsp;
    <a href="">Kanghao Chen</a><sup>2</sup>&nbsp;&nbsp;
    <a href="">Zhiming Dai</a><sup>1</sup>&nbsp;&nbsp;
    <a href="">Ruixuan Wang</a><sup>1,3,4</sup>&nbsp;&nbsp;
</p>

<p>
    <sup>1</sup>Sun Yat-sen University
    <sup>2</sup>The Hong Kong University of Science and Technology (Guangzhou)
    <sup>3</sup>Peng Cheng Laboratory
    <sup>4</sup>Key Laboratory of Machine Intelligence and Advanced Computing
</p>
</div>


<div align="center" style="display: flex; gap: 5px; justify-content: center;">
<a href="https://arxiv.org/abs/2508.03094"><img src="https://img.shields.io/badge/arXiv-red?style=for-the-badge&logo=arxiv"/></a>
<a href="https://github.com/MPX0222/VisualConcepts4CL"><img src="https://img.shields.io/badge/GitHub-black?style=for-the-badge&logo=github"/></a>
<a href="https://github.com/MPX0222/VisualConcepts4CL/stargazers"><img src="https://img.shields.io/github/stars/MPX0222/VisualConcepts4CL?style=for-the-badge&color=white"/></a>
</div>

---

## 📖 Abstract

This repository contains the official implementation of our paper on augmenting continual learning for disease classification using LLM-generated visual concepts. The method leverages Large Language Models (LLMs) to generate rich visual concept descriptions for classes, which are then integrated with CLIP-based vision-language models to enhance class-incremental learning performance.

## ✨ Key Features

- **LLM-Generated Visual Concepts**: Utilizes LLM-generated textual descriptions to provide rich semantic information for each class
- **CLIP-Based Architecture**: Leverages CLIP (Contrastive Language-Image Pre-training) as the backbone for vision-language understanding
- **Class-Incremental Learning (CIL)**: Supports incremental learning scenarios where new classes are introduced sequentially
- **Cross-Attention Mechanism**: Employs attention mechanisms to fuse visual and textual features effectively
- **Two-Stage Training**: Implements a two-stage training strategy with class-aware regularization
- **Replay Memory Support**: Optional replay memory mechanism to mitigate catastrophic forgetting
- **Multiple Dataset Support**: Compatible with various datasets including medical imaging (Skin40, MedMNIST) and general vision datasets (CIFAR100, ImageNet-R, CUB200, Cars196)

## 🚀 Installation

### Requirements

- Python 3.7+
- PyTorch 2.3.1
- CUDA (for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/MPX0222/VisualConcepts4CL.git
cd VisualConcepts4CL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download pretrained CLIP models:
   - Create a `pretrained_model/` directory in the project root
   - Download CLIP models and place them in `pretrained_model/`
   - Default path: `pretrained_model/CLIP_ViT-B-16.pt`
   - You can download CLIP models from [OpenAI CLIP](https://github.com/openai/CLIP) or [OpenCLIP](https://github.com/mlfoundations/open_clip)
   - **Note**: Pretrained models are not included in the repository due to their size

4. Download and prepare datasets:
   - Follow the instructions in the [Dataset Preparation](#-dataset-preparation) section
   - Ensure datasets are placed in the expected locations
   - **Note**: Raw dataset files are not included in the repository (see `.gitignore`)

## 📁 Dataset Preparation

### Important Note

**Datasets are not included in this repository** due to their large size and licensing restrictions. You need to download and prepare the datasets separately. The repository only includes the LLM-generated class descriptions (`datasets/class_descs/`), which are part of the project.

### Supported Datasets

- **CIFAR100**: General object classification (100 classes)
  - Download from: https://www.cs.toronto.edu/~kriz/cifar.html
  - The dataset will be automatically downloaded by torchvision when first used
  
- **ImageNet-R**: ImageNet-Rendition dataset
  - Download from: https://github.com/hendrycks/imagenet-r
  
- **ImageNet100**: Subset of ImageNet (100 classes)
  - Create a subset from ImageNet dataset
  
- **Skin40**: Skin disease classification (40 classes, subset of SD-198)
  - Download from: https://link.springer.com/chapter/10.1007/978-3-319-46466-4_13
  - Expected location: `$HOME/Data/skin40/`
  - Required files: `train_1.txt`, `val_1.txt`, and `images/` directory
  
- **CUB200**: Caltech-UCSD Birds-200-2011 (200 classes)
  - Download from: http://www.vision.caltech.edu/datasets/cub_200_2011/
  
- **Cars196**: Stanford Cars dataset (196 classes)
  - Download from: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
  
- **MedMNIST**: Medical imaging dataset
  - Download from: https://medmnist.com/

### Dataset Structure

#### Raw Dataset Files

Place your downloaded datasets according to the expected paths in each dataset's implementation file. For example:
- Skin40: `$HOME/Data/skin40/` with `train_1.txt`, `val_1.txt`, and `images/` directory
- CIFAR100: Automatically handled by torchvision
- Other datasets: Check the respective dataset class in `datasets/` for expected paths

#### Class Descriptions (Included in Repository)

Each dataset should have its class descriptions stored in `datasets/class_descs/{DATASET_NAME}/`:
- `description_pool.json`: Dictionary mapping class names to lists of LLM-generated descriptions
- `unique_descriptions.txt`: List of unique descriptions used for training

Example structure:
```
datasets/
  class_descs/
    CIFAR100/
      description_pool.json
      unique_descriptions.txt
    Skin40/
      description_pool.json
      unique_descriptions.txt
```

**Note**: The class description files are already included in this repository and do not need to be downloaded separately.

## 🎯 Usage

### Training

1. Configure your experiment in a YAML file under `config_yaml/CLIP_Concept/`:
```bash
python main.py --yaml_path config_yaml/CLIP_Concept/cifar100.yaml
```

2. Or specify parameters directly via command line:
```bash
python main.py \
    --yaml_path config_yaml/CLIP_Concept/cifar100.yaml \
    --batch_size 32 \
    --epochs 10 \
    --lr 0.002
```

### Evaluation

Set `is_train = False` in `main.py` to evaluate a trained model:
```python
is_train = False
```

The evaluation script will load checkpoints from the specified save path and evaluate on all tasks.

## ⚙️ Configuration

Configuration files are organized in YAML format under `config_yaml/CLIP_Concept/`. Key parameters include:

### Basic Settings
- `method`: Training method (e.g., "CLIP_Concept")
- `increment_type`: Type of incremental learning ("CIL" for class-incremental)
- `increment_steps`: List defining number of classes per task (e.g., `[10, 10, 10, ...]`)

### Model Settings
- `backbone`: Backbone architecture ("CLIP", "OpenCLIP", or "MedCLIP")
- `pretrained_path`: Path to pretrained model weights
- `alpha`: Weight for combining direct and attention-based logits

### Training Settings
- `batch_size`: Batch size for training
- `epochs`: Number of training epochs per task
- `lr`: Learning rate
- `optimizer`: Optimizer type (e.g., "AdamW")
- `scheduler`: Learning rate scheduler (e.g., "Cosine")

### Concept Settings
- `desc_path`: Path to class descriptions file
- `desc_num`: Number of descriptions per class
- `prompt_template`: Template for text prompts (e.g., "a photo of a {}.")
- `lambd`: Weight for attention loss

### Memory Settings (Optional)
- `memory_size`: Total size of replay memory
- `memory_per_class`: Number of exemplars per class
- `sampling_method`: Method for selecting exemplars (e.g., "herding")

### Stage 2 Training (Class-Aware Regularization)
- `ca_epoch`: Number of epochs for stage 2 training
- `ca_lr`: Learning rate for stage 2 training
- `num_sampled_pcls`: Number of samples per class for stage 2
- `ca_logit_norm`: Logit normalization factor (0 to disable)

Example configuration file:
```yaml
basic:
  random_seed: 1993
  version_name: "cifar100_b0i10"
  method: "CLIP_Concept"
  increment_type: "CIL"
  increment_steps: [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

usual:
  dataset_name: "cifar100"
  backbone: "CLIP"
  pretrained_path: "pretrained_model/CLIP_ViT-B-16.pt"
  batch_size: 32
  epochs: 10
  lr: 0.002
  optimizer: AdamW
  scheduler: Cosine

special:
  desc_path: "./datasets/class_descs/CIFAR100/unique_descriptions.txt"
  prompt_template: "a photo of a {}."
  desc_num: 3
  alpha: 0.5
  ca_epoch: 5
  ca_lr: 0.002
```

## 📂 Project Structure

```
VisualConcepts4CL/
├── config_yaml/          # Configuration files
│   └── CLIP_Concept/     # CLIP_Concept method configs
├── datasets/              # Dataset implementations
│   ├── class_descs/      # LLM-generated class descriptions
│   └── *.py              # Dataset classes
├── methods/               # Training methods
│   ├── Base.py           # Base class for methods
│   └── CLIP_Concept.py   # CLIP_Concept implementation
├── model/                 # Model architectures
│   ├── backbone/         # Backbone networks (CLIP, etc.)
│   ├── CLIP_Base_Net.py  # Base CLIP network
│   └── CLIP_Concept_Net.py  # CLIP_Concept network
├── utils/                 # Utility functions
├── config.py             # Configuration parser
├── data_manager.py       # Data management
├── main.py               # Main training/evaluation script
├── ReplayBank.py         # Replay memory implementation
└── requirements.txt      # Dependencies
```

## 📊 Evaluation Metrics

The framework evaluates models using:
- **Overall Accuracy**: Accuracy across all seen classes
- **Task Accuracy**: Accuracy per task at each incremental step
- **Mean Class Recall (MCR)**: Average recall across all classes
- **Backward Transfer**: Performance improvement on previous tasks
- **Average Forgetting**: Measure of catastrophic forgetting
- **Open-Set Metrics** (optional): AUC, FPR95, AP for open-set scenarios

## 🔬 Methodology

The method consists of two main stages:

1. **Stage 1 - Incremental Training**: 
   - Freezes most of the CLIP backbone
   - Fine-tunes the last transformer layers
   - Trains with cross-attention between image features and LLM-generated text descriptions
   - Uses a combination of classification loss and attention loss

2. **Stage 2 - Class-Aware Regularization** (optional):
   - Computes class means and covariances from training data
   - Samples synthetic features from multivariate normal distributions
   - Fine-tunes the model on synthetic features to maintain class boundaries

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{tan2024augmenting,
  title={Augmenting Continual Learning of Diseases with LLM-Generated Visual Concepts},
  author={Tan, Jiantao and Ma, Peixian and Chen, Kanghao and Dai, Zhiming and Wang, Ruixuan},
  journal={arXiv preprint arXiv:2508.03094},
  year={2024}
}
```

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🙏 Acknowledgments

- OpenAI for the CLIP model
- The open-source community for various dataset implementations
- Contributors and collaborators

## 📝 Notes

### Git Ignore

This repository uses `.gitignore` to exclude large files and datasets from version control:

- **Datasets**: Raw dataset files and images are excluded (stored in `Data/`, `data/`, or dataset-specific directories)
- **Checkpoints**: Model checkpoints and logs (stored in `checkpoint&log/`)
- **Pretrained Models**: Pretrained model weights (stored in `pretrained_model/`)
- **Python Cache**: `__pycache__/` and compiled Python files
- **IDE Files**: Editor-specific configuration files

The following are **included** in the repository:
- All source code
- Configuration files
- LLM-generated class descriptions (`datasets/class_descs/`)
- Documentation

Make sure to download datasets and pretrained models separately as described in the installation instructions.

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the authors.