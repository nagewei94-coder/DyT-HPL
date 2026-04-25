# Dynamic Topic-based Hierarchical Prompt Learning for Multi-Label Image Classification

## Description
**DyT-HPL** (Dynamic Topic-based Hierarchical Prompt Learning) is a novel framework designed to overcome the limitations of flat label supervision in multi-label image classification (MLC). 

Featuring a **dual-branch architecture with a frozen query branch**, DyT-HPL integrates offline hierarchical clustering priors with the Vision Transformer (ViT) hierarchy to extract multi-granularity semantic priors. We introduce a **transient prompt injection strategy**—dynamically injecting and subsequently truncating layer-specific prompts to achieve efficient multi-scale semantic alignment without permanent token accumulation or sequence expansion. To stabilize this dynamic routing, we formulate a **tripartite optimization objective** (combining asymmetric classification, surrogate matching, and intra-pool diversity losses) that explicitly mitigates prompt mode collapse and decouples the routing updates from the prediction process.

Extensive experiments demonstrate that DyT-HPL achieves state-of-the-art performance across MS-COCO, NUS-WIDE, and Corel5k benchmarks with minimal parameter overhead.

## Dataset Information
* **MS-COCO**: Contains 122,218 images, with 82,081 images in the training set and 40,137 images in the test set used for model evaluation. The dataset includes 80 object categories commonly found in scenes, with each image annotated with an average of 2.9 labels.
* **NUS-WIDE**: A large-scale real-world web-based image dataset containing 269,648 images and 81 visual concepts. We trained our model on 161,789 images and evaluated it on 107,859 images.
* **Corel5k**: A widely used multi-label dataset consisting of 4,999 images, with 4,500 images designated for the training set and 499 images for the validation set. The dataset is annotated with 260 categories.

## Code Information
The codebase is organized into two main directories:

* **Semantic Clustering**:
  This folder contains scripts to perform offline hierarchical clustering on label combinations from the training samples. Instead of relying on predefined rigid label graphs, it extracts a multi-granularity tiered *coarse-mid-fine* prior pool, serving as semantic guidance for the dynamic routing module.
  
* **Dythpl**
* `l2pprompt.py`: Implements the dynamic prompt routing and generation mechanisms, enabling cross-depth semantic alignment.
* `model_learn.py`: Contains the overall DyT-HPL architecture definition, including the frozen query branch, progressive prompt pool, and transient injection mechanisms.
* `losses.py`: Defines the tailored **tripartite objective** used in our approach, integrating asymmetric classification, surrogate matching, and intra-pool diversity constraints.
* `helper_functions.py`: Provides utilities for dataset preparation, DataLoader creation, and evaluation metric computation.
* `coco_dythpl.py`: Trains and evaluates DyT-HPL on the MS-COCO dataset using multi-GPU support.
* `corel5k_dythpl.py`: Trains and evaluates the model on the Corel5k dataset.
* `nus_dythpl.py`: Trains and evaluates the model on the NUS-WIDE dataset.

## Implementation Steps

### 1. Dataset Download
* **Corel5k**: All necessary files for this dataset are provided in the repository.
* **MS-COCO**: Download the COCO 2014 dataset from [MS-COCO Official Website](https://cocodataset.org/#download) and place it under the `MSCOCO` directory.
* **NUS-WIDE**: Download the Flickr folder from Kaggle and place it under the `NUS-WIDE` directory.

### 2. Ground-Truth Semantic Prior Preparation
Before extracting multi-granularity topics, we first aggregate the label co-occurrence patterns from the training set. For each dataset, extract the ground-truth label combinations of all training samples and save them as a text file. These serve as the foundational input for our semantic routing. The files used are as follows:
* `Corel5k/Corel5k/my_train_label.txt` for Corel5k
* `MSCOCO/targets2014/train/train_labels.txt` for MSCOCO
* `NUS-WIDE/mine/train_image_label.txt` for NUS-WIDE

### 3. Offline Hierarchical Semantic Clustering (`warmup_prompts.py`)
Instead of relying on predefined rigid label graphs, we execute `warmup_prompts.py` to perform offline hierarchical clustering on the aggregated label combinations. This step constructs the essential multi-granularity (coarse-mid-fine) semantic prior pool. 
For each training sample, this module assigns hierarchical priors, outputting a distribution file that maps each image ID to its specific multi-scale semantic prompt indices. 
*(Note: Pre-processed distribution files are provided in the respective target/train folders, e.g., `img_to_index2+3(300).txt`).*

### 4. DyT-HPL Training and Evaluation
This stage activates the core **Dynamic Topic-based Hierarchical Prompt Learning** framework. The model utilizes a frozen ViT query branch and transiently injects the extracted hierarchical prompts. The training process is supervised by our tailored tripartite optimization objective (defined in `losses.py`), which explicitly mitigates prompt mode collapse.

Using the Corel5k dataset as an example:
First, ensure that `args.data_corel5k` points to the `Corel5k` directory and `args.corel5k_num_class` is set to 260. Verify that the file path in the `helper_functions.py` DataLoader correctly points to the extracted prior file generated in Step 3.

Then, launch the training and evaluation pipeline using the dedicated script:
```bash
python corel5k_dythpl.py
```

## System Requirements
This code has been tested on Windows 11 / Linux with the following configuration:
* **CPU**: Xeon Gold 614 (or equivalent)
* **GPU**: Tesla V100 16G / RTX 3090

## Requirements
* `python == 3.9.0`
* `pytorch == 2.0.0`
* `timm == 0.4.12`
* `torchvision == 0.15.0`
* `randaugment`
