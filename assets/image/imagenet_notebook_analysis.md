# ImageNet Image Classification — Comprehensive Notebook Analysis

---

## PART 1: NOTEBOOK DESCRIPTION

---

### Overview

This notebook presents a comprehensive study of image classification on a 20-class subset of ImageNet. It benchmarks **7 models** spanning three architectural families — CNN, ResNet, and Vision Transformer (ViT) — across two training regimes (from scratch vs. pretrained fine-tuning). The study is end-to-end, covering exploratory data analysis, model definition, training pipelines, robustness testing, interpretability, ensemble methods, ablation studies, and a final summary dashboard. The notebook is well-structured and intended as a reference-grade experiment, not merely a quick prototype.

---

### Section 1: Setup & Imports

This section installs and imports all required dependencies. Key libraries include:

- **PyTorch + torchvision**: Core deep learning framework, model zoo (ResNet-50, ResNet-101, ViT-B/16, ViT-L/16).
- **timm**: Hugging Face's model library, imported but used as a supplement.
- **einops**: Tensor rearrangement, critical for the ViT patch embedding implementation.
- **scikit-learn**: Evaluation metrics (accuracy, F1, AUC, confusion matrix, calibration).
- **matplotlib / seaborn**: Visualization throughout.
- **tqdm**: Progress bars for training loops.

The hardware used is a **Tesla T4 GPU with 15.6 GB VRAM** (Kaggle environment). Reproducibility is enforced with `SEED=42` applied to Python, NumPy, and PyTorch. Output directories `./figures/` and `./saved_models/` are created for storing all plots and model weights.

**Comment**: The environment uses Kaggle's competition download API (`kagglehub`) for the ImageNet dataset, which means the data pipeline is specific to that platform. The `savefig()` helper function consistently saves all figures with `dpi=120`, ensuring print-quality outputs.

---

### Section 2: Hyperparameter Configuration

All hyperparameters are centralized in a single cell, making the notebook highly reproducible and easy to tune. Key parameters include:

| Category | Parameter | Value |
|---|---|---|
| Dataset | NUM_CLASSES | 20 |
| Dataset | SAMPLES_PER_CLASS | 500 |
| Dataset | Train/Val/Test split | 70% / 15% / 15% |
| Image | IMG_SIZE | 224×224 |
| Training | EPOCHS (scratch) | 30 |
| Training | FINETUNE_EPOCHS (pretrained) | 5 |
| Training | BATCH_SIZE | 16 |
| Training | LR_SCRATCH | 1e-3 |
| Training | LR_FINETUNE | 1e-4 |
| Training | WEIGHT_DECAY | 1e-4 |
| Training | LABEL_SMOOTHING | 0.1 |
| Scheduler | SCHEDULER_TYPE | cosine |
| Scheduler | WARMUP_EPOCHS | 3 |
| MixUp/CutMix | MIXUP_ALPHA | 0.4 |
| MixUp/CutMix | CUTMIX_ALPHA | 1.0 |
| CNN (scratch) | CNN_BASE_CHANNELS | 64 |
| CNN (scratch) | CNN_DROPOUT | 0.4 |
| ResNet (scratch) | RESNET_LAYERS | [2, 2, 2, 2] (ResNet-18-like) |
| ViT (scratch) | VIT_PATCH_SIZE | 16 |
| ViT (scratch) | VIT_DIM | 384 |
| ViT (scratch) | VIT_DEPTH | 8 blocks |
| ViT (scratch) | VIT_HEADS | 6 heads |

**Comment**: The choice of 20 classes and 500 samples/class results in a total dataset of only 10,000 images (7,000 train / 1,500 val / 1,500 test). This is an intentionally small subset, appropriate for fair comparison across many models under compute constraints. The small dataset size, however, heavily disadvantages scratch ViTs (which are data-hungry), a pattern clearly visible in the results. The FINETUNE_EPOCHS=5 is very low, but given that the pretrained backbone already has strong ImageNet features, even a few epochs suffice for head adaptation.

---

### Section 3: Rich EDA — Exploratory Data Analysis

This section provides an unusually thorough EDA for an image classification task, with eight distinct sub-analyses:

#### 3.0 Dataset Loading
The ImageNet competition data is downloaded from Kaggle. A custom `BalancedImageNetSubset` class loads synset directories and maps them to human-readable labels via `LOC_synset_mapping.txt`. Classes are selected with a stride (`step=5`) to ensure diversity across the 1,000 class space. The 20 selected classes are: tench, brambling, water ouzel, bullfrog, American chameleon, American alligator, night snake, harvestman, black grouse, lorikeet, black swan, and others.

#### 3.1 Class Distribution
Counts images per class. **Result**: Perfect class balance (`std=0.0`), confirming the balanced sampling procedure. This eliminates class imbalance as a confound in the comparison.

#### 3.2 Sample Grid
Displays 5 images per class for the first 10 classes in a 10×5 grid. **Comment**: This visualization reveals significant intra-class visual diversity (e.g., frogs photographed in water vs. on land) and inter-class overlap (snakes and alligators share similar textures).

#### 3.3 Channel Statistics
Computes per-pixel mean and std across channels empirically from the subset:
- Computed mean (R,G,B): [0.4709, 0.454, 0.3866]
- Standard ImageNet mean: [0.485, 0.456, 0.406]

The computed values are close but not identical to the standard constants. The notebook opts to use the standard constants for consistency with pretrained model normalization, which is the correct choice. **Comment**: Using subset-computed statistics for scratch models while using standard statistics for pretrained models would actually be slightly more principled, but the difference is negligible.

#### 3.4 Per-Class Mean Images
Computes the pixel-wise mean of 100 images per class. **Comment**: High-contrast mean images (e.g., "lorikeet" — a colorful bird) indicate strong spatial and color consistency within the class. Blurry, nondescript mean images (e.g., "harvestman" — a spider-like creature in varied environments) suggest high intra-class variance and likely harder classification difficulty.

#### 3.5 Pixel Intensity Histograms
Plots RGB channel intensity distributions for each of the first 10 classes. **Comment**: Classes like "black swan" would be expected to show a left-skewed (dark) distribution, while outdoor animal shots tend to have more balanced channel distributions. This analysis is useful for detecting color-biased classes.

#### 3.6 Inter-Class Cosine Similarity Heatmap
Computes cosine similarity between per-class mean pixel vectors. Similarity values range within `[0.7, 1.0]`. **Comment**: High off-diagonal similarity (close to 1.0) between two classes reveals pairs that are visually confusable in terms of average color/texture, predicting which class pairs will produce the most confusion matrix errors. Classes like "bullfrog" and "American alligator" (both greenish, aquatic animals) likely show elevated similarity.

#### 3.7 Per-Class Variance Maps
Computes pixel-wise variance across images within each class, averaged across channels. Hot-colored regions represent areas of high within-class variance. **Comment**: This map is directly interpretable as "background clutter" — the model should ideally learn to focus on low-variance (consistent) regions, which correspond to the actual object. Classes with high variance throughout the entire image are particularly hard for CNNs relying on texture/spatial statistics.

#### 3.8 Augmentation Preview
Visualizes 12 augmentation strategies on a single sample: Original, RandomCrop, HorizontalFlip, ColorJitter, RandomRotation, GaussianBlur, Grayscale, RandomPerspective, RandAugment, Brightness ×1.5, Brightness ×0.4, and AugMix. **Comment**: This is a useful sanity check that augmentations behave as expected and that RandAugment does not produce degenerate transforms.

---

### Section 4: Dataset Preparation

#### Transforms
Training uses an aggressive augmentation pipeline:
- `RandomResizedCrop(224, scale=(0.6, 1.0))` — crops 60–100% of the image area
- `RandomHorizontalFlip()`
- `ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)`
- `RandAugment(num_ops=2, magnitude=9)`
- `Normalize(IMAGENET_MEAN, IMAGENET_STD)`

Validation and test use only `Resize(256) → CenterCrop(224) → Normalize`.

#### Split Results
- Train: 7,000 images
- Val: 1,500 images
- Test: 1,500 images
- Classes: 20

**Comment**: The small training set (350 images/class) is a key constraint. Aggressive augmentation is essential here. However, this also means that augmentation strategies like CutMix/MixUp may actually hurt early-epoch performance (due to label mixing making the task harder before the model has learned basic features), which is exactly what the ablation study later confirms.

---

### Section 5: Model Definitions — Scratch Implementations

#### 5.1 ScratchCNN (DeepCNN)
A 7-block custom CNN with:
- **Stem**: 7×7 conv (stride 2) → BN → ReLU → MaxPool (56×56)
- **6 ConvBlocks**: each contains two 3×3 convolutions with BN+ReLU, optional MaxPool, and a residual skip connection via 1×1 conv when channels change
- Channel progression: 64 → 128 → 256 → 256 → 512 → 512 → 512
- **Global Average Pooling** → FC(512, 512) → Dropout → FC(512, num_classes)
- **Parameters**: 15,726,676 (~15.7M)

**Comment**: This is more than a standard CNN — it incorporates residual connections at every ConvBlock, effectively making it a non-standard ResNet-like architecture. The `Dropout2d` after spatial blocks (dropout=0.4) is aggressive; this likely accounts for the model's relatively strong generalization despite having fewer parameters than a ResNet-50.

#### 5.2 ScratchResNet
A faithful ResNet-18 implementation using `BasicBlock` (two 3×3 convs with BN) and `Bottleneck` (1×1 → 3×3 → 1×1) classes. Layer config `[2, 2, 2, 2]` matches ResNet-18 exactly. The stem mirrors the standard ImageNet ResNet (7×7, stride-2, MaxPool).
- **Parameters**: 11,186,772 (~11.2M)

**Comment**: ResNet-18 is well-suited for small datasets due to its moderate capacity and strong inductive biases (local receptive fields, skip connections). Its parameter count is actually the smallest of the three scratch architectures, yet it achieves the best scratch performance (70.3% accuracy), which demonstrates the superiority of the ResNet's well-tuned architectural priors.

#### 5.3 ScratchViT
A standard ViT implementation with:
- Patch embedding via `einops.Rearrange` (patch_size=16 → 196 patches from 224×224)
- `LayerNorm → Linear → LayerNorm` patch projection
- Learnable CLS token and positional embeddings
- 8 `TransformerBlock`s, each with: Pre-Norm → Multi-Head Self-Attention → residual + Pre-Norm → MLP(GELU)
- MHA returns attention weights for interpretability
- `forward(return_attn=True)` mode for attention rollout visualization
- **Parameters**: 14,568,596 (~14.6M)

**Comment**: The ViT-Scratch config is approximately ViT-Small. The critical limitation is that ViTs require large amounts of data to learn inductive biases that CNNs have baked in by design. With only 7,000 training images, the scratch ViT essentially cannot learn meaningful patch relationships, resulting in near-random accuracy (14.7%). This is a well-known finding in the literature and the notebook confirms it clearly.

---

### Section 6: Pretrained Models

Four torchvision pretrained models are loaded and adapted for 20-class classification:

| Model | Pretrained Weights | Head Replacement | Parameters |
|---|---|---|---|
| ResNet-50-PT | `IMAGENET1K_V2` | `fc` → Linear(2048, 20) | 23,549,012 (~23.5M) |
| ResNet-101-PT | `IMAGENET1K_V2` | `fc` → Linear(2048, 20) | 42,541,140 (~42.5M) |
| ViT-B/16-PT | `IMAGENET1K_V1` | `heads.head` → Linear(768, 20) | 85,814,036 (~85.8M) |
| ViT-L/16-PT | `IMAGENET1K_V1` | `heads.head` → Linear(1024, 20) | 303,322,132 (~303.3M) |

**Fine-tuning strategy (two-phase)**:
- **Phase 1** (≈ 1/3 of FINETUNE_EPOCHS): Freeze backbone, train only the new classification head at `LR_FINETUNE=1e-4`.
- **Phase 2** (remaining epochs): Unfreeze all layers, fine-tune at `LR_FINETUNE_FULL=5e-5`.

**Comment**: The two-phase strategy is sound and prevents catastrophic forgetting of pretrained features, particularly important when FINETUNE_EPOCHS=5 is very short. Note that `IMAGENET1K_V2` weights for ResNets are higher-quality (trained with improved augmentation) than V1. The choice of V1 for ViTs is due to availability — ViT-B/16 and ViT-L/16 V2 weights are not available in older torchvision versions.

---

### Section 7: Training Pipeline

#### Loss Functions
- **Primary**: `CrossEntropyLoss(label_smoothing=0.1)` for all models.
- **FocalLoss**: Defined (gamma=2) but not used as the primary loss; available as an option.

#### Data Augmentation During Training
- **MixUp**: `Beta(alpha, alpha)` mixing, applied with probability 50% (vs CutMix).
- **CutMix**: Random rectangular patch replacement between two images.
- Both use a `mixup_criterion` that combines the loss for both mixed labels proportionally.

**Per-model augmentation choices**:
- CNN-Scratch: MixUp only
- ResNet-Scratch: CutMix only
- ViT-Scratch: CutMix only
- Pretrained models: No MixUp/CutMix (standard training)

#### Optimizer & Scheduler
- `AdamW` with cosine annealing LR scheduler.
- Gradient clipping at `max_norm=1.0`.
- Best model state is saved based on validation accuracy.

#### Training Logs (Key Results)

**CNN-Scratch (30 epochs)**:
- Ep 1: train_acc=0.0653, val_acc=0.1187
- Ep 10: train_acc=0.1229, val_acc=0.2773
- Ep 15: train_acc=0.1486, val_acc=0.36
- Training time: ~23 min

**ResNet-Scratch (30 epochs)**:
- Training time: ~22 min

**ViT-Scratch (30 epochs)**:
- Training time: ~27.8 min (slowest scratch model due to attention computation)

**Comment**: The training logs reveal CNN-Scratch converges slowly — train accuracy is still below 15% at epoch 15, suggesting the aggressive augmentation (MixUp on a small dataset) substantially slows learning. The gap between training and validation accuracy in early epochs is typical for heavily augmented training.

---

### Section 8: Results Analysis

#### Final Test Set Metrics (Full Table)

| Model | Type | Test Acc | Macro F1 | Macro AUC | Params (M) | Train (min) | Inf (s) | Inf/Sample (ms) |
|---|---|---|---|---|---|---|---|---|
| ViT-L16-PT | Pretrained | **1.0000** | **1.0000** | **1.0000** | 303.32 | 70.6 | 53.30 | 35.53 |
| ViT-B16-PT | Pretrained | 0.9980 | 0.9979 | 1.0000 | 85.81 | 22.0 | 19.77 | 13.18 |
| ResNet-101-PT | Pretrained | 0.9933 | 0.9935 | 0.9999 | 42.54 | 8.5 | 8.28 | 5.52 |
| ResNet-50-PT | Pretrained | 0.9913 | 0.9913 | 0.9998 | 23.55 | 5.7 | 5.28 | ~3.5 |
| ResNet-Scratch | Scratch | 0.7033 | 0.6978 | 0.9688 | 11.19 | 22.2 | 4.46 | ~3.0 |
| CNN-Scratch | Scratch | 0.4580 | 0.4338 | 0.9108 | 15.73 | 23.0 | 4.74 | ~3.2 |
| ViT-Scratch | Scratch | **0.1467** | **0.1161** | **0.6822** | 14.57 | 27.8 | 5.32 | ~3.5 |

**Key observations**:

1. **Pretrained vs. scratch gap is enormous**: The best pretrained model (ViT-L16-PT, 100%) is 29+ percentage points above the best scratch model (ResNet-Scratch, 70.3%). This demonstrates the transformative power of pretraining on the full ImageNet-1K before fine-tuning on a subset.

2. **ResNet-Scratch outperforms CNN-Scratch by 24.5%** despite having *fewer parameters* (11.2M vs. 15.7M). This highlights that the ResNet's clean modular design and skip connections provide stronger gradient flow and regularization than the custom CNN architecture.

3. **ViT-Scratch nearly fails** (14.7% accuracy, barely above the random baseline of 5% for 20 classes). This is a textbook demonstration that ViTs require large-scale data to learn their spatial inductive biases from scratch.

4. **AUC vs. Accuracy discrepancy for ViT-Scratch**: Despite accuracy of 14.7%, the AUC is only 0.6822 (vs. 0.9108 for CNN-Scratch). This indicates the ViT-Scratch model's probability scores are largely uncalibrated and not well-ordered.

5. **Efficiency**: ResNet-50-PT is the most efficient pretrained model — only 23.5M params, 5.7 min training, 3.5ms/sample inference, yet achieves 99.1% accuracy. ViT-L16-PT achieves 100% but at 303.3M params and 35.5ms/sample — a 10× inference cost for a 0.87% absolute gain.

---

### Section 9: Augmentation & Robustness Study

#### 9.1 Augmentation Study
Three strategies were compared on scratch models over 5 epochs: No Augmentation, Standard Aug (training pipeline), and RandAugment.

**5-epoch results**:
- CNN-Scratch: No Aug (0.3973) > Standard Aug (0.2587) ≈ RandAugment (0.2593)
- ResNet-Scratch: No Aug (0.5413) > Standard Aug (0.4153) ≈ RandAugment (0.4133)
- ViT-Scratch: No Aug (0.1240) < Standard Aug (0.1427) > RandAugment (0.1133)

**Comment**: Augmentation *hurts* short-term performance for CNN and ResNet (5 epochs is too few to benefit from regularization). This is expected — augmentation is a regularizer and only helps when the model has enough epochs to adapt. At 5 epochs, it effectively reduces training signal quality. No-augmentation training shows faster early convergence. Over the full 30 epochs, augmentation would be expected to close the gap and improve generalization. The ViT result is anomalous and noisy due to near-random performance.

#### 9.2 Gaussian Noise Robustness
Test accuracy measured at noise levels σ ∈ {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75}:

| Model | σ=0 | σ=0.1 | σ=0.3 | σ=0.5 | σ=0.75 |
|---|---|---|---|---|---|
| CNN-Scratch | 0.458 | 0.455 | 0.421 | 0.326 | 0.211 |
| ResNet-Scratch | 0.703 | 0.695 | 0.568 | 0.413 | 0.250 |
| ViT-Scratch | 0.147 | 0.145 | 0.147 | 0.149 | 0.148 |
| ResNet-50-PT | 0.991 | 0.991 | 0.983 | 0.945 | 0.843 |
| ResNet-101-PT | similar to ResNet-50-PT | | | | |

**Comment**: Pretrained models are dramatically more noise-robust. At σ=0.5 (heavy noise), ResNet-50-PT maintains 94.5% accuracy vs. ResNet-Scratch at 41.3%. The ViT-Scratch result is peculiar — its accuracy barely changes across noise levels (ranging 14.7–14.9%), because it's essentially guessing randomly regardless of noise, making it "immune" to noise in a degenerate sense.

#### 9.3 Brightness Robustness
Test accuracy at brightness factors ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5}:

| Model | ×0.0 | ×0.2 | ×0.4 | ×0.8 | ×1.0 | ×1.5 |
|---|---|---|---|---|---|---|
| CNN-Scratch | 0.039 | 0.121 | 0.373 | 0.449 | 0.458 | 0.431 |
| ResNet-Scratch | 0.047 | 0.254 | 0.587 | 0.693 | 0.703 | 0.679 |
| ResNet-50-PT | 0.047 | 0.989 | 0.989 | 0.991 | 0.991 | 0.990 |

**Comment**: All models collapse at brightness=0 (pure black image), which is expected since all information is destroyed. The dramatic finding is that pretrained ResNet-50 maintains near-perfect accuracy from ×0.2 to ×1.5, while scratch models show steep degradation below ×0.6. This indicates pretrained models have learned highly robust feature representations that generalize across lighting conditions.

#### 9.4 Robustness Summary Heatmap
Combines both robustness experiments into side-by-side heatmaps for all 7 models. The heatmap format makes the gradient of degradation immediately visual.

---

### Section 10: Interpretability — Grad-CAM & Attention Maps

#### Grad-CAM
A custom `GradCAM` class is implemented using PyTorch hooks:
1. A **forward hook** captures feature activations at the target layer.
2. A **backward hook** captures gradient flows.
3. The CAM is computed as `relu(sum over channels of [weight_c × activation_c])`.
4. The CAM is upsampled to the input resolution via bilinear interpolation.

Applied to CNN-Scratch (last ConvBlock), ResNet-Scratch (layer4), and ResNet-50-PT (layer4). Each visualization shows: original image, Grad-CAM heatmap only, and overlay (50% alpha blend).

**Comment**: Grad-CAM on ResNet-50-PT typically produces tight, object-focused activations — the model attends to the actual animal/object. Scratch models tend to produce more diffuse activations, reflecting less discriminative features. Misclassified samples (shown in red) often show Grad-CAM attending to background regions, which is a common failure mode.

#### Attention Rollout (ViT-Scratch)
**Attention rollout** (Abnar & Zuidema, 2020) is implemented:
1. For each transformer block, the attention matrix is computed (averaged across heads).
2. Identity matrix is added to model residual connections.
3. Row-normalize the combined matrix.
4. Multiply attention matrices sequentially across all 8 blocks.
5. Extract the row corresponding to the CLS token and reshape to 14×14 (for 16×16 patches).
6. Upsample to 224×224 via bilinear interpolation.

**Comment**: For the ViT-Scratch model, attention rollout maps are expected to be nearly random/uniform since the model has not learned meaningful attention patterns with only 7,000 training images. In contrast, attention maps from well-trained ViTs typically highlight the main object clearly.

---

### Section 11: Error Analysis & Calibration

#### 11.1 Top Confused Class Pairs
For CNN-Scratch, the top confused pairs are:
- True=American Staff → Predicted=Afghan hound (29 samples)
- True=Irish wolfhound → Predicted=Afghan hound (23 samples)
- True=bullfrog → Predicted=American alligator (22 samples)
- True=water ouzel → Predicted=red-backed sandpiper (21 samples)

**Comment**: These confusions are semantically meaningful — dog breeds are visually similar (American Staffordshire, Irish Wolfhound, Afghan Hound all have four legs and fur). The bullfrog/alligator confusion is notable — both are aquatic animals with green/brown coloration. This kind of error analysis provides insight into whether the model is making "reasonable" mistakes or completely arbitrary ones.

#### 11.2 Confidence Distribution
For each model, the distribution of maximum softmax probability is plotted separately for correctly and incorrectly classified samples. **Comment**: Well-calibrated, high-accuracy models (ViT-L16-PT) should show correct predictions concentrated near probability=1.0 and incorrect predictions near 0.5–1.0 (overconfident errors). Poorly performing models like CNN-Scratch may show incorrect predictions spread across the entire confidence range.

#### 11.3 Expected Calibration Error (ECE) & Reliability Diagrams
ECE is computed with 15 bins. Results (lower is better):

| Model | ECE |
|---|---|
| ViT-Scratch | 0.0386 |
| ViT-B16-PT | 0.1018 |
| ResNet-101-PT | 0.1048 |
| ResNet-Scratch | 0.1104 |
| ResNet-50-PT | 0.1145 |
| ViT-L16-PT | 0.1238 |
| CNN-Scratch | 0.1837 |

**Comment**: ViT-Scratch has the **best ECE (0.039)** — a paradoxical result. Since the model is essentially random, it assigns low, uniformly-spread confidence values, which happen to be well-calibrated (low confidence = low accuracy). CNN-Scratch has the worst ECE (0.184), indicating it is significantly overconfident on its incorrect predictions. All pretrained models have moderate ECE around 0.10–0.12, likely due to the label smoothing (0.1) slightly widening the predicted distributions.

---

### Section 12: Ensemble Methods

All valid combinations of scratch and pretrained models are evaluated as **logit-averaging ensembles**. The combinations include: scratch-only subsets, pretrained-only subsets, and same-architecture scratch+pretrained pairs.

#### Top Ensemble Results

| Combination | Size | Acc | Macro F1 | Gain vs Best Single |
|---|---|---|---|---|
| ResNet-101-PT + ViT-L16-PT | 2 | 0.9993 | 0.9994 | -0.0007 |
| ViT-B16-PT + ViT-L16-PT | 2 | 0.9993 | 0.9994 | -0.0007 |
| ResNet-50-PT + ViT-B16-PT + ViT-L16-PT | 3 | 0.9993 | 0.9994 | -0.0007 |
| Scratch-only ensemble (all 3) | 3 | 0.6720 | 0.6663 | — |
| Pretrained-only ensemble (all 4) | 4 | 0.9987 | 0.9987 | — |
| Mixed ensemble (all 7) | 7 | 0.9960 | 0.9958 | — |

**Comment**: Ensembles of pretrained models cannot beat the single best model (ViT-L16-PT at 100%). This is because ViT-L16-PT is already near-perfect, so averaging with other models can only introduce noise. The mixed ensemble (all 7 models) actually performs *worse* than the best pretrained single model — the scratch models act as noise when combined with near-perfect pretrained models. This is a key insight: ensemble diversity is only beneficial when constituent models have complementary strengths. Combining a high-accuracy model with low-accuracy models degrades ensemble performance.

---

### Section 13: Ablation Study

#### 13.1 Label Smoothing
CNN-Scratch trained for 10 epochs with ε ∈ {0.0, 0.1, 0.2}. **Comment**: In short training runs on small datasets, the difference between smoothing levels is typically small. ε=0.1 is generally recommended as a regularizer that prevents the model from becoming overconfident.

#### 13.2 MixUp vs CutMix (ResNet-Scratch, 5 epochs)
Results (best validation accuracy over 5 epochs):
- No MixUp/CutMix: **0.4160** (best)
- MixUp only: 0.3313
- CutMix only: 0.3320
- Both (random): 0.3140

**Comment**: This result strongly confirms the observation from Section 9.1 — on a small dataset with few training epochs, MixUp and CutMix *hurt* performance. The label mixing makes each training step harder than pure label training, requiring more epochs to converge. In the main experiment (30 epochs), the regularization benefit would likely appear. This ablation should ideally be run for 20–30 epochs to get the full picture.

#### 13.3 ViT Patch Size
Commented out (to save compute). Would test patch sizes 8, 16, 32. Patch size 8 gives 784 tokens (4× more than patch 16), which dramatically increases compute but may improve accuracy.

#### 13.4 Ensemble Diversity
- Scratch-only ensemble: 0.672
- Pretrained-only ensemble: 0.999
- Mixed (all models): 0.996

**Comment**: The pretrained-only ensemble is better than the mixed ensemble. This conclusively shows that including scratch models (which have 14–70% accuracy) in an ensemble with near-perfect pretrained models dilutes the ensemble's performance.

---

### Section 14: Final Summary Dashboard

A comprehensive 6-panel figure:
- Panel 1: Test accuracy bar chart (all models)
- Panel 2: Macro F1 bar chart
- Panel 3: ECE bar chart
- Panel 4: Parameters vs. Accuracy bubble scatter (bubble size ∝ inference time)
- Panel 5: Per-class accuracy heatmap (all models × all classes)
- Panel 6: Training time vs. accuracy scatter

**Final Results Summary**:

```
Model              Type          Acc     F1    AUC   Params(M)  Train(min)  Inf(s)  ECE
ViT-L16-PT         Pretrained  1.0000 1.0000 1.0000    303.32      70.6    53.30  0.1238
ViT-B16-PT         Pretrained  0.9980 0.9979 1.0000     85.81      22.0    19.77  0.1018
ResNet-101-PT      Pretrained  0.9933 0.9935 0.9999     42.54       8.5     8.28  0.1048
ResNet-50-PT       Pretrained  0.9913 0.9913 0.9998     23.55       5.7     5.28  0.1145
ResNet-Scratch     Scratch     0.7033 0.6978 0.9688     11.19      22.2     4.46  0.1104
CNN-Scratch        Scratch     0.4580 0.4338 0.9108     15.73      23.0     4.74  0.1837
ViT-Scratch        Scratch     0.1467 0.1161 0.6822     14.57      27.8     5.32  0.0386
```

**Key takeaways from the notebook**:
- Best scratch model: **ResNet-Scratch** (70.33%)
- Best pretrained model: **ViT-L16-PT** (100%)
- Best ensemble: **ResNet-101-PT + ViT-L16-PT** (99.93%)
- Most parameter-efficient: **ResNet-Scratch** (11.19M params)
- Best calibrated: **ViT-Scratch** (ECE=0.039, degenerate reason)
- Fastest inference: **ResNet-Scratch** (4.46s for 1,500 test images)

---

---

## PART 2: FIGURE DESCRIPTIONS

---

Below is a description of every figure saved to `./figures/` by the notebook, listed in the order they are generated.

---

### `eda_class_distribution.png`

A horizontal bar chart (or vertical bar chart) showing the number of images per class across the 20 ImageNet classes in the EDA subset. Each bar is colored with a distinct color from the `tab20` palette. The x-axis represents class index (0–19) and the y-axis represents image count. The reported standard deviation of class counts is 0.0, confirming perfectly balanced sampling — all bars are exactly equal height. The title reads "Class Distribution — ImageNet (20 classes, EDA subset)". Figure size: 14×5 inches.

---

### `eda_samples.png`

A 10-row × 5-column grid of sample images from the first 10 ImageNet classes. Each row corresponds to one class, with 5 randomly selected images displayed side-by-side. The class name (up to 15 characters) is shown as the y-axis label on the leftmost image. Images are displayed as raw RGB tensors (no normalization applied). The title reads "ImageNet — 5 Samples per Class (first 10 classes)". Figure size: 12×24 inches.

---

### `eda_mean_images.png`

A 2-row × 5-column grid showing the per-class mean image for the first 10 ImageNet classes. Each mean image is computed by averaging 100 images per class along the batch dimension. Values are min-max normalized to [0, 1] for display. High-contrast mean images indicate classes with strong spatial consistency (e.g., birds always photographed at the center); blurry/indistinct mean images indicate high intra-class diversity. Class names are shown above each panel. Figure size: 15×6 inches.

---

### `eda_pixel_histograms.png`

A 2-row × 5-column grid of RGB pixel intensity histograms for the first 10 ImageNet classes. Each subplot overlays three histograms: R (red, `#e74c3c`), G (green, `#2ecc71`), and B (blue, `#3498db`), each with 32 bins and density normalization. The x-axis is pixel intensity [0, 1] and the y-axis is density. Classes dominated by particular colors (e.g., outdoor ground animals with lots of green backgrounds) will show dominant channel spikes. A legend (RGB) appears on the first subplot only. Figure size: 18×7 inches.

---

### `eda_class_similarity.png`

A 20×20 annotated heatmap showing the pairwise cosine similarity between per-class mean pixel vectors (computed from 100 images per class). Values are in the range [0.7, 1.0] with a `coolwarm` colormap (blue = lower similarity, red = higher similarity). Each cell shows the numeric similarity value to 2 decimal places. Class names (up to 12 characters) are shown as both row and column tick labels, rotated 45° on the x-axis. The diagonal is always 1.0 (self-similarity). Off-diagonal high values indicate visually confusable class pairs. Figure size: 12×10 inches.

---

### `eda_variance_maps.png`

A 2-row × 5-column grid of per-class pixel variance maps for the first 10 ImageNet classes. Each map is computed as the pixel-wise variance across 100 images, averaged across the 3 color channels, and displayed as a grayscale/heatmap image using the `hot` colormap (brighter = higher variance). High-variance regions correspond to backgrounds and object poses that vary across images; low-variance regions correspond to consistent object locations or textures. Figure size: 15×6 inches.

---

### `eda_augmentation_preview.png`

A 3-row × 4-column grid (12 panels total) showing the same source image after 12 different augmentation transforms: Original, RandomCrop, HorizontalFlip, ColorJitter, RandomRotation(15°), GaussianBlur, Grayscale, RandomPerspective, RandAugment, BrightnessBoost×1.5, BrightnessDim×0.4, and AugMix. Each panel is labeled with the augmentation name in bold. The figure demonstrates the visual effect of each transform on the same input, serving as a qualitative sanity check. Figure size: 16×12 inches.

---

### `training_curves.png`

A two-panel figure (side-by-side) showing training and validation curves for all 7 models over all epochs. Left panel shows **Loss** (y-axis) vs. Epoch (x-axis); right panel shows **Accuracy** (y-axis) vs. Epoch (x-axis). For each model, a **dashed line** represents training metrics and a **solid line** represents validation metrics, both in the same color. All 7 models are shown with unique colors from the `tab10` palette. A shared legend lists all model names. The pretrained models (only 5 epochs) appear as short curves on the left side of the x-axis; scratch models span 30 epochs. Figure size: 18×6 inches.

---

### `efficiency_plot.png`

A two-panel scatter plot analyzing model efficiency:

- **Left panel**: X-axis = Model parameters (in millions); Y-axis = Test accuracy. Bubble size is proportional to inference time. Scratch models use circle markers (`o`); pretrained models use triangle markers (`^`). Each point is annotated with the model name.
- **Right panel**: X-axis = Training time (minutes); Y-axis = Test accuracy. All bubbles are the same size (150 pt²). Same marker convention.

Both panels show a clear cluster of pretrained models at the top-right (high accuracy, high params/training time) and scratch models spread at lower accuracy. Legend distinguishes Scratch vs. Pretrained marker types. Figure size: 18×6 inches.

---

### `confusion_matrices.png`

A grid of 7 normalized confusion matrices (one per model), arranged in 3 rows × 3 columns (one cell unused). Each matrix is row-normalized (true class axis) to show per-class recall — values represent the fraction of true-class samples predicted as each class. Color scale is `Blues` (0=white, 1=dark blue). Class names (up to 10 characters) appear as both x (Predicted) and y (True) tick labels, rotated 45° on x-axis. Each subplot title includes the model name and its test accuracy. Figure size: 21×18 inches. The pretrained models should show strong diagonals (near-perfect recall for all classes), while scratch models, especially ViT-Scratch, will show highly diffuse off-diagonal errors.

---

### `per_class_accuracy.png`

A single large heatmap with models as rows (7 rows) and classes as columns (up to 40 columns). Color scale is `RdYlGn` (Red=0, Yellow=0.5, Green=1.0), showing per-class recall for each model. Class names (up to 10 characters) are shown on the x-axis, rotated 45°. This visualization immediately reveals which classes are consistently hard for all models (column-wise low values) vs. which models fail on specific classes. Pretrained models should show nearly all green; scratch CNN and ViT will show significant red/yellow. Figure size: 16×8 inches.

---

### `roc_curves.png`

A 3-row × 3-column grid of ROC curve plots (one per model, one cell unused). Each subplot shows individual per-class OvR (One vs. Rest) ROC curves as thin, semi-transparent colored lines, plus a bold black macro-average ROC curve. A diagonal dashed line represents the random classifier baseline. The legend in each subplot reports the macro AUC value. Figure size: 18×15 inches. Pretrained models will have macro AUC ≈ 1.0 with tight curves pressed to the upper-left corner; scratch models will show wider spread with the macro average curve closer to (but still above) the diagonal.

---

### `scratch_vs_pretrained.png`

A three-panel horizontal bar chart comparing all 7 models on three metrics: Test Accuracy (left), Macro F1 (center), and Parameters in Millions (right). Scratch models are colored **blue** (`#3498db`); pretrained models are colored **red** (`#e74c3c`). Models are sorted by value in each panel. Each bar is annotated with its exact value. An inverted y-axis ensures the top-performing model appears at the top. Figure size: 18×6 inches.

---

### `augmentation_study.png`

A 1-row × 3-column figure showing validation accuracy curves over 5 training epochs for each of the 3 scratch models (CNN-Scratch, ResNet-Scratch, ViT-Scratch), one per column. Each subplot contains three curves: No Augmentation (red, `#e74c3c`), Standard Aug (green, `#2ecc71`), and RandAugment (blue, `#3498db`). Curves are plotted epoch-by-epoch with circular markers. Shared y-axis scale. The figure demonstrates that No Augmentation converges faster in early epochs, while augmented strategies yield slower initial convergence. Figure size: 18×5 inches.

---

### `robustness_gaussian_noise.png`

A single line plot showing test accuracy (y-axis) vs. Gaussian noise standard deviation σ (x-axis, range 0.0 to 0.75) for all 7 models simultaneously. Each model has a unique color. Pretrained models are drawn with solid lines and heavier weight (lw=2.5); scratch models with dashed lines (lw=1.5). All curves are marked with circular markers (`o`) at each σ level. A legend identifies all models. Pretrained models show relatively flat curves at high accuracy until σ≈0.5–0.75 where they begin to drop; scratch CNN and ResNet show gradual monotonic decline; ViT-Scratch shows essentially a flat, near-random line throughout all σ levels. Figure size: 11×5 inches.

---

### `robustness_brightness.png`

A single line plot showing test accuracy (y-axis) vs. brightness multiplication factor (x-axis, range 0.0 to 1.5) for all 7 models. Same visual convention as the noise plot, except markers are squares (`s`) and a vertical dashed gray line marks the original brightness level (factor=1.0). All models collapse to near-zero at factor=0.0 (pure black image). Pretrained models maintain near-perfect accuracy from 0.2× to 1.5×, while scratch models show a steep performance cliff for factors below 0.6×. Figure size: 11×5 inches.

---

### `robustness_heatmap.png`

A two-panel heatmap (side-by-side) summarizing robustness results:

- **Left panel** (Gaussian Noise): Rows = 7 models, columns = 7 noise levels (σ=0.0 to 0.75). Cell values are annotated accuracy values formatted to 3 decimal places. Colormap is `RdYlGn`.
- **Right panel** (Brightness): Same layout with 7 brightness factors instead.

This is the most information-dense robustness visualization, allowing direct comparison of all 7 models × 7 perturbation levels in a compact form. Figure size: 20×5 inches.

---

### `gradcam_cnn-scratch.png`

A 3-row × 6-column interpretability figure for CNN-Scratch:

- **Row 1 (Original)**: The raw un-normalized test images for 6 selected samples. Each image is titled with ground truth class (GT) and predicted class (P) in green (correct) or red (incorrect).
- **Row 2 (Grad-CAM)**: The Grad-CAM heatmap only, displayed with the `jet` colormap (blue=low activation, red=high activation).
- **Row 3 (Overlay)**: The original image with the Grad-CAM heatmap overlaid at 50% alpha transparency.

Row labels on the left column identify the row type. The target layer for Grad-CAM is the last ConvBlock's `block` sequential module. Figure size: 15×9 inches.

---

### `gradcam_resnet-scratch.png`

Same 3-row × 6-column layout as the CNN-Scratch Grad-CAM figure, but applied to the ResNet-Scratch model. The target layer is `layer4` (the last residual block). ResNet-Scratch typically shows more focused, object-centered activations compared to CNN-Scratch, reflecting its better-learned features at 70% accuracy. Correct predictions (green titles) should show Grad-CAM concentrated on the actual animal/object; incorrect predictions (red titles) will often show attention on background regions. Figure size: 15×9 inches.

---

### `gradcam_resnet-50-pt.png`

Same 3-row × 6-column Grad-CAM layout for ResNet-50-PT (pretrained). Target layer is `layer4`. At 99.1% test accuracy, most or all of the 6 shown samples will be correctly classified (green titles). The Grad-CAM maps are expected to be highly discriminative, tightly localized to the object of interest, demonstrating the quality of pretrained ImageNet representations. Figure size: 15×9 inches.

---

### `attention_vit-scratch.png`

A 3-row × 6-column interpretability figure for ViT-Scratch using attention rollout:

- **Row 1 (Original)**: Raw test images with GT/predicted class labels.
- **Row 2 (Attention)**: Attention rollout map upsampled to 224×224 using the `inferno` colormap (black=low, bright yellow=high attention).
- **Row 3 (Overlay)**: Original image with attention map overlaid at 55% alpha.

Given the model's near-random accuracy, the attention maps are expected to be noisy and diffuse — the model has not learned to attend to meaningful image regions, so the rollout maps will appear as scattered or uniform heatmaps rather than focused object localizations. Figure size: 15×9 inches.

---

### `confidence_distribution.png`

A 3-row × 3-column grid (one panel per model, one cell unused) of dual histograms showing the distribution of **maximum softmax probability** (prediction confidence) for correctly classified samples (green, `#2ecc71`) and incorrectly classified samples (red, `#e74c3c`). X-axis: confidence [0, 1]; Y-axis: density. Each histogram uses 40 bins with 60% opacity. A legend identifies Correct/Incorrect. Well-calibrated, high-accuracy models show correct predictions concentrated near confidence=1.0; poorly calibrated models (CNN-Scratch) show significant overlap between correct and incorrect confidence distributions. Figure size: 18×15 inches.

---

### `reliability_diagrams.png`

A 3-row × 3-column grid of reliability diagrams (one per model) for calibration analysis. Each subplot contains:
- A diagonal dashed line (`y=x`) representing perfect calibration.
- Blue bars representing average accuracy in 15 confidence bins (average accuracy is the actual fraction of correct predictions within that confidence bucket).
- Red shaded areas between the bars and the diagonal line, highlighting the calibration gap.
- Title includes the model name and ECE value.

Figure size: 18×15 inches. The ViT-Scratch diagram will show bars closely following the diagonal (low ECE=0.039) due to its uniformly low confidence scores. Pretrained models will show minor overconfidence (bars slightly below the diagonal at high confidence levels).

---

### `ensemble_top_combos.png`

A horizontal bar chart showing the top-15 ensemble combinations ranked by test accuracy (descending). Bar colors distinguish ensemble size (number of models combined) using different colors from `tab10`. Each bar is annotated with its exact accuracy value. A vertical dashed black line marks the best single model's accuracy (ViT-L16-PT at 1.0000), allowing visual assessment of whether any ensemble exceeds it. A legend maps colors to ensemble sizes. Figure size: 14×7.5 inches. Most bars will be clustered just below 1.0, with the ViT-L16-PT single model line acting as a ceiling.

---

### `ensemble_size_boxplot.png`

A box-and-whisker plot showing the distribution of ensemble test accuracies grouped by ensemble size (2-model, 3-model, 4-model, 5-model, etc.). Each box is filled with a unique color. A horizontal dashed black line indicates the best single model accuracy. The plot reveals whether larger ensembles systematically improve accuracy or show diminishing returns. Box width represents interquartile range; whiskers extend to 1.5×IQR. Figure size: 10×5 inches.

---

### `ablation_label_smoothing.png`

A line plot showing validation accuracy (y-axis) vs. epoch (x-axis, 1–10) for CNN-Scratch trained with three label smoothing values: ε=0 (no smoothing, red), ε=0.1 (green), and ε=0.2 (blue). Each curve uses circular markers. Legend entries include the best validation accuracy for each configuration in the format `ε=X.X (best=Y.YYYY)`. A grid with 0.3 alpha aids readability. Figure size: 10×5 inches.

---

### `ablation_mixup_cutmix.png`

A line plot showing validation accuracy (y-axis) vs. epoch (x-axis, 1–5) for ResNet-Scratch under four augmentation conditions: No MixUp/CutMix, MixUp only, CutMix only, and Both (random). Each condition is plotted with a unique color from `tab10` palette with circular markers (markersize=4). Legend entries include the best validation accuracy. The plot clearly shows that No Augmentation converges fastest in 5 epochs, confirming that MixUp/CutMix are regularizers that require more epochs to show benefit. Figure size: 10×5 inches.

---

### `final_summary_dashboard.png`

A comprehensive 3-row × 4-column (effectively 7-panel) summary figure with title "ImageNet Classification — Final Summary Dashboard". The panels are:

- **Panel 1 (top-left)**: Bar chart of test accuracy for all 7 models. Blue = scratch, red = pretrained. Each bar is annotated with its value (rotated text).
- **Panel 2 (top-center-left)**: Bar chart of Macro F1 for all 7 models, same color convention.
- **Panel 3 (top-center-right)**: Bar chart of ECE (lower = better) for all 7 models, same color convention.
- **Panel 4 (top-right)**: Scatter plot of Parameters (M) vs. Test Accuracy. Bubble sizes proportional to inference time. Scratch = circle, Pretrained = triangle. Points annotated with model names (10-char truncated).
- **Panel 5 (middle row, spanning full width)**: Per-class accuracy heatmap showing accuracy for each model × class combination. Colormap `RdYlGn`. Rows = models, columns = up to 30 classes.
- **Panel 6 (bottom-left or spanning)**: Training time (minutes) vs. test accuracy scatter plot, same marker conventions as Panel 4.

Figure size: 24×18 inches. This is the notebook's highest-information-density visualization and serves as the primary takeaway figure.

---

### `summary_table.csv` and `ensemble_results.csv`

These are not image figures but tabular files saved alongside the figures. `summary_table.csv` contains the 8-column results table for all 7 individual models. `ensemble_results.csv` contains all ensemble combinations with their accuracy and gain-vs-best-single metric.

---

*End of Document*
