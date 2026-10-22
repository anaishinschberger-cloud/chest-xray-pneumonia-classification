# Chest X-Ray Pneumonia Classification

This project explores the detection of pneumonia from chest X-ray images, with a focus on data preparation, transfer learning, model evaluation, and generalization.

The objective is to build a reproducible machine learning pipeline while addressing common challenges in medical image classification, including class imbalance, duplicate images, and potential data leakage.

## Dataset

The project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset, containing **5,856 chest X-ray images** belonging to two classes:

- **NORMAL:** Chest X-rays without pneumonia.
- **PNEUMONIA:** Chest X-rays labeled as pneumonia.

Instead of relying on the original Kaggle train/validation/test directories, the original images are pooled and redistributed into new subsets.

The splitting procedure includes:

- Detection and removal of exact duplicate images.
- Grouping images using identifiers inferred from filenames.
- A target split of **70% training, 15% validation, and 15% testing**.
- Data augmentation applied only to the training set.

Patient grouping is inferred from filename conventions rather than independently verified patient metadata. Therefore, the procedure reduces identifiable leakage risks but cannot guarantee complete patient-level independence.

## Model and training

The classification model is based on **EfficientNetB0**, pretrained on ImageNet and adapted for binary classification.

Training is divided into two stages:

**Phase 1 — Classifier training** : The pretrained feature extractor is frozen, and only the classification head is trained.

**Phase 2 — Fine-tuning** : The last two feature-extraction blocks and the classification head are trained using lower learning rates to adapt the pretrained representations to chest X-ray images.

The implementation uses:

- PyTorch and Torchvision
- Binary cross-entropy with logits (BCEWithLogitsLoss)
- Adam  optimizer
- Validation-loss-based checkpoint selection
- Threshold-dependent classification evaluation

## Evaluation

Model performance is assessed using:

- Accuracy
- Precision, recall, and F1-score
- ROC-AUC
- Binary cross-entropy loss
- Class-specific performance for NORMAL and PNEUMONIA

Classification thresholds are compared on the validation set. A selected threshold is then applied to the test set.

### Latest reported test results

| Metric                   |      Value |
| ------------------------ | ---------: |
| Accuracy                 | **96.22%** |
| ROC-AUC                  | **0.9913** |
| Precision — PNEUMONIA    |     97.51% |
| Recall — PNEUMONIA       |     97.36% |
| Recall — NORMAL          |     93.04% |
| Test loss                |     0.1030 |
| Classification threshold |       0.60 |

These results correspond to the latest reported evaluation on **873 test images**.

The test set was previously inspected during experimentation, so these results should be interpreted as an internal benchmark rather than a completely untouched external evaluation.

## Repository structure

```text
chest-xray-pneumonia-classification/
├── 00_download_dataset.ipynb
├── 01_new_split.ipynb
├── 02_train.ipynb
└── .gitignore
```

**00_download_dataset.ipynb** — Download and organize the original dataset.

**01_new_split.ipynb** — Audit the original images, detect duplicates, create image groups, and build the new training, validation, and test splits.

**02_train.ipynb** — Load the processed data, train and fine-tune EfficientNetB0, select checkpoints using validation loss, and evaluate classification performance.

The datasets and trained model weights are intentionally excluded from version control.

## Limitations

Despite strong internal classification results, several limitations remain:

- **Patient identification:** Filename-derived groups may not correspond perfectly to unique patients.
- **Dataset representativeness:** The data comes from a limited source and may not represent other hospitals, devices, or patient populations.
- **Evaluation independence:** The test data was inspected during model development.
- **Generalization:** Performance on a newly constructed internal split does not establish performance on independent clinical datasets.
- **Clinical applicability:** The model has not undergone clinical validation.

This is an educational machine learning project, **not a medical diagnostic tool**.

## Dataset reference

Kermany et al., *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, Cell, 2018.

Dataset available on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
