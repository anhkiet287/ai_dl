
# Multimodal Homepage — Asset Index (News Category Dataset)

---

## Dataset & EDA

Dataset: **News Category Dataset**  
Source: https://www.kaggle.com/datasets/rmisra/news-category-dataset  

---

## References

```bibtex
@article{misra2022news,
  title={News Category Dataset},
  author={Misra, Rishabh},
  journal={arXiv preprint arXiv:2209.11429},
  year={2022}
}

@book{misra2021sculpting,
  author = {Misra, Rishabh and Grover, Jigyasa},
  title = {Sculpting Data for ML: The First Act of Machine Learning},
  year = {2021},
  isbn = {9798585463570}
}
## Figures

| File | Description | Notes |
|---|---|---|
| `samples.png` | Sample texts from the dataset across multiple categories | Qualitative inspection of dataset content |
| `category_distribution.png` | Class distribution across 42 categories | Highly imbalanced dataset |
| `text_length.png` | Distribution of text lengths | Mean = 33.5, median = 32, mode = 32 |
| `top_stopword_freq.png` | Frequency distribution of stopwords | Highlights common stopword dominance |
| `most_frequent.png` | Most frequent words in corpus | Raw frequency-based lexical overview |
| `vocab_stats.png` | Vocabulary richness per category | Shows lexical diversity differences |
| `category_similarity.png` | Category similarity heatmap | High similarity between Wellness, Healthy Living, Parenting, Parents |

---

## Dataset Statistics

| Metric | Value |
|---|---|
| Number of samples | 209,527 |
| Number of categories | 42 |
| Vocabulary size | 120,874 |
| Total words | 7,025,173 |
| Total stopwords | 2,464,543 |
| Stopword ratio | 0.3508 |

---

## Text Statistics

| Metric | Value |
|---|---|
| Mean text length | 33.5 |
| Median text length | 32.0 |
| Mode text length | 32 |

---

## TF-IDF Visualizations

| File | Description |
|---|---|
| `TF-IDF-POLITICS.png` | Top TF-IDF terms for Politics |
| `TF-IDF-WELLNESS.png` | Top TF-IDF terms for Wellness |
| `TF-IDF-ENTERTAINMENT.png` | Top TF-IDF terms for Entertainment |
| `TF-IDF-TRAVEL.png` | Top TF-IDF terms for Travel |
| `TF-IDF-STYLE&BEAUTY.png` | Top TF-IDF terms for Style & Beauty |

---

## N-Gram Analysis

| File | Description |
|---|---|
| `N-GRAMS-POLITICS.png` | Frequent n-gram patterns in Politics |
| `N-GRAMS-WELLNESS.png` | Frequent n-gram patterns in Wellness |
| `N-GRAMS-ENTERTAINMENT.png` | Frequent n-gram patterns in Entertainment |
| `N-GRAMS-TRAVEL.png` | Frequent n-gram patterns in Travel |
| `N-GRAMS-STYLE&BEAUTY.png` | Frequent n-gram patterns in Style & Beauty |

---

## Preprocessing

### Dataset Filtering & Setup

| Component | Value |
|---|---|
| Selected subset strategy | Filtered low-similarity categories |
| Final vocabulary size | 33,464 |
| Max sequence length | 60 |

---

## Train / Validation / Test Split

| Split | Ratio |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

---

## Training Configuration

| Parameter | Value |
|---|---|
| Batch size | 32 |

---

## Pipeline

| File | Description |
|---|---|
| `DistilBERT_architecture.png` | DistilBERT model pipeline |
| `BiLSTM_architecture.png` | BiLSTM architecture |
| `LSTM_architecture.png` | LSTM architecture |
| `RNN_architecture.png` | SimpleRNN architecture |

---

## Training Figures

| File | Description |
|---|---|
| `simpleRNN-loss.png` | Training vs validation loss curve for SimpleRNN |
| `LSTM-loss.png` | Training vs validation loss curve for LSTM |
| `BiLSTM-loss.png` | Training vs validation loss curve for BiLSTM |

---

## Models Trained

| Model | Type | Notes |
|---|---|---|
| SimpleRNN | Baseline RNN | Fast but weak at long-range dependencies |
| LSTM | RNN variant | Better long-term dependency modeling |
| BiLSTM | Bidirectional LSTM | Uses past + future context for classification |
| DistilBERT | Transformer | Pretrained language model for contextual understanding |

---

## Results & Analysis

### Model Performance

| Model | Accuracy | F1-score |
|---|---|---|
| LSTM | 0.9003 | 0.8996 |
| BiLSTM | 0.8985 | 0.8979 |
| DistilBERT | 0.8778 | 0.8732 |
| SimpleRNN | 0.4849 | 0.3229 |

---

## Confusion Matrices

| File | Description |
|---|---|
| `RNN_confusionmatrix.png` | Confusion matrix for SimpleRNN |
| `LSTM_confusionmatrix.png` | Confusion matrix for LSTM |
| `BiLSTM_confusionmatrix.png` | Confusion matrix for BiLSTM |

---

## Key Insights

- LSTM achieves best performance, slightly outperforming BiLSTM  
- BiLSTM provides marginal improvement over LSTM  
- DistilBERT performs competitively but does not surpass LSTM  
- SimpleRNN performs poorly due to weak long-range dependency modeling  
- Dataset imbalance and category similarity contribute to misclassification  

---
### Expansion
## Loss Function Comparison

| Model | Loss Function | Accuracy | F1-score |
|---|---|---|---|
| SimpleRNN | CrossEntropy | 0.4864 | 0.3239 |
| SimpleRNN | Focal | 0.4844 | 0.3218 |
| LSTM | CrossEntropy | 0.8959 | 0.8955 |
| LSTM | Focal | 0.8911 | 0.8904 |
| BiLSTM | CrossEntropy | 0.8955 | 0.8956 |
| BiLSTM | Focal | 0.8876 | 0.8878 |
| DistilBERT | CrossEntropy | 0.8612 | 0.8530 |
| DistilBERT | Focal | 0.8760 | 0.8721 |