# Multimodal Homepage — Asset Index (News Category Dataset)

---

## Dataset & EDA

Dataset: **News Category Dataset**  
Source: https://www.kaggle.com/datasets/rmisra/news-category-dataset  

### Figures

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

### Dataset Statistics

| Metric | Value |
|---|---|
| Number of samples | 209,527 |
| Number of categories | 42 |
| Vocabulary size | 120,874 |
| Total words | 7,025,173 |
| Total stopwords | 2,464,543 |
| Stopword ratio | 0.3508 |

Text statistics:
- Mean text length: 33.5  
- Median text length: 32.0  
- Mode text length: 32  

---

### TF-IDF Visualizations

| File | Description |
|---|---|
| `TF-IDF-POLITICS.png` | Top TF-IDF terms for Politics |
| `TF-IDF-WELLNESS.png` | Top TF-IDF terms for Wellness |
| `TF-IDF-ENTERTAINMENT.png` | Top TF-IDF terms for Entertainment |
| `TF-IDF-TRAVEL.png` | Top TF-IDF terms for Travel |
| `TF-IDF-STYLE&BEAUTY.png` | Top TF-IDF terms for Style & Beauty |

---

### N-Gram Analysis

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

### Train / Validation / Test Split

| Split | Ratio |
|---|---|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

---

### Training Configuration

| Parameter | Value |
|---|---|
| Batch size | 32 |

---

## Training

### Figures

| File | Description |
|---|---|
| `simpleRNN-loss.png` | Training vs validation loss curve for SimpleRNN |
| `LSTM-loss.png` | Training vs validation loss curve for LSTM |
| `BiLSTM-loss.png` | Training vs validation loss curve for BiLSTM |

---

### Models Trained

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

### Figures

| File | Description |
|---|---|
| `RNN_confusionmatrix.png` | Confusion matrix for SimpleRNN |
| `LSTM_confusionmatrix.png` | Confusion matrix for LSTM |
| `BiLSTM_confusionmatrix.png` | Confusion matrix for BiLSTM |

---

### Key Insights

- LSTM achieves the best overall performance, slightly outperforming BiLSTM.
- BiLSTM provides marginal improvement over LSTM in context understanding but not significantly in metrics.
- DistilBERT performs well but does not surpass LSTM, possibly due to dataset size or fine-tuning limitations.
- SimpleRNN performs poorly, confirming its limitation in capturing long-range dependencies.
- Dataset imbalance and category similarity likely affect misclassification patterns.

---