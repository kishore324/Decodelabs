# Project 2: Data Classification Using AI
**DecodeLabs Industrial Training Kit - Artificial Intelligence (Batch 2026)**

Goal: build a basic supervised-learning classification model on a small dataset (Iris) and validate it properly.

## Pipeline (IPO framework)

| Stage | What happens | Where in `classify_iris.py` |
|---|---|---|
| **Input** | Load the Iris benchmark (150 samples, 3 classes, 4 features), inspect it, scale features with `StandardScaler` (mean 0, variance 1) | `load_data`, `scale_features` |
| **Process** | Shuffled 80/20 train-test split, choose K via the "elbow", train K-Nearest Neighbors | `split_data`, `tune_k`, `train_model` |
| **Output** | Confusion matrix (TP/FP/FN/TN), precision, recall, F1 score | `evaluate` |

## Key design decisions
- **Shuffle before splitting** to remove order bias (the Iris file is sorted by class). The split is also stratified so each class appears equally in train and test.
- **Scaler is fit on training data only**, then applied to test data, so nothing leaks from the test set.
- **K is chosen with 5-fold cross-validation on the training set**, never the test set. The result is plotted as the error-rate curve ("the elbow").
- **Accuracy is not trusted alone**; the confusion matrix and per-class precision / recall / F1 are reported.

## Run it
```bash
pip install -r requirements.txt
python classify_iris.py
```

## Outputs (`outputs/`)
- `k_selection.png` - error rate vs K (the elbow)
- `confusion_matrix.png` - confusion matrix on the test set
- `model_comparison.csv` - KNN vs Logistic Regression vs Decision Tree

Console output includes dataset summary, scaling check, classification report and a prediction for a brand-new flower.

## Extensions (from the Conclusion slide)
Compare different algorithms (included), test with completely new data (included), and treat any low score as a learning opportunity. Next track: Deep Learning and CNNs.
