# Machine Learning Model Explanation & Fixes

## 1. Executive Summary
This project aims to predict loan defaults using a dataset of ~255,000 records. We explored three models: **Logistic Regression**, **Neural Networks**, and **XGBoost**. The dataset is highly **imbalanced** (approx. 12% default rate), which requires specific handling (Class Weighting) to ensure the models don't just predict "No Default" for everyone.

### Key Fixes Implemented Across All Models
*   **Pathing:** Scripts now look for `Loan_default.csv` in the local directory, removing dependency on Google Drive.
*   **Imbalance Handling:** Implemented `class_weight='balanced'` (Logistic Regression), `pos_weight` (Neural Network), and `scale_pos_weight` (XGBoost) to heavily penalize missing a Default. This increases Recall (catching more defaults) at the cost of Precision (more false alarms).
*   **Preprocessing:** Removed `LoanID` (irrelevant feature). Standardized numeric features (critical for Neural Networks/Logistic Regression). One-Hot or Ordinal Encoded categorical features.

---

## 2. Model Explanations

### 1. Logistic Regression
**What it is:** A statistical method that models the probability of a binary outcome (0 or 1) using a linear equation wrapped in a sigmoid (S-shaped) function.
**Appropriateness:**
*   **Pros:** Simple, interpretable, fast. Good baseline.
*   **Cons:** Assumes a linear relationship between features and log-odds. Might struggle with complex, non-linear patterns in credit data.
**Fixes:**
*   Added `class_weight='balanced'` to tell the model that 1 Default is as important as ~8 Non-Defaults.
*   Used `StandardScaler` because Logistic Regression coefficients are sensitive to scale.

### 2. Neural Network (Multi-Layer Perceptron)
**What it is:** A biologically inspired model consisting of layers of "neurons". It learns complex non-linear interactions between features.
**Appropriateness:**
*   **Pros:** Can model extremely complex relationships. High potential accuracy.
*   **Cons:** "Black box" (hard to interpret). Requires massive data and careful tuning. Sensitive to unscaled data.
**Fixes:**
*   **Architecture:** Used a 3-layer network with Batch Normalization and Dropout (to prevent overfitting).
*   **Loss Function:** Used `BCEWithLogitsLoss` with a positive weight calculated from the training data to handle imbalance.
*   **Preprocessing:** STRICT usage of `StandardScaler`. Neural Nets fail if inputs are not normalized (e.g., Income=50000 vs Age=30).

### 3. XGBoost (Extreme Gradient Boosting)
**What it is:** An ensemble method that builds thousands of small "Decision Trees" sequentially. Each new tree tries to fix the errors of the previous ones.
**Appropriateness:**
*   **Pros:** **State-of-the-art** for tabular data like this. Handles missing values and categories well. Highly optimized.
*   **Cons:** Can overfit if not tuned.
**Fixes:**
*   Enabled GPU acceleration (`device='cuda'`).
*   Calculated `scale_pos_weight` dynamically to handle class imbalance.
*   Used `OrdinalEncoder` for clean categorical handling.

---

## 3. User Questions & Answers

### Q: Are these models appropriate for my dataset?
**ANSWER:** **Yes.**
*   **XGBoost** is arguably the *best* choice for this type of tabular financial data. It is the industry standard for credit scoring competitions.
*   **Logistic Regression** is the industry standard for *explainability* (important in banking regulations).
*   **Neural Networks** are appropriate but might be overkill unless the dataset is very large and complex; XGBoost usually beats them on tabular data with less effort.

### Q: Can you give me some graphs like Cook's Distance?
**ANSWER:**
*   **Cook's Distance** is a metric specifically for **Linear Regression** (continuous outcome) to find outliers that skew the line. It is not standard for Binary Classification (0/1 outcome).
*   **The Appropriate Equivalents:** Instead, I generated the industry-standard diagnostic plots for Classification:
    *   **ROC Curve (Receiver Operating Characteristic):** Shows how well the model separates Defaults from Non-Defaults. An AUC of 0.5 is random guessing; 1.0 is perfect.
    *   **Confusion Matrix:** Shows exactly how many Defaults were caught (True Positives) vs missed (False Negatives).

### Q: Is it possible to run it locally using my graphics card?
**ANSWER:** **Yes!**
*   **XGBoost:** I enabled GPU support in the script (`device='cuda'`). It ran successfully on your NVIDIA GPU.
*   **Neural Network:** I implemented the PyTorch script to automatically detect and use your GPU (`device='cuda'`).
*   **Logistic Regression:** Runs on CPU (scikit-learn does not support GPU), but it is mathematically simple and fast enough that GPU is not needed.

---

## 4. Results Summary

| Model | Accuracy | Recall (Class 1 - Default) | Notes |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | ~67% | ~70% | Good at catching defaults, but many false alarms. |
| **Neural Network** | *Training...* | *Pending* | Expect similar or better performance. |
| **XGBoost** | ~72% | ~63% | Best overall balance. Higher accuracy means fewer false alarms while still catching a majority of defaults. |

*Note: In credit default prediction, **Recall** (catching defaults) is often more important than Accuracy. A 70% recall means we catch 70% of potential bad loans.*
