# Predictive Modeling: Loan Default Prediction

A Python project that trains and compares three supervised learning models to predict whether a loan applicant will default.

## What it does

- **Applies three algorithms**: Logistic Regression, Decision Tree, and Random Forest
- **Trains and tests models properly**: stratified train/test split, feature scaling for Logistic Regression
- **Evaluates performance**: accuracy, precision, recall, F1, and ROC-AUC for each model
- **Visualizes performance**: confusion matrices and ROC curves for direct comparison
- **Shows feature importance** from the Random Forest
- **Addresses class imbalance**: compares baseline models against `class_weight="balanced"` versions, demonstrating the precision/recall tradeoff

## Tech stack

- Python 3
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn

## How to run

```bash
pip install -r requirements.txt
python loan_default_prediction.py
```

This will:
1. Generate a synthetic loan applicant dataset (`loan_data.csv`) with a realistic ~16% default rate
2. Train and evaluate Logistic Regression, Decision Tree, and Random Forest
3. Save chart images: `confusion_matrices.png`, `roc_curves.png`, `feature_importance.png`
4. Print a full metrics comparison table, including class-weight-balanced versions

## Using a real dataset instead

Replace the call to `generate_data()` in `main()` with:

```python
df = pd.read_csv("your_dataset.csv")
```

and make sure your target column is named `default` (or update the `target` parameter in `split_and_scale`).

## Key finding

Baseline models are conservative and miss most real defaulters (recall as low as ~5%) despite ~82-86% accuracy — because accuracy alone is misleading on an imbalanced dataset. Using `class_weight="balanced"` raises recall substantially (e.g. Logistic Regression: 15% → 66%) at the cost of precision — a classic real-world tradeoff in credit risk modeling, where missing an actual default is usually costlier than a false alarm.

## License

Feel free to use or adapt this project for learning purposes.

