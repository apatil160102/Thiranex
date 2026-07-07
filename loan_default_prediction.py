"""
Predictive Modeling Using Machine Learning
--------------------------------------------
Predicts loan default using Logistic Regression, a Decision Tree, and a
Random Forest, then evaluates and compares them using accuracy, precision,
recall, F1, ROC-AUC, confusion matrices, and ROC curves.

Author: Namra
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


# --------------------------------------------------------------------------
# 1. Load / generate the dataset
# --------------------------------------------------------------------------
def generate_data(n=2000, seed=42):
    """
    Generates a synthetic loan applicant dataset where default risk
    realistically depends on credit score, debt-to-income ratio, income,
    loan amount, employment history, and existing loans.
    Replace this function with pd.read_csv("your_file.csv") to use a
    real dataset instead.
    """
    np.random.seed(seed)

    age = np.random.randint(21, 65, n)
    annual_income = np.round(np.random.gamma(6, 8000, n), -2)
    loan_amount = np.round(np.random.gamma(4, 6000, n), -2)
    credit_score = np.clip(np.random.normal(650, 90, n), 300, 850).astype(int)
    employment_years = np.round(np.clip(np.random.exponential(5, n), 0, 40), 1)
    existing_loans = np.random.poisson(1.2, n)
    debt_to_income_ratio = np.round(np.clip(np.random.normal(0.35, 0.15, n), 0.01, 1.2), 2)

    logit = (
        -4
        + 0.006 * (700 - credit_score)
        + 3.5 * debt_to_income_ratio
        + 0.15 * existing_loans
        - 0.00002 * annual_income
        + 0.00005 * loan_amount
        - 0.03 * employment_years
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default = np.random.binomial(1, prob_default)

    return pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "existing_loans": existing_loans,
        "debt_to_income_ratio": debt_to_income_ratio,
        "default": default,
    })


# --------------------------------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------------------------------
def split_and_scale(df, target="default", test_size=0.25, seed=42):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


# --------------------------------------------------------------------------
# 3. Train models
# --------------------------------------------------------------------------
def train_models(X_train, X_train_scaled, y_train, seed=42, balanced=False):
    """Trains Logistic Regression, Decision Tree, and Random Forest."""
    weight = "balanced" if balanced else None

    models = {
        "Logistic Regression": LogisticRegression(random_state=seed, class_weight=weight),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=seed, class_weight=weight),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=seed, class_weight=weight
        ),
    }

    fitted = {}
    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_scaled, y_train)
        else:
            model.fit(X_train, y_train)
        fitted[name] = model

    return fitted


# --------------------------------------------------------------------------
# 4. Evaluation
# --------------------------------------------------------------------------
def evaluate_models(fitted_models, X_test, X_test_scaled, y_test):
    """Returns predictions, probabilities, and a metrics summary table."""
    predictions, probabilities, results = {}, {}, []

    for name, model in fitted_models.items():
        X_eval = X_test_scaled if name == "Logistic Regression" else X_test
        preds = model.predict(X_eval)
        probs = model.predict_proba(X_eval)[:, 1]

        predictions[name] = preds
        probabilities[name] = probs

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds),
            "F1": f1_score(y_test, preds),
            "ROC-AUC": roc_auc_score(y_test, probs),
        })

    results_df = pd.DataFrame(results).set_index("Model").round(3)
    return predictions, probabilities, results_df


def plot_confusion_matrices(predictions, y_test, save_path=None):
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_test, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curves(probabilities, y_test, save_path=None):
    plt.figure(figsize=(7, 6))

    for name, probs in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(model, feature_names, save_path=None):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh", color="teal")
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return importances


# --------------------------------------------------------------------------
# 5. Main
# --------------------------------------------------------------------------
def main():
    print("Generating dataset...")
    df = generate_data()
    df.to_csv("loan_data.csv", index=False)
    print(f"Shape: {df.shape}")
    print("Default rate:\n", df["default"].value_counts(normalize=True).round(3))

    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = split_and_scale(df)

    print("\nTraining baseline models...")
    baseline_models = train_models(X_train, X_train_scaled, y_train, balanced=False)
    predictions, probabilities, results_df = evaluate_models(
        baseline_models, X_test, X_test_scaled, y_test
    )
    print("\nBaseline results:\n", results_df)

    plot_confusion_matrices(predictions, y_test, save_path="confusion_matrices.png")
    plot_roc_curves(probabilities, y_test, save_path="roc_curves.png")
    plot_feature_importance(
        baseline_models["Random Forest"], X_train.columns, save_path="feature_importance.png"
    )

    print("\nTraining class-weight-balanced models (to improve recall on defaults)...")
    balanced_models = train_models(X_train, X_train_scaled, y_train, balanced=True)
    _, _, balanced_results_df = evaluate_models(
        balanced_models, X_test, X_test_scaled, y_test
    )
    balanced_results_df.index = [f"{name} (balanced)" for name in balanced_results_df.index]
    print("\nBalanced results:\n", balanced_results_df)

    print("\nCombined comparison:")
    print(pd.concat([results_df, balanced_results_df]))


if __name__ == "__main__":
    main()
