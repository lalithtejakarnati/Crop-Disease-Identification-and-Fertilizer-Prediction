"""
Fertilizer Recommendation Model Training for Crop AI Assistant.

This script implements:
1. Agronomic feature engineering (NPK ratios, totals, fractions, differences, environmental interactions).
2. Stratified 5-Fold Cross-Validation evaluation.
3. Class-imbalance-aware ExtraTrees model fitting.
4. Out-of-sample evaluation on an untouched holdout test set.
5. Top-1, Top-2, and Top-3 accuracy evaluation.
6. Serialization of the complete pipeline to FERTILIZER_MODEL_PATH.
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.config.settings import (
    FERTILIZER_DATA_PATH,
    FERTILIZER_MODEL_PATH,
)
from src.utils.fertilizer_utils import build_agronomic_features


def main():
    print("=" * 60)
    print("FERTILIZER MODEL TRAINING & CROSS-VALIDATION AUDIT")
    print("=" * 60)

    print(f"Dataset: {FERTILIZER_DATA_PATH}")

    data = pd.read_csv(FERTILIZER_DATA_PATH)
    data.columns = data.columns.str.strip()

    print(f"Total Rows: {len(data)}")
    print(f"Columns: {list(data.columns)}")
    print()

    # ============================================================
    # FEATURES AND TARGET
    # ============================================================

    target_column = "Fertilizer Name"
    feature_columns = [
        "Temparature",
        "Humidity",
        "Moisture",
        "Soil Type",
        "Crop Type",
        "Nitrogen",
        "Potassium",
        "Phosphorous",
    ]

    X = data[feature_columns]
    y = data[target_column]

    categorical_features = ["Soil Type", "Crop Type"]
    sample_feat = build_agronomic_features(X.iloc[:2])
    engineered_numeric_features = [
        c for c in sample_feat.columns if c not in categorical_features
    ]

    # ============================================================
    # PIPELINE DEFINITION
    # ============================================================

    feature_transformer = FunctionTransformer(build_agronomic_features)

    column_preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numeric",
                "passthrough",
                engineered_numeric_features,
            ),
        ]
    )

    classifier = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=3,
        criterion="log_loss",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineer", feature_transformer),
            ("preprocessor", column_preprocessor),
            ("model", classifier),
        ]
    )

    # ============================================================
    # STRATIFIED 5-FOLD CROSS-VALIDATION
    # ============================================================

    print("=" * 60)
    print("STRATIFIED 5-FOLD CROSS-VALIDATION EVALUATION")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accs = []
    fold_macro_f1s = []
    fold_weighted_f1s = []
    cv_trues = []
    cv_preds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold, y_train_fold = X.iloc[train_idx], y.iloc[train_idx]
        X_val_fold, y_val_fold = X.iloc[val_idx], y.iloc[val_idx]

        pipeline.fit(X_train_fold, y_train_fold)
        pred_fold = pipeline.predict(X_val_fold)

        fold_acc = accuracy_score(y_val_fold, pred_fold)
        fold_mf1 = f1_score(
            y_val_fold, pred_fold, average="macro", zero_division=0
        )
        fold_wf1 = f1_score(
            y_val_fold, pred_fold, average="weighted", zero_division=0
        )

        fold_accs.append(fold_acc)
        fold_macro_f1s.append(fold_mf1)
        fold_weighted_f1s.append(fold_wf1)
        cv_trues.extend(y_val_fold)
        cv_preds.extend(pred_fold)

        print(
            f"Fold {fold_idx}: Accuracy = {fold_acc:.2%}, "
            f"Macro F1 = {fold_mf1:.2%}, Weighted F1 = {fold_wf1:.2%}"
        )

    print("-" * 60)
    print(
        f"Mean CV Accuracy:    {np.mean(fold_accs):.2%} "
        f"± {np.std(fold_accs):.2%}"
    )
    print(
        f"Mean CV Macro F1:    {np.mean(fold_macro_f1s):.2%} "
        f"± {np.std(fold_macro_f1s):.2%}"
    )
    print(
        f"Mean CV Weighted F1: {np.mean(fold_weighted_f1s):.2%} "
        f"± {np.std(fold_weighted_f1s):.2%}"
    )
    print()

    # ============================================================
    # UNTOUCHED HOLDOUT TEST EVALUATION
    # ============================================================

    print("=" * 60)
    print("UNTOUCHED HOLDOUT TEST SET EVALUATION (20% Split)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print()

    pipeline.fit(X_train, y_train)

    test_preds = pipeline.predict(X_test)
    test_probs = pipeline.predict_proba(X_test)
    classes = pipeline.classes_

    test_accuracy = accuracy_score(y_test, test_preds)
    test_macro_f1 = f1_score(
        y_test, test_preds, average="macro", zero_division=0
    )
    test_weighted_f1 = f1_score(
        y_test, test_preds, average="weighted", zero_division=0
    )

    # Top-K accuracy
    y_test_arr = y_test.values
    top1_acc = test_accuracy
    top2_acc = np.mean(
        [
            true in classes[np.argsort(prob)[-2:]]
            for prob, true in zip(test_probs, y_test_arr)
        ]
    )
    top3_acc = np.mean(
        [
            true in classes[np.argsort(prob)[-3:]]
            for prob, true in zip(test_probs, y_test_arr)
        ]
    )

    print(f"Test Accuracy (Top-1): {top1_acc:.2%}")
    print(f"Test Top-2 Accuracy:   {top2_acc:.2%}")
    print(f"Test Top-3 Accuracy:   {top3_acc:.2%}")
    print(f"Test Macro F1:         {test_macro_f1:.2%}")
    print(f"Test Weighted F1:      {test_weighted_f1:.2%}")
    print()

    print("Holdout Classification Report:")
    print(
        classification_report(
            y_test,
            test_preds,
            zero_division=0,
        )
    )

    print("Holdout Confusion Matrix:")
    cm = confusion_matrix(y_test, test_preds, labels=sorted(list(classes)))
    cm_df = pd.DataFrame(
        cm,
        index=sorted(list(classes)),
        columns=sorted(list(classes)),
    )
    print(cm_df)
    print()

    # ============================================================
    # SAVE MODEL
    # ============================================================

    joblib.dump(
        pipeline,
        FERTILIZER_MODEL_PATH,
    )

    print("=" * 60)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 60)
    print(FERTILIZER_MODEL_PATH)


if __name__ == "__main__":
    main()
