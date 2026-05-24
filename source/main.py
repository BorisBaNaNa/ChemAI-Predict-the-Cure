import os
import pandas as pd

from preprocess_logic import preprocess_data
from train_logic import (
    check_submission_sanity,
    train_and_predict_ensemble,
)

# Константы
TARGET_COLUMNS = ["IC50, mM", "CC50, mM", "SI"]

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"

OUTPUT_FILENAME = "submission_optimized_weights.csv"

if __name__ == "__main__":
    # Загрузка данных
    script_dir = os.path.dirname(os.path.abspath(__file__))

    train_path = os.path.join(
        script_dir,
        "..",
        "data",
        TRAIN_FILENAME,
    )

    test_path = os.path.join(
        script_dir,
        "..",
        "data",
        TEST_FILENAME,
    )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # выделение признаков
    feature_cols = [
        col for col in train.columns if col not in ["index", *TARGET_COLUMNS]
    ]

    # обработка дубликатов
    print("[Data Eng] Схлопывание противоречивых дубликатов...")

    # В датасете могут существовать одинаковые молекулы
    # с немного различающимися таргетами из-за биологического шума
    # и вариативности лабораторных измерений.
    #
    # Группировка по признакам с медианой:
    #     - уменьшает влияние шумных измерений
    #     - делает train distribution стабильнее
    #     - снижает вероятность переобучения на выбросах
    train_aggregated = train.groupby(feature_cols, as_index=False).median()

    # Разделение features / targets
    X_train_raw = train_aggregated[feature_cols]

    y_train_raw = train_aggregated[TARGET_COLUMNS]

    X_test_raw = test[feature_cols]

    # Предобработка
    print("[Data Eng] Предобработка признаков...")

    X_train_processed, X_test_processed = preprocess_data(
        X_train_raw,
        X_test_raw,
        feature_cols,
    )

    # Обучение и предсказание
    print("[ML] Обучение ансамбля моделей...")

    submission = train_and_predict_ensemble(
        X_train_processed,
        y_train_raw,
        X_test_processed,
    )

    check_submission_sanity(
        submission,
        train,
    )

    submission.to_csv(
        OUTPUT_FILENAME,
        index=False,
    )

    print(f"[Успех] Файл {OUTPUT_FILENAME} успешно сохранен!")
