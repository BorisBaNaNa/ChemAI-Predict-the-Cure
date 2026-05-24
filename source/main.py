from preprocess_logic import preprocess_data
from train_logic import check_submission_sanity, train_and_predict_ensemble
import pandas as pd
import numpy as np
import os

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    train = pd.read_csv(os.path.join(script_dir, "..", "data", "train.csv"))
    test = pd.read_csv(os.path.join(script_dir, "..", "data", "test.csv"))

    # Выбор режима работы: 'baseline' или 'advanced'
    METHOD = "advanced"

    feature_cols = [
        col
        for col in train.columns
        if col not in ["index", "IC50, mM", "CC50, mM", "SI"]
    ]

    if METHOD == "advanced":
        print(
            f"[Data Eng] Выбран режим {METHOD}. Схлопываем противоречивые дубликаты..."
        )
        # Группируем по признакам и усредняем таргеты медианой, чтобы убрать биологический шум
        train_aggregated = train.groupby(feature_cols, as_index=False).median()

        # ToDo: Возможна утечка?
        # Пересчитываем строго математический SI для агрегированных строк
        # train_aggregated["SI"] = train_aggregated["CC50, mM"] / (
        #     train_aggregated["IC50, mM"] + 1e-8
        # )

        X_train_raw = train_aggregated[feature_cols]
        y_train_raw = train_aggregated[["IC50, mM", "CC50, mM", "SI"]]
    else:
        print(f"[Data Eng] Выбран режим {METHOD}. Оставляем сырые данные.")
        X_train_raw = train[feature_cols]
        y_train_raw = train[["IC50, mM", "CC50, mM", "SI"]]

    X_test_raw = test[feature_cols]

    X_tr_processed, X_te_processed = preprocess_data(
        X_train_raw, X_test_raw, feature_cols, method=METHOD
    )

    submission = train_and_predict_ensemble(X_tr_processed, y_train_raw, X_te_processed)

    check_submission_sanity(submission, train)

    # Жесткий пост-процессинг SI по чистой формуле
    # if METHOD == "advanced":
    #     print(
    #         "[Post-processing] Пересчет тест-таргета SI строго по формуле CC50 / IC50..."
    #     )
    #     submission["SI"] = submission["CC50"] / (submission["IC50"] + 1e-8)

    submission.to_csv("submission_optimized_weights.csv", index=False)
    print(
        f"[Успех] Файл submission_optimized_weights.csv успешно сохранен (Режим: {METHOD})!"
    )
