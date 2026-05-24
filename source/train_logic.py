import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# ============================================================
# КОНСТАНТЫ
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 5
WEIGHT_SEARCH_RANGE = np.arange(0.4, 0.71, 0.05)

ENSEMBLE_SEEDS = [42, 123, 456]

PERCENTILE_CLIP = 99
MIN_TARGET_VALUE = 1e-6
MIN_SI_VALUE = 0.1
EPSILON = 1e-8

TARGET_COLUMNS = ["IC50", "CC50", "SI"]

# ============================================================
# ПАРАМЕТРЫ МОДЕЛЕЙ
# ============================================================

# Multi-output модель обучается сразу на всех таргетах,
# что позволяет учитывать связь:
# SI = CC50 / IC50
PARAMS_MULTI = {
    "iterations": 800,              # Количество деревьев (оптимально для 751 объектов)
    "learning_rate": 0.04,          # Скорость обучения (баланс качества и переобучения)
    "depth": 6,                     # Глубина деревьев (умеренная, чтобы не переобучаться)
    "l2_leaf_reg": 3,               # L2-регуляризация (предотвращает переобучение)
    "border_count": 128,            # Количество бинов для дискретизации
    "random_strength": 1,           # Случайность при выборе признаков
    "bagging_temperature": 1,       # Контроль бэггинга
    "od_type": "Iter",              # Early stopping
    "od_wait": 50,                  # Ждем 50 итераций для улучшения
    "loss_function": "MultiRMSE",   # Важно: учитывает все 3 таргета!
    "verbose": False,
    "allow_writing_files": False,
}

# Single-output модели чуть сильнее регуляризованы,
# так как переобучаются быстрее multi-output варианта.
PARAMS_SINGLE = {
    "iterations": 800,
    "learning_rate": 0.05,
    "depth": 5,                     # Чуть меньше глубина для отдельных моделей
    "l2_leaf_reg": 5,               # Чуть больше регуляризации
    "border_count": 128,
    "random_strength": 1,
    "bagging_temperature": 1,
    "od_type": "Iter",
    "od_wait": 50,
    "loss_function": "RMSE",        # Стандартная функция потерь
    "verbose": False,
    "allow_writing_files": False,
}


def train_and_predict_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> pd.DataFrame:

    np.random.seed(RANDOM_STATE)

    # Логарифмирование стабилизирует распределение таргетов
    # и уменьшает влияние экстремальных значений.
    y_train_log = np.log1p(y_train)

    X_train_array = np.array(X_train)
    X_test_array = np.array(X_test)

    print("\n[ML] Запуск 5-Fold кросс-валидации")

    kf = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    best_weight = 0.5
    best_score = float("inf")

    # OOF предсказания используются для объективной оценки качества:
    # каждый объект предсказывается моделью,
    # которая не обучалась на нём.
    oof_predictions = np.zeros_like(y_train_log.values)

    for weight_multi in WEIGHT_SEARCH_RANGE:

        fold_scores = []
        temp_oof_predictions = np.zeros_like(y_train_log.values)

        for train_indices, validation_indices in kf.split(X_train_array):

            X_train_fold = X_train_array[train_indices]
            X_validation_fold = X_train_array[validation_indices]

            y_train_fold = y_train_log.iloc[train_indices]
            y_validation_fold = y_train_log.iloc[validation_indices]

            # ====================================================
            # MULTI-OUTPUT МОДЕЛЬ
            # ====================================================

            multi_model = CatBoostRegressor(
                **PARAMS_MULTI,
                random_seed=RANDOM_STATE,
            )

            multi_model.fit(X_train_fold, y_train_fold)

            multi_predictions = multi_model.predict(X_validation_fold)

            # ====================================================
            # SINGLE-OUTPUT МОДЕЛИ
            # ====================================================

            single_predictions = []

            for target_index in range(len(TARGET_COLUMNS)):

                single_model = CatBoostRegressor(
                    **PARAMS_SINGLE,
                    random_seed=RANDOM_STATE,
                )

                single_model.fit(
                    X_train_fold,
                    y_train_fold.iloc[:, target_index],
                )

                target_predictions = single_model.predict(X_validation_fold)

                single_predictions.append(target_predictions)

            single_predictions = np.column_stack(single_predictions)

            # Вес multi-output модели подбирается через CV,
            # а не фиксируется вручную.
            ensemble_predictions = (
                weight_multi * multi_predictions
                + (1 - weight_multi) * single_predictions
            )

            temp_oof_predictions[validation_indices] = ensemble_predictions

            rmse = np.sqrt(
                mean_squared_error(
                    y_validation_fold,
                    ensemble_predictions,
                    multioutput="raw_values",
                )
            )

            fold_scores.append(np.mean(rmse))

        average_score = np.mean(fold_scores)

        if average_score < best_score:
            best_score = average_score
            best_weight = weight_multi
            oof_predictions = temp_oof_predictions

    # ============================================================
    # ОТЧЁТ ПО КАЧЕСТВУ
    # ============================================================

    print("\n" + "=" * 60)
    print(" МЕТРИКИ КАЧЕСТВА НА КРОСС-ВАЛИДАЦИИ")
    print("=" * 60)

    for target_index, target_name in enumerate(TARGET_COLUMNS):

        target_rmse = np.sqrt(
            mean_squared_error(
                y_train_log.iloc[:, target_index],
                oof_predictions[:, target_index],
            )
        )

        print(f"  • RMSE {target_name:4}: {target_rmse:.4f}")

    print(f"  --> Средний CV Score: {best_score:.4f}")
    print(f"  --> Оптимальный вес multi-output: {best_weight:.2f}")
    print("=" * 60 + "\n")

    # ============================================================
    # ФИНАЛЬНОЕ ОБУЧЕНИЕ
    # ============================================================

    print("[ML] Обучение финальных моделей")

    multi_model = CatBoostRegressor(
        **PARAMS_MULTI,
        random_seed=RANDOM_STATE,
    )

    multi_model.fit(X_train_array, y_train_log)

    multi_predictions = np.expm1(
        multi_model.predict(X_test_array)
    )

    # Усреднение нескольких seed уменьшает дисперсию модели
    # и делает предсказания стабильнее.
    single_predictions_per_seed = []

    for seed in ENSEMBLE_SEEDS:

        target_predictions_list = []

        for target_index in range(len(TARGET_COLUMNS)):

            single_model = CatBoostRegressor(
                **PARAMS_SINGLE,
                random_seed=seed,
            )

            single_model.fit(
                X_train_array,
                y_train_log.iloc[:, target_index],
            )

            target_predictions = single_model.predict(X_test_array)

            target_predictions_list.append(target_predictions)

        single_predictions_per_seed.append(
            np.column_stack(target_predictions_list)
        )

    single_predictions_average = np.mean(
        single_predictions_per_seed,
        axis=0,
    )

    single_predictions = np.expm1(
        single_predictions_average
    )

    final_predictions = (
        best_weight * multi_predictions
        + (1 - best_weight) * single_predictions
    )

    # ============================================================
    # ПОСТ-ПРОЦЕССИНГ
    # ============================================================

    # Ограничение экстремальных выбросов.
    for target_index in range(len(TARGET_COLUMNS)):

        percentile_99 = np.percentile(
            final_predictions[:, target_index],
            PERCENTILE_CLIP,
        )

        final_predictions[:, target_index] = np.clip(
            final_predictions[:, target_index],
            MIN_TARGET_VALUE,
            percentile_99,
        )

    # SI физически определяется как:
    # SI = CC50 / IC50
    #
    # Если модель слишком сильно нарушает эту связь,
    # применяется адаптивная коррекция.
    calculated_ratio = (
        final_predictions[:, 1]
        / (final_predictions[:, 0] + EPSILON)
    )

    ratio_difference = (
        np.abs(final_predictions[:, 2] - calculated_ratio)
        / (final_predictions[:, 2] + EPSILON)
    )

    adaptive_weight = np.clip(
        ratio_difference,
        0.1,
        0.5,
    )

    final_predictions[:, 2] = (
        (1 - adaptive_weight) * final_predictions[:, 2]
        + adaptive_weight * calculated_ratio
    )

    final_predictions[:, 2] = np.maximum(
        final_predictions[:, 2],
        MIN_SI_VALUE,
    )

    submission = pd.DataFrame(
        {
            "index": X_test.index,
            "IC50": final_predictions[:, 0],
            "CC50": final_predictions[:, 1],
            "SI": final_predictions[:, 2],
        }
    )

    return submission


def check_submission_sanity(
    submission_df: pd.DataFrame,
    train_df: pd.DataFrame,
):
    print("\n" + "=" * 60)
    print(" АНАЛИЗ ФИЗИЧЕСКОЙ АДЕКВАТНОСТИ ПРЕДСКАЗАНИЙ")
    print("=" * 60)

    targets = {
        "IC50": "IC50, mM",
        "CC50": "CC50, mM",
        "SI": "SI",
    }

    for submission_column, train_column in targets.items():

        print(f"Показатель {submission_column}:")

        print(
            f"  • [Train] "
            f"mean: {train_df[train_column].mean():.2f} | "
            f"min: {train_df[train_column].min():.2f} | "
            f"max: {train_df[train_column].max():.2f}"
        )

        print(
            f"  • [Test ] "
            f"mean: {submission_df[submission_column].mean():.2f} | "
            f"min: {submission_df[submission_column].min():.2f} | "
            f"max: {submission_df[submission_column].max():.2f}"
        )

        print("-" * 40)

    calculated_si = (
        submission_df["CC50"]
        / (submission_df["IC50"] + EPSILON)
    )

    mean_absolute_si_error = np.mean(
        np.abs(submission_df["SI"] - calculated_si)
    )

    correlation = np.corrcoef(
        submission_df["SI"],
        calculated_si,
    )[0, 1]

    print("Связь SI и CC50/IC50:")
    print(
        f"  • Среднее отклонение от формулы: "
        f"{mean_absolute_si_error:.2f}"
    )

    print(
        f"  • Корреляция SI с CC50/IC50: "
        f"{correlation:.4f}"
    )

    print("=" * 60 + "\n")