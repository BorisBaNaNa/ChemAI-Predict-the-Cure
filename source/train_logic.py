import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")


def train_and_predict_ensemble(
    X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame
) -> pd.DataFrame:
    y_train_log = np.log1p(y_train)

    params_multi = {
        "iterations": 800,
        "learning_rate": 0.04,
        "depth": 6,
        "l2_leaf_reg": 3,
        "border_count": 128,
        "random_strength": 1,
        "bagging_temperature": 1,
        "od_type": "Iter",
        "od_wait": 50,
        "loss_function": "MultiRMSE",
        "verbose": False,
        "allow_writing_files": False,
    }

    params_single = {
        "iterations": 800,
        "learning_rate": 0.05,
        "depth": 5,
        "l2_leaf_reg": 5,
        "border_count": 128,
        "random_strength": 1,
        "bagging_temperature": 1,
        "od_type": "Iter",
        "od_wait": 50,
        "loss_function": "RMSE",
        "verbose": False,
        "allow_writing_files": False,
    }

    X_tr_arr = np.array(X_train)
    X_te_arr = np.array(X_test)

    # Сюда будем собирать чистые предсказания ансамбля на валидации для оценки качества
    oof_preds = np.zeros_like(y_train_log.values)

    print("\n[ML] Запуск 5-Fold кросс-валидации для поиска весов и оценки качества...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_weight = 0.5
    best_score = float("inf")

    # Поиск весов
    for weight_multi in np.arange(0.4, 0.71, 0.05):
        fold_scores = []
        temp_oof = np.zeros_like(y_train_log.values)

        for train_idx, val_idx in kf.split(X_tr_arr):
            X_tr, X_val = X_tr_arr[train_idx], X_tr_arr[val_idx]
            y_tr, y_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]

            multi = CatBoostRegressor(**params_multi, random_seed=42)
            multi.fit(X_tr, y_tr)
            multi_pred = multi.predict(X_val)

            single_preds = []
            for target in range(3):
                single = CatBoostRegressor(**params_single, random_seed=42)
                single.fit(X_tr, y_tr.iloc[:, target])
                single_preds.append(single.predict(X_val))
            single_pred = np.column_stack(single_preds)

            ensemble_pred = weight_multi * multi_pred + (1 - weight_multi) * single_pred
            temp_oof[val_idx] = ensemble_pred

            rmse = np.sqrt(
                mean_squared_error(y_val, ensemble_pred, multioutput="raw_values")
            )
            fold_scores.append(np.mean(rmse))

        avg_score = np.mean(fold_scores)
        if avg_score < best_score:
            best_score = avg_score
            best_weight = weight_multi
            oof_preds = temp_oof  # сохраняем лучшие предсказания для детального отчета

    # Вывод объективной метрики качества (CV Score)
    print("\n" + "=" * 50)
    print(" МЕТРИКИ КАЧЕСТВА МОДЕЛИ НА КРОСС-ВАЛИДАЦИИ (Log-Scale RMSE)")
    print("=" * 50)
    target_names = ["IC50", "CC50", "SI"]
    for i, name in enumerate(target_names):
        target_rmse = np.sqrt(
            mean_squared_error(y_train_log.iloc[:, i], oof_preds[:, i])
        )
        print(f"  • Ошибка на {name:4}: {target_rmse:.4f}")
    print(f"  --> ИТОГОВЫЙ СРЕДНИЙ SCORE: {best_score:.4f}")
    print(f"  --> Оптимальный вес multi-output: {best_weight:.2f}")
    print("=" * 50 + "\n")

    # Финальное обучение
    print("[ML] Обучение финальных моделей на полной выборке...")
    multi_model = CatBoostRegressor(**params_multi, random_seed=42)
    multi_model.fit(X_tr_arr, y_train_log)
    multi_preds = np.expm1(multi_model.predict(X_te_arr))

    seeds = [42, 123, 456]
    single_preds_list = []
    for seed in seeds:
        preds = []
        for i in range(3):
            model = CatBoostRegressor(**params_single, random_seed=seed)
            model.fit(X_tr_arr, y_train_log.iloc[:, i])
            preds.append(model.predict(X_te_arr))
        single_preds_list.append(np.column_stack(preds))

    single_preds_avg = np.mean(single_preds_list, axis=0)
    single_preds = np.expm1(single_preds_avg)

    final_preds = best_weight * multi_preds + (1 - best_weight) * single_preds

    # Пост-процессинг
    for i in range(3):
        p99 = np.percentile(final_preds[:, i], 99)
        final_preds[:, i] = np.clip(final_preds[:, i], 1e-6, p99)

    ratio = final_preds[:, 1] / (final_preds[:, 0] + 1e-8)
    diff_ratio = np.abs(final_preds[:, 2] - ratio) / (final_preds[:, 2] + 1e-8)
    adaptive_weight = np.clip(diff_ratio, 0.1, 0.5)
    final_preds[:, 2] = (1 - adaptive_weight) * final_preds[
        :, 2
    ] + adaptive_weight * ratio
    final_preds[:, 2] = np.maximum(final_preds[:, 2], 0.1)

    sub = pd.DataFrame(
        {
            "index": X_test.index,
            "IC50": final_preds[:, 0],
            "CC50": final_preds[:, 1],
            "SI": final_preds[:, 2],
        }
    )

    return sub

def check_submission_sanity(sub_df: pd.DataFrame, train_df: pd.DataFrame):
    print("\n" + "="*50)
    print(" АНАЛИЗ ФИЗИЧЕСКОЙ АДЕКВАТНОСТИ ПРЕДСКАЗАНИЙ С ТЕСТА")
    print("="*50)
    
    targets = {
        'IC50': 'IC50, mM',
        'CC50': 'CC50, mM',
        'SI': 'SI'
    }
    
    for t_sub, t_train in targets.items():
        print(f"Показатель {t_sub}:")
        print(f"  • [Train] реальный mean: {train_df[t_train].mean():.2f} | min: {train_df[t_train].min():.2f} | max: {train_df[t_train].max():.2f}")
        print(f"  • [Test ] предсказ mean: {sub_df[t_sub].mean():.2f} | min: {sub_df[t_sub].min():.2f} | max: {sub_df[t_sub].max():.2f}")
        print("-" * 30)
        
    # Проверка формульного соответствия SI = CC50 / IC50
    calculated_si = sub_df['CC50'] / (sub_df['IC50'] + 1e-8)
    mae_si_logic = np.mean(np.abs(sub_df['SI'] - calculated_si))
    
    print(f"Связь показателей:")
    print(f"  • Среднее отклонение предсказанного SI от формулы (CC50/IC50): {mae_si_logic:.2f}")
    print("="*50 + "\n")