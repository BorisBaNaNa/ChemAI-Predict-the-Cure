import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')

print("ОПТИМИЗАЦИЯ ВЕСОВ АНСАМБЛЯ")

# Загрузка данных
script_dir = os.path.dirname(os.path.abspath(__file__))

train = pd.read_csv(os.path.join(script_dir, '..', 'data', 'train.csv'))
test = pd.read_csv(os.path.join(script_dir, '..', 'data', 'test.csv'))

feature_cols = [col for col in train.columns if col not in ['index', 'IC50, mM', 'CC50, mM', 'SI']]
X_train = train[feature_cols]
y_train = train[['IC50, mM', 'CC50, mM', 'SI']]
X_test = test[feature_cols]

# Предобработка
imputer = SimpleImputer(strategy='median')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

scaler = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

y_train_log = np.log1p(y_train)

# Параметры моделей
params_multi = {
    'iterations': 800,
    'learning_rate': 0.04,
    'depth': 6,
    'l2_leaf_reg': 3,
    'border_count': 128,
    'random_strength': 1,
    'bagging_temperature': 1,
    'od_type': 'Iter',
    'od_wait': 50,
    'loss_function': 'MultiRMSE',
    'verbose': False,
    'allow_writing_files': False
}

params_single = {
    'iterations': 800,
    'learning_rate': 0.05,
    'depth': 5,
    'l2_leaf_reg': 5,
    'border_count': 128,
    'random_strength': 1,
    'bagging_temperature': 1,
    'od_type': 'Iter',
    'od_wait': 50,
    'loss_function': 'RMSE',
    'verbose': False,
    'allow_writing_files': False
}

# Кросс-валидация для поиска оптимальных весов
print("\nПоиск оптимальных весов через кросс-валидацию...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
best_weight = 0.5
best_score = float('inf')

for weight_multi in np.arange(0.4, 0.71, 0.05):
    scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled), 1):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train_log.iloc[train_idx], y_train_log.iloc[val_idx]
        
        # Multi-output модель
        multi = CatBoostRegressor(**params_multi, random_seed=42)
        multi.fit(X_tr, y_tr)
        multi_pred = multi.predict(X_val)
        
        # Отдельные модели
        single_preds = []
        for target in range(3):
            single = CatBoostRegressor(**params_single, random_seed=42)
            single.fit(X_tr, y_tr.iloc[:, target])
            single_preds.append(single.predict(X_val))
        single_pred = np.column_stack(single_preds)
        
        # Ансамбль
        ensemble_pred = weight_multi * multi_pred + (1 - weight_multi) * single_pred
        
        # RMSE
        rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred, multioutput='raw_values'))
        scores.append(np.mean(rmse))
    
    avg_score = np.mean(scores)
    print(f"  weight_multi={weight_multi:.2f}, CV score={avg_score:.4f}")
    
    if avg_score < best_score:
        best_score = avg_score
        best_weight = weight_multi

print(f"\n✓ Оптимальный вес multi-output: {best_weight:.2f}")
print(f"  Оптимальный вес отдельных моделей: {1-best_weight:.2f}")

# Обучение финальных моделей с оптимальными весами
print("ОБУЧЕНИЕ ФИНАЛЬНЫХ МОДЕЛЕЙ")

# Multi-output
multi_model = CatBoostRegressor(**params_multi, random_seed=42)
multi_model.fit(X_train_scaled, y_train_log)
multi_preds = np.expm1(multi_model.predict(X_test_scaled))

# Отдельные модели (3 seeds)
seeds = [42, 123, 456]
single_preds_list = []

for seed in seeds:
    print(f"Seed {seed}...", end=" ", flush=True)
    preds = []
    for i in range(3):
        model = CatBoostRegressor(**params_single, random_seed=seed)
        model.fit(X_train_scaled, y_train_log.iloc[:, i])
        preds.append(model.predict(X_test_scaled))
    single_preds_list.append(np.column_stack(preds))
    print("✓")

single_preds_avg = np.mean(single_preds_list, axis=0)
single_preds = np.expm1(single_preds_avg)

# Ансамбль с оптимальными весами
final_preds = best_weight * multi_preds + (1 - best_weight) * single_preds

# Улучшенный пост-процессинг
print("УЛУЧШЕННЫЙ ПОСТ-ПРОЦЕССИНГ")

# Ограничение выбросов (по перцентилям)
for i in range(3):
    p99 = np.percentile(final_preds[:, i], 99)
    final_preds[:, i] = np.clip(final_preds[:, i], 1e-6, p99)

# SI корректировка с вычислением через логарифмы (более стабильно)
log_ic50 = np.log1p(final_preds[:, 0])
log_cc50 = np.log1p(final_preds[:, 1])
log_si_computed = log_cc50 - log_ic50
si_computed = np.expm1(log_si_computed)

# Адаптивное смешивание
ratio = final_preds[:, 1] / (final_preds[:, 0] + 1e-8)
diff_ratio = np.abs(final_preds[:, 2] - ratio) / (final_preds[:, 2] + 1e-8)
adaptive_weight = np.clip(diff_ratio, 0.1, 0.5)
final_preds[:, 2] = (1 - adaptive_weight) * final_preds[:, 2] + adaptive_weight * ratio

# Гарантируем, что SI >= 0.1
final_preds[:, 2] = np.maximum(final_preds[:, 2], 0.1)

print(f"Пост-процессинг применен")

# СОХРАНЕНИЕ
submission = pd.DataFrame({
    'index': test.index,
    'IC50': final_preds[:, 0],
    'CC50': final_preds[:, 1],
    'SI': final_preds[:, 2]
})

submission.to_csv('submission_optimized_weights.csv', index=False)

print("Результаты")
print(f"Файл submission_optimized_weights.csv создан!")
print(f"\nIC50: min={submission['IC50'].min():.2f}, max={submission['IC50'].max():.2f}, mean={submission['IC50'].mean():.2f}")
print(f"CC50: min={submission['CC50'].min():.2f}, max={submission['CC50'].max():.2f}, mean={submission['CC50'].mean():.2f}")
print(f"SI: min={submission['SI'].min():.2f}, max={submission['SI'].max():.2f}, mean={submission['SI'].mean():.2f}")

print(f"\nПроверка SI:")
print(f"  Предсказанный SI: {submission['SI'].mean():.2f}")
print(f"  Вычисленный CC50/IC50: {(submission['CC50'].mean() / submission['IC50'].mean()):.2f}")

print("\nМодель с оптимизированными весами готова!")