import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

def preprocess_data(X_train_raw: pd.DataFrame, X_test_raw: pd.DataFrame, feature_cols: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Делаем копии, чтобы не портить исходные данные
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    
    # 1. Логарифмирование физических дескрипторов со скошенным распределением
    # Ищем признаки, где максимальные значения в десятки раз превышают медиану (тяжелые хвосты)
    skewed_features = []
    for col in feature_cols:
        if X_train[col].min() >= 0 and X_train[col].max() > 0:
            median = X_train[col].median()
            p95 = X_train[col].quantile(0.95)
            
            if median > 0 and (p95 / median) > 3:
                skewed_features.append(col)
                
    print(f"[Prep] Найдено признаков с тяжелыми хвостами для логарифмирования: {len(skewed_features)}")
    
    # Применяем log1p к этим признакам
    for col in skewed_features:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])
        
    # 2. Заполнение пропусков (Imputation)
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # 3. Бережное масштабирование через RobustScaler (устойчив к выбросам, сохраняет форму распределения)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)
    
    X_train_processed = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_processed = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    print(f"[Prep] Предобработка завершена. Форма данных: {X_train_processed.shape}")
    
    return X_train_processed, X_test_processed