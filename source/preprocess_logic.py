import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Константы
QUASI_CONSTANT_THRESHOLD = 0.99
LARGE_VALUE_THRESHOLD = 10000

# Предобработка
def preprocess_data(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    feature_cols: list,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_indices = X_train_raw.index
    test_indices = X_test_raw.index

    X_train = X_train_raw[feature_cols].copy()
    X_test = X_test_raw[feature_cols].copy()

    # CatBoost и статистические преобразования плохо работают
    # с inf значениями, поэтому заменяем их на NaN
    # перед дальнейшей обработкой.
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Удаление константных и квазиконстантных признаков
    constant_and_quasi = [
        col for col in feature_cols
        if X_train[col].nunique() <= 1
    ]

    for col in feature_cols:

        if col in constant_and_quasi:
            continue

        predominant_share = (
            X_train[col]
            .value_counts(normalize=True)
            .values
        )

        # Если одно значение занимает почти весь признак,
        # он считается квазиконстантным.
        if (
            len(predominant_share) > 0
            and predominant_share[0] > QUASI_CONSTANT_THRESHOLD
        ):
            constant_and_quasi.append(col)

    active_features = [
        col for col in feature_cols
        if col not in constant_and_quasi
    ]

    X_train = X_train[active_features]
    X_test = X_test[active_features]

    # Признаки с экстремально большим разбросом
    # могут доминировать при обучении.
    #
    # log1p стабилизирует распределение и уменьшает
    # влияние выбросов.
    for col in active_features:

        if X_train[col].max() > LARGE_VALUE_THRESHOLD:

            min_value = X_train[col].min()

            # log1p требует x >= -1,
            # поэтому отрицательные значения сдвигаются.
            shift = abs(min_value) if min_value < 0 else 0

            X_train[col] = np.log1p(
                X_train[col] + shift
            )

            X_test[col] = np.log1p(
                X_test[col] + shift
            )

    # Заполнение пропусков

    # Медиана устойчива к выбросам
    # и лучше подходит для химических дескрипторов,
    # чем среднее значение.
    imputer = SimpleImputer(strategy="median")

    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    X_train_df = pd.DataFrame(
        X_train_imputed,
        index=train_indices,
        columns=active_features,
    )

    X_test_df = pd.DataFrame(
        X_test_imputed,
        index=test_indices,
        columns=active_features,
    )

    return X_train_df, X_test_df