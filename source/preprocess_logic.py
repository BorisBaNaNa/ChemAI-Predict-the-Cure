import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer


def preprocess_data(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    feature_cols: list,
    method: str = "baseline",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_indices = X_train_raw.index
    test_indices = X_test_raw.index

    X_train = X_train_raw[feature_cols].copy()
    X_test = X_test_raw[feature_cols].copy()

    if method == "baseline":
        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        scaler = QuantileTransformer(output_distribution="normal", random_state=42)
        X_train_processed = scaler.fit_transform(X_train_imp)
        X_test_processed = scaler.transform(X_test_imp)

        X_train_df = pd.DataFrame(
            X_train_processed, index=train_indices, columns=feature_cols
        )
        X_test_df = pd.DataFrame(
            X_test_processed, index=test_indices, columns=feature_cols
        )

        return X_train_df, X_test_df

    elif method == "advanced":
        X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

        constant_and_quasi = [
            col for col in feature_cols if X_train[col].nunique() <= 1
        ]
        for col in feature_cols:
            if col not in constant_and_quasi:
                predominant_share = X_train[col].value_counts(normalize=True).values
                # Берем долю самого частого значения [0]
                if len(predominant_share) > 0 and predominant_share[0] > 0.99:
                    constant_and_quasi.append(col)

        active_features = [col for col in feature_cols if col not in constant_and_quasi]
        X_train = X_train[active_features]
        X_test = X_test[active_features]

        for col in active_features:
            if X_train[col].max() > 10000:
                min_val = X_train[col].min()
                shift = abs(min_val) if min_val < 0 else 0
                X_train[col] = np.log1p(X_train[col] + shift)
                X_test[col] = np.log1p(X_test[col] + shift)

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        X_train_df = pd.DataFrame(
            X_train_imp, index=train_indices, columns=active_features
        )
        X_test_df = pd.DataFrame(
            X_test_imp, index=test_indices, columns=active_features
        )

        return X_train_df, X_test_df

    else:
        raise ValueError(f"Неизвестный метод предобработки: {method}")
