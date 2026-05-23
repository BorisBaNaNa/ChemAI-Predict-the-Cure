from preprocess_logic import preprocess_data
from train_logic import check_submission_sanity, train_and_predict_ensemble
import pandas as pd
import os

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    train = pd.read_csv(os.path.join(script_dir, '..', 'data', 'train.csv'))
    test = pd.read_csv(os.path.join(script_dir, '..', 'data', 'test.csv'))

    # Выделение сырых признаков
    feature_cols = [col for col in train.columns if col not in ['index', 'IC50, mM', 'CC50, mM', 'SI']]
    X_train_raw = train[feature_cols]
    y_train_raw = train[['IC50, mM', 'CC50, mM', 'SI']]
    X_test_raw = test[feature_cols]

    X_tr_processed, X_te_processed = preprocess_data(X_train_raw, X_test_raw, feature_cols)
    submission = train_and_predict_ensemble(X_tr_processed, y_train_raw, X_te_processed)
    check_submission_sanity(submission, train)

    submission.to_csv('submission_optimized_weights.csv', index=False)
    print("[Успех] Файл submission_optimized_weights.csv успешно сохранен!")