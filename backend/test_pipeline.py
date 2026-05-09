from utils import download_data
from preprocess import prepare_data

df = download_data()

(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test
) = prepare_data(df)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)