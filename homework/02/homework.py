import numpy as np
import pandas as pd

import seaborn as sns


def prepare_x(data, features, fill_with_zeros=True, value_to_fill=0.0):
    if fill_with_zeros:
        value_to_fill = 0.0
    data = data.copy()
    features = features.copy()
    data["age"] = max(data["model_year"]) - data["model_year"]
    features.append("age")
    df_num = data[features]
    df_num = df_num.fillna(value=value_to_fill)
    result = df_num.values
    return result


def train_linear_regression(features_matrix, target_values):
    ones = np.ones(features_matrix.shape[0])
    X = np.column_stack([ones, features_matrix])
    XTX = X.T @ X
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv @ X.T @ target_values
    return w_full[0], w_full[1:]


def train_linear_regression_regularized(features_matrix, target_values, r=0.001):
    ones = np.ones(features_matrix.shape[0])
    X = np.column_stack([ones, features_matrix])
    XTX = X.T @ X
    XTX = XTX + r * np.eye(XTX.shape[0])  # add regularization
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv @ X.T @ target_values
    return w_full[0], w_full[1:]


def rmse(target_values, target_values_pred):
    error = target_values - target_values_pred
    se = error**2
    mse = se.mean()
    return np.sqrt(mse)


df = pd.read_csv(
    "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"
)
columns = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
    "fuel_efficiency_mpg",
]

df = df[columns]
df.columns = df.columns.str.lower().str.replace(" ", "_")
all_str = list(df.dtypes[df.dtypes == "string"].index)
df[all_str] = df[all_str].apply(
    lambda x: x.str.lower().str.replace(" ", "_").str.strip()
)
print(df.isna().sum() > 0)
print(df["horsepower"].mean())
n = len(df)
n_val = int(n * 0.2)
n_test = int(n * 0.2)
n_train = sum([n, -n_val, -n_test])
idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train : n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val :]]
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.fuel_efficiency_mpg.values
y_val = df_val.fuel_efficiency_mpg.values
y_test = df_test.fuel_efficiency_mpg.values
del df_train["fuel_efficiency_mpg"]
del df_val["fuel_efficiency_mpg"]
del df_test["fuel_efficiency_mpg"]
features = ["engine_displacement", "horsepower", "vehicle_weight"]
rmse_val = 0
better_with_zeroes = False
mean = df_train.horsepower.mean()
for train_method in [True, False]:
    # Training part
    X_train = prepare_x(
        df_train, features, fill_with_zeros=train_method, value_to_fill=mean
    )
    w0, w = train_linear_regression(X_train, y_train)
    # Validation part
    X_val = prepare_x(df_val, features)
    y_pred = w0 + X_val @ w
    # Get RMSE
    rmse_curr = rmse(y_val, y_pred)
    if rmse_curr < rmse_val or rmse_val == 0:
        rmse_val = rmse_curr
        better_with_zeroes = train_method
print(
    f"min value of RMSE is {rmse_val.round(2)=} and is it better with zeroes? {better_with_zeroes=}"
)
min_r = 0
rmse_val = 0
for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    # Training part
    X_train = prepare_x(
        df_train,
        features,
        fill_with_zeros=True,
    )
    w0, w = train_linear_regression_regularized(X_train, y_train, r=r)
    # Validation part
    X_val = prepare_x(df_val, features)
    y_pred = w0 + X_val @ w
    # Get RMSE
    rmse_curr = rmse(y_val, y_pred).round(2)
    if rmse_curr < rmse_val or rmse_val == 0:
        rmse_val = rmse_curr
        min_r = r
print(f"min rmse value: {rmse_val}, and min regularization value {min_r}")
rmse_scores = []
for seed_value in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    idx = np.arange(n)
    np.random.seed(seed_value)
    np.random.shuffle(idx)
    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train : n_train + n_val]]
    df_test = df.iloc[idx[n_train + n_val :]]
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    y_train = df_train.fuel_efficiency_mpg.values
    y_val = df_val.fuel_efficiency_mpg.values
    y_test = df_test.fuel_efficiency_mpg.values
    del df_train["fuel_efficiency_mpg"]
    del df_val["fuel_efficiency_mpg"]
    del df_test["fuel_efficiency_mpg"]
    features = ["engine_displacement", "horsepower", "vehicle_weight"]
    # Training part
    X_train = prepare_x(
        df_train,
        features,
        fill_with_zeros=True,
    )
    w0, w = train_linear_regression(X_train, y_train)
    # Validation part
    X_val = prepare_x(df_val, features)
    y_pred = w0 + X_val @ w
    # Get RMSE
    rmse_curr = rmse(y_val, y_pred)
    rmse_scores.append(rmse_curr)
print(f"std deviation is {round(np.std(rmse_scores),3)}")
idx = np.arange(n)
np.random.seed(9)
np.random.shuffle(idx)
df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train : n_train + n_val]]
df_test = df.iloc[idx[n_train + n_val :]]
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
y_train = df_train.fuel_efficiency_mpg.values
y_val = df_val.fuel_efficiency_mpg.values
y_test = df_test.fuel_efficiency_mpg.values
del df_train["fuel_efficiency_mpg"]
del df_val["fuel_efficiency_mpg"]
del df_test["fuel_efficiency_mpg"]
features = ["engine_displacement", "horsepower", "vehicle_weight"]
df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)
y_full_train = np.concatenate([y_train, y_val])

# Training part
X_full_train = prepare_x(
    df_full_train,
    features,
    fill_with_zeros=True,
)
w0, w = train_linear_regression_regularized(X_full_train, y_full_train, r=0.001)
# Validation part
X_test = prepare_x(df_test, features)
y_pred = w0 + X_test @ w
print(rmse(y_test, y_pred).round(3))

car = df_test.iloc[20].to_dict()
df_small = pd.DataFrame([car])
X_small = prepare_x(df_small, features)
y_pred = w0 + X_small @ w
print(y_pred[0], y_test[20])
