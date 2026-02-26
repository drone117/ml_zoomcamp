import pandas as pd
import numpy as np
import requests


def download():
    with open("car_fuel_efficiency.csv", "wb") as car_file:
        res = requests.get(
            url="https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv",
        )
        car_file.write(res.content)


def homework_1():
    with open("car_fuel_efficiency.csv", "rb") as f:
        df = pd.read_csv(f)
        print(f"total values: {df.count().max()}")
        print(f"unique fuel types count: {df["fuel_type"].nunique()}")
        print(
            f"number of columns with na: {(df.isna().sum(axis=0, min_count=1) > 0).sum()}"
        )
        print(
            f"max mpg in Asia: {df[["origin", "fuel_efficiency_mpg"]].groupby("origin").fuel_efficiency_mpg.max().at["Asia"]}"
        )
        hp = df.horsepower
        print(f"median HP: {hp.median()}")
        print(f"most frequent value: {hp.mode()[0]}")
        print(f"median with filled na: {hp.fillna(value=hp.mode()[0]).median()}")
        lr = df[df["origin"] == "Asia"]
        X = np.array(lr[["vehicle_weight", "model_year"]].head(n=7))
        XT = X.T
        XTX = XT @ X
        XTX = np.linalg.inv(XTX)
        y = [1100, 1300, 800, 900, 1000, 1100, 1200]
        w = XTX @ XT @ y
        print(f"linear regression result: {w.sum()}")


homework_1()
