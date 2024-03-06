from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# 簡易的なプロット関数
def plot(x1, y1, x2, y2, scale_x, scale_y, loc, name="figure", xlabel="x", ylabel="y"):
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_subplot(111)
    ax.set_title(name, loc="center", y=1.05, fontsize=30)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xlim(scale_x.min(), scale_x.max())
    ax.set_ylim(min(scale_y), max(scale_y))
    ax.set_xticks(scale_x)
    ax.set_yticks(scale_y)
    ax.plot(x1, y1, color="k", label="pred")
    ax.scatter(x2, y2, color="r", label="data")
    plt.legend(loc=loc, fontsize=20)
    plt.show()


# データの読み込み
def load_data(path):
    file_path = path
    columns = ["Energy", "Monitor", "Detector"]
    data = pd.read_csv(
        file_path, sep=r"\s+", names=columns, dtype=float, usecols=[0, 19, 20]
    )

    return data


# 強度への変換
def calc_counts(data):
    data["counts"] = (data["Detector"] / data["Monitor"]) * 102336

    return data


# バックグラウンドの引き算
def res_backgrounds(data, backgrounds):
    data["norm_counts"] = data["counts"] - backgrounds["counts"]

    return data


def predict_abs_edges_spectrum(data, edges, exp_data):
    model_low = LinearRegression()
    model_high = LinearRegression()

    mu_yb_low = data[data["Energy"] < edges]
    mu_yb_high = data[data["Energy"] > edges]

    result_low = model_low.fit(
        mu_yb_low["Energy"].values.reshape(-1, 1),
        mu_yb_low["M/R"].values.reshape(-1, 1),
    )
    result_high = model_high.fit(
        mu_yb_high["Energy"].values.reshape(-1, 1),
        mu_yb_high["M/R"].values.reshape(-1, 1),
    )

    x = exp_data["Energy"].values
    exp_data_low = exp_data[exp_data["Energy"] < 8.943]
    exp_data_high = exp_data[exp_data["Energy"] >= 8.943]

    y_low = result_low.predict(exp_data_low["Energy"].values.reshape(-1, 1))
    y_high = result_high.predict(exp_data_high["Energy"].values.reshape(-1, 1))

    y_low = y_low
    y_high = y_high

    y = np.concatenate([y_high, y_low])
    x = np.ravel(x)
    y = np.ravel(y)

    mass_abs = pd.DataFrame({"Energy": x, "Yb": y})
    return mass_abs


N_A = 6.022e23


# 密度の計算
def calc_rho(volume, key):
    atom_mass = {
        "Yb": 173.054,
        "Pd": 106.42,
        "Pb": 207.2,
        "O": 15.9994,
        "YbPdPb": 0,
        "Yb2O3": 0,
    }
    atom_mass["YbPdPb"] = atom_mass["Yb"] + atom_mass["Pd"] + atom_mass["Pb"]
    atom_mass["Yb2O3"] = atom_mass["Yb"] * 2 + atom_mass["O"] * 3

    # rho:1mol g/cm^3
    return atom_mass[key] / (volume * N_A)


# 吸収係数の計算
def calc_mu(database, data_exp, key):
    key_database = key + "_data"
    mu_h = database.loc[0, key_database]
    mu_l = database.loc[110, key_database]
    i_h = data_exp.loc[0, "norm_counts"]
    i_l = data_exp.loc[110, "norm_counts"]
    mu = mu_l + (
        (2 * mu_l * (data_exp["norm_counts"] - i_l) * (mu_h - mu_l))
        / (
            (i_h - i_l) * (mu_h + mu_l)
            - (data_exp["norm_counts"] - i_l) * (mu_h - mu_l)
        )
    )
    key_exp = key + "_mu"
    data_exp[key_exp] = mu

    return data_exp


# データの読み込み
def load_absdata(path):
    file_path = path
    data = pd.read_csv(file_path, sep=r"\s+", dtype=float)
    print(data.columns)

    return data


# 強度への変換
def transform(data, monitor, detector):
    result = data[["Energy"]].copy()
    result["I"] = (data[detector] / data[monitor]) * 102336

    return result
