import re
import json

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import interpolate


# データベースのデータの読み込み
def load_from_database(path):
    database = pd.read_csv(
        path,
        sep=r"\s+",
        index_col=0,
        skiprows=2,
        header=0,
        names=["WaveLength", "Energy", "f'", "f''"],
    )
    database.reset_index(inplace=True)
    database = database.rename(
        columns={
            "index": "WaveLength",
            "WaveLength": "Energy",
            "Energy": "f'",
            "f'": "f''",
            "f''": "temp",
        }
    )
    database = database.drop("temp", axis=1)
    return database


# データ範囲を指定してデータを抽出する関数
def sampling_data(base, data, column):
    base_min = base[column].min()
    base_max = base[column].max()
    result = data[(data[column] > base_min) & (data[column] < base_max)]
    return result


# 線形回帰を行うクラス
class Regression:
    def __init__(self, data):
        self.data = data
        self.model = None
        self.fit_model = None

    def linear_regression(self, x, y):
        self.model = LinearRegression()
        self.fit_model = self.model.fit(
            self.data[x].to_numpy().reshape(-1, 1),
            self.data[y].to_numpy().reshape(-1, 1),
        )

    def predict(self, x):
        pred = self.fit_model.predict(x.to_numpy().reshape(-1, 1))
        return np.squeeze(pred)


# データのプロットを行う関数
def plot_m(
    x,
    y,
    names,
    colors,
    x_scale=None,
    y_scale=None,
    name="figure",
    xlabel="x",
    ylabel="y",
    save=False,
    path="./pics1.png",
):

    if len(names) != len(colors):
        print("データ、カラー数が一致していません")
        return

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(name, loc="center", y=1.05, fontsize=30)

    if x_scale is None:
        x_min = min(list(map(lambda v: min(v), x)))
        x_max = max(list(map(lambda v: max(v), x)))
        diff = x_max - x_min
        x_scale = np.linspace(x_min - (diff / 20), x_max + (diff / 20), 7)

    if y_scale is None:
        y_max = max(list(map(lambda x: max(x), y)))
        y_min = min(list(map(lambda x: min(x), y)))
        diff = y_max - y_min
        y_scale = np.linspace(y_min - (diff / 20), y_max + (diff / 20), 7)

    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_xticks(x_scale)
    ax.set_ylabel(ylabel, fontsize=25)
    ax.set_yticks(y_scale)
    ax.set_xlim(x_scale.min(), x_scale.max())
    ax.set_ylim(y_scale.min(), y_scale.max())

    for name, c, value_x, value_y in zip(names, colors, x, y):
        ax.plot(value_x, value_y, color=c, label=name, linewidth=1.5)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize=25)

    if save:
        plt.savefig(path, bbox_inches="tight")

    plt.show()


# 粉末X線回折データの計算ファイルからデータを読み込むクラス
class XRDdata:
    def __init__(self, path):
        self.file = path
        self.file_length = 0
        self.data_start = []
        self.data_end = []
        self.data_range = None
        self.data = None

        # データの詳細を表示

    def detail_data(self):
        pattern = r"\*\*\*(.*?)\*\*\*"
        with open(self.file, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            self.file_length = len(lines)
            print("lines:", self.file_length)
            self.data_start = [
                (i, line) for i, line in enumerate(lines) if re.match(pattern, line)
            ]
            self.data_end = [
                (i, line) for i, line in enumerate(lines) if line.startswith("0")
            ]
            print("-----start line-----")
            for i, (idx, line) in enumerate(self.data_start):
                print(f"{i}番目,{idx+1}行目:{line}")
            print("-----end line-----")
            for i, (idx, line) in enumerate(self.data_end):
                print(f"{i}番目,{idx+1}行目:{line}")

        # データの範囲を指定してデータを抽出

    def select_data(self, num_st, num_en):
        forward = [row for row in range(1, self.data_start[num_st][0] + 3)]
        backward = [
            row for row in range(self.data_end[num_en][0], self.file_length + 1)
        ]
        self.data_range = forward + backward
        self.data = pd.read_csv(
            self.file, sep="\s+", index_col=0, skiprows=self.data_range
        )
        df = self.data.copy()
        df[["h", "k", "l"]] = df[["h", "k", "l"]].astype(str)
        df["hkl"] = df["h"] + df["k"] + df["l"]
        df.drop(
            ["Phase", "Code", "|F(magn)|", "POF", "FWHM", "h", "k", "l"],
            axis=1,
            inplace=True,
        )
        df.set_index("hkl", inplace=True)

        return df


# json形式のデータを読み込む関数
def load_jsondata(path):
    with open(path, "r") as f:
        json_data = json.load(f)
    return json_data


# ガウス関数
def gaussian(b, s):
    return np.exp(-b * (s**2))


# X線散乱計算クラス
class XRayScatteringCalculator:
    df_f = pd.read_csv("./data/f_keisuu.csv", index_col=0)

    def __init__(
        self,
        atom_list=None,
        lattice_parameter=None,
        site=None,
        reflection_data=None,
        energy=0,
    ):
        """

        atom_list:物質中の原子の種類(リスト-[O,H,Yb2+,...])
        lattice_parameter:格子定数(辞書-{定数名:値})
        site:各原子の位置(辞書-{原子種類:{原子ラベル:位置[x,y,z]}})
        reflection_data:各hkl面においてのX線散乱データ(DataFrame-hkl,d,θ,I_cal,F)
        energy:考える(測定した)エネルギー範囲(Pandas Series)

        """

        self.atom_list = atom_list
        self.lattice_parameter = lattice_parameter
        self.site = site.copy()
        self.model = dict.fromkeys(atom_list)
        self.reflection_data = reflection_data
        self.energy = energy
        self.length = len(energy)
        self.lattice_basis = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32"
        )
        self.reciprocal_lattice_basis = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32"
        )
        self.f0 = dict.fromkeys(atom_list)
        self.f1 = dict.fromkeys(atom_list)
        self.f2 = dict.fromkeys(atom_list)
        self.f_atom = dict.fromkeys(atom_list)

        temp = []
        for k in self.site.keys():
            temp = temp + list(self.site[k].keys())

        self.f_site = dict.fromkeys(
            temp,
            pd.DataFrame(
                {
                    "Energy": self.energy.to_numpy(),
                    "f_real": np.zeros(self.length),
                    "f_img": np.zeros(self.length),
                }
            ),
        )
        self.result = pd.DataFrame(
            {
                "Energy": self.energy.to_numpy(),
                "F_Real": np.zeros(self.length),
                "F_Img": np.zeros(self.length),
            }
        )

        # サイトの入れ替え

    # names:dict型{入れ替えるキー(atom,label):入れ替え先のキー(atom,label)}
    def swap(self, atom_names, label_names):
        site1 = self.site[atom_names[0]].pop(label_names[0])
        site2 = self.site[atom_names[1]].pop(label_names[1])
        self.site[atom_names[1]][label_names[0]] = site1
        self.site[atom_names[0]][label_names[1]] = site2

        # 六方晶形での直交座標系への変換

    def to_cartesian(self):
        rad = np.radians(90 - self.lattice_parameter["gamma"])
        rotate = np.array([np.sin(rad), np.cos(rad), 0])
        self.lattice_basis[:, 1] = self.lattice_parameter["b"] * (
            self.lattice_basis @ rotate
        )
        self.lattice_basis[:, 0] = (
            self.lattice_parameter["a"] * self.lattice_basis[:, 0]
        )
        self.lattice_basis[:, 2] = (
            self.lattice_parameter["c"] * self.lattice_basis[:, 2]
        )

    def to_reciprocal(self):
        basis = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
        self.reciprocal_lattice_basis[:, 0] = (
            2
            * np.pi
            * (
                (np.cross(self.lattice_basis[:, 1], self.lattice_basis[:, 2]))
                / (
                    np.matmul(
                        self.lattice_basis[:, 0],
                        np.cross(self.lattice_basis[:, 1], self.lattice_basis[:, 2]),
                    )
                )
            )
        )
        self.reciprocal_lattice_basis[:, 1] = (
            2
            * np.pi
            * (
                (np.cross(self.lattice_basis[:, 2], self.lattice_basis[:, 0]))
                / (
                    np.matmul(
                        self.lattice_basis[:, 1],
                        np.cross(self.lattice_basis[:, 2], self.lattice_basis[:, 0]),
                    )
                )
            )
        )
        self.reciprocal_lattice_basis[:, 2] = (
            2
            * np.pi
            * (
                (np.cross(self.lattice_basis[:, 0], self.lattice_basis[:, 1]))
                / (
                    np.matmul(
                        self.lattice_basis[:, 2],
                        np.cross(self.lattice_basis[:, 0], self.lattice_basis[:, 1]),
                    )
                )
            )
        )

        # 面間隔の計算

    def calc_d(self, h=0, k=0, l=0):
        g = (
            h * self.reciprocal_lattice_basis[:, 0]
            + k * self.reciprocal_lattice_basis[:, 1]
            + l * self.reciprocal_lattice_basis[:, 2]
        )
        d = (2 * np.pi) / np.linalg.norm(g, ord=2)
        return d

        # f0の計算

    def calc_f0(self, d):
        s = 1 / (2 * d)
        for atom in self.f0.keys():
            self.f0[atom] = (
                self.df_f.loc[atom, "a_1"] * gaussian(self.df_f.loc[atom, "b_1"], s)
                + self.df_f.loc[atom, "a_2"] * gaussian(self.df_f.loc[atom, "b_2"], s)
                + self.df_f.loc[atom, "a_3"] * gaussian(self.df_f.loc[atom, "b_3"], s)
                + self.df_f.loc[atom, "a_4"] * gaussian(self.df_f.loc[atom, "b_4"], s)
                + self.df_f.loc[atom, "c"]
            )

        # f1,f2のデータを設定

    def set_f1f2(self, atom, data=None, anomaly=True):
        if anomaly:
            self.f1[atom] = data[["Energy", "f1"]]
            self.f2[atom] = data[["Energy", "f2"]]
        else:
            self.f1[atom] = pd.DataFrame(
                {"Energy": self.energy.to_numpy(), "f1": np.zeros(self.length)}
            )
            self.f2[atom] = pd.DataFrame(
                {"Energy": self.energy.to_numpy(), "f2": np.zeros(self.length)}
            )

        # 散乱因子の実部と虚部に分ける

    def sum_f(self):
        for atom in self.f0.keys():
            self.f_atom[atom] = pd.merge(self.f1[atom], self.f2[atom], on="Energy")
            self.f_atom[atom]["f_real"] = self.f_atom[atom].apply(
                lambda row: self.f0[atom] + row["f1"], axis=1
            )
            self.f_atom[atom]["f_img"] = self.f_atom[atom]["f2"].copy()

        # 価数モデルの作成

    def dicision_model(self):
        for k in self.site.keys():
            self.model[k] = list(self.site[k].keys())

        temp = {}
        for atom, labels in self.model.items():
            f_real = self.f_atom[atom]["f_real"].to_numpy()
            f_img = self.f_atom[atom]["f_img"].to_numpy()
            for label in labels:
                temp[label] = pd.DataFrame(
                    {
                        "Energy": self.energy.to_numpy(),
                        "f_real": f_real.copy(),
                        "f_img": f_img.copy(),
                    }
                )

        self.f_site = temp

        # 構造因子の計算

    def calc_F(self, h=0, k=0, l=0):
        F_real_site = {}
        F_img_site = {}
        for atom, labels in self.model.items():
            for label in labels:
                exp_real = np.cos(
                    2
                    * np.pi
                    * (
                        h * self.site[atom][label][0]
                        + k * self.site[atom][label][1]
                        + l * self.site[atom][label][2]
                    )
                )
                exp_img = np.sin(
                    2
                    * np.pi
                    * (
                        h * self.site[atom][label][0]
                        + k * self.site[atom][label][1]
                        + l * self.site[atom][label][2]
                    )
                )
                F_real = (
                    exp_real * self.f_site[label]["f_real"]
                    - exp_img * self.f_site[label]["f_img"]
                )
                F_img = (
                    exp_img * self.f_site[label]["f_real"]
                    + exp_real * self.f_site[label]["f_img"]
                )
                F_real_site[label] = F_real.copy()
                F_img_site[label] = F_img.copy()

        for label in self.f_site.keys():
            self.f_site[label]["F_real"] = F_real_site[label].copy()
            self.f_site[label]["F_img"] = F_img_site[label].copy()

        sum = pd.DataFrame(
            {
                "Energy": self.energy.to_numpy(),
                "F_Real": np.zeros(self.length),
                "F_Img": np.zeros(self.length),
            }
        )
        for label in self.f_site.keys():
            sum["F_Real"] += self.f_site[label]["F_real"].to_numpy()
            sum["F_Img"] += self.f_site[label]["F_img"].to_numpy()

        self.result = sum

        # 強度の計算|F|^2

    def calc_intensity(self):
        intensity = pd.DataFrame(
            {"Energy": self.energy.to_numpy(), "|F|": np.zeros(self.length)}
        )
        intensity["|F|"] = self.result.apply(
            lambda row: (row["F_Real"] ** 2) + (row["F_Img"] ** 2), axis=1
        )
        return intensity


# ピークの取得
def get_peak(data, target, invert=False):
    if invert:
        peak = data[data[target] == data[target].min()]
    else:
        peak = data[data[target] == data[target].max()]

    return peak["Energy"].to_numpy()[0]


# データの正規化
def max_normalization(data_th, data_exp):
    max_th = data_th["|F|"].max()
    max_exp = data_exp["norm_counts"].max()
    data_th["|F|"] = data_th["|F|"].apply(lambda x: x / max_th)
    data_exp["norm_counts"] = data_exp["norm_counts"].apply(lambda x: x / max_exp)

    return data_th, data_exp


# データのシフト
def shift(data_th, data_exp):
    scale_exp = data_exp["norm_counts"].max() - data_exp["norm_counts"].min()
    scale_th = data_th["|F|"].max() - data_th["|F|"].min()
    scale = scale_exp / scale_th
    data_th["|F|"] = data_th["|F|"].apply(lambda x: x * scale)

    return data_th


# データの補完
def augumentation(observed_x, observed_y, fit_x):
    df = pd.DataFrame(fit_x.to_numpy(), columns=[fit_x.name])
    fitted = interpolate.interp1d(observed_x.to_numpy(), observed_y.to_numpy())
    df[observed_y.name] = fitted(fit_x.to_numpy())
    df.sort_values(fit_x.name, ignore_index=True)
    return df
