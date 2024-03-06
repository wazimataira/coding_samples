import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import lmfit as lf
import lmfit.models as lfm
import json
import inspect


# 簡易的なプロット関数
def plot_m(
    x,
    y,
    x_scale,
    y_scale,
    data=None,
    columns=[],
    colors=[],
    y_name="y",
    name="figure",
    xlabel="x",
    ylabel="y",
    save=False,
    path="./pics1.png",
):
    if len(columns) != len(colors):
        print("データ、カラー数が一致していません")
        return
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(name, loc="center", y=1.05, fontsize=30)
    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)
    ax.set_xticks(x_scale)
    ax.set_yticks(y_scale)
    ax.set_xlim(x_scale.min(), x_scale.max())
    ax.set_ylim(y_scale.min(), y_scale.max())
    ax.scatter(x, y, color="r", label=y_name, s=50)
    if data is not None:
        for name, c in zip(columns, colors):
            ax.plot(x, data[name], color=c, label=name, linewidth=3.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, fontsize=25)
    if save:
        plt.savefig(path, bbox_inches="tight")
    plt.show()


PI = np.pi


# ローレンツ関数
def lorenz_func(x, a, b, x0):
    x0 = x0 * np.ones_like(x)
    return (a * b / PI) / (((x - x0) ** 2) + b**2)


# ロジスティック関数
def logistic_func(x, a, b, x0):
    x0 = x0 * np.ones_like(x)
    return a / (1 + np.exp(-b * (x - x0)))


# lmfit内部のモデルの確認
def get_buildin_model():
    i = 1
    classes = map(lambda x: x[0], inspect.getmembers(lfm, inspect.isclass))
    for name in classes:
        if i % 10 == 0:
            print()
        print(name, end=", ")
        i += 1
    print()
    print("Model Details:https://lmfit.github.io/lmfit-py/builtin_models.html")


# モデルの外形を確認
def explain_model(cls):
    print(cls.param_names)
    x = np.linspace(-5, 5, 500)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, cls.func(x), linewidth=3.0)
    plt.show()


# 辞書への入力を行う関数
def add_dict(dic):
    for k in dic.keys():
        s = input(k + ":")
        if s != "n":
            dic[k] = int(s)

    return dic


# 辞書のリストを作成する関数
def make_dict_list(*args):
    dict_list = []
    for dic in args:
        dict_list.append(add_dict(dic))
    return dict_list


# パラメータに指定した値を代入する関数
def setting(**kwars):
    settings = dict.fromkeys(list(kwars.keys()))
    for name, p in kwars.items():
        dic = {k: v for k, v in p.items() if (v is not None)}
        settings[name] = dic

    return settings


def to_list(data):
    return (data.T).values.tolist()


# 拘束条件を設定する関数
# condition[(a,b),(String,String),…]:a=拘束したいパラメータ、b条件
def set_condition(params, condition):
    for k, v in condition:
        if len(v) != 0:
            params[k].set(expr=v)
    return params


# lmfitのモデルをラップするクラス
class Fit:
    def __init__(self, x, y, x_name, y_name, model, func_name, path="./fit1/"):
        self.x = x
        self.x_name = x_name
        self.y = y
        self.y_name = y_name
        self.model = model
        self.func_name = func_name
        self.path = path
        self.params_name = None
        self.params = None
        self.result = None
        self.best_params = None
        self.result_report = None
        self.fit_data = None
        self.result_stderr = None

        if not os.path.exists(path):
            os.makedirs(path)

        # 指定したモデルのパラメータを確認し、辞書を作成する関数

    def make_paramsdict(self, dic=False):
        self.params_name = self.model.param_names
        for k in self.params_name:
            print(k, end=",  ")
        if dic:
            params0 = dict.fromkeys(self.params_name, None)
            param_min = dict.fromkeys(self.params_name, None)
            param_max = dict.fromkeys(self.params_name, None)
            param_var = dict.fromkeys(self.params_name, True)
            condition = dict.fromkeys(self.params_name, "")

            return params0, param_min, param_max, param_var, condition

        return None

        # パラメータを設定する関数

    # params_list:[params0,param_max,param_min]
    def set_params(self, params_list, condition=None):
        if len(params_list) != 4:
            print("パラメータ条件が足りていません")
            return
        settings = setting(
            params0=params_list[0], param_max=params_list[1], param_min=params_list[2]
        )
        for k1 in settings.keys():
            print("{}:".format(k1))
            for k2, c in settings[k1].items():
                print(f"{k2}:{c}")
        self.params = self.model.make_params()
        # 初期値
        for name in settings["params0"].keys():
            self.params[name].set(value=settings["params0"][name])
        # 上限値
        for name in settings["param_max"].keys():
            self.params[name].set(max=settings["param_max"][name])
        # 下限値
        for name in settings["param_min"].keys():
            self.params[name].set(min=settings["param_min"][name])
        # パラメータを動かすかどうか
        if condition is not None:
            for name in self.params_name:
                self.params[name].set(vary=params_list[3][name])
            self.params = set_condition(self.params, condition)

        # パラメータの設定を保存する関数

    def save_settings(self, num=None):
        if num is not None:
            name = "setting_" + str(num) + ".csv"
        else:
            name = "setting_0.csv"
        columns = ["init_value", "min", "max", "expr", "vary"]
        p = []
        for k in self.params_name:
            p.append(
                [
                    self.params[k].init_value,
                    self.params[k].min,
                    self.params[k].max,
                    self.params[k].expr,
                    self.params[k].vary,
                ]
            )
        df = pd.DataFrame(p, index=self.params_name, columns=columns)
        df.to_csv(self.path + name)
        print("Parameter Settings Saved")

        # フィッティングを行う関数

    def fit(self):
        self.result = self.model.fit(
            x=self.x, data=self.y, params=self.params, method="leastsq"
        )
        self.best_params = self.result.best_values
        fig = plt.figure(figsize=(14, 12))
        self.result.plot(fig=fig)
        plt.show()

        # フィッティング結果を表示する関数

    def show_result(self):
        self.result_report = self.result.fit_report()
        print(self.result_report)
        comp = self.result.eval_components(x=self.x)
        df_temp = pd.DataFrame(
            np.array([self.x.values, self.y.values, self.result.best_fit]).T,
            columns=[self.x_name, self.y_name, "best fit"],
        )
        columns = self.x_name
        self.fit_data = pd.DataFrame(
            np.array(list(comp.values())).T, columns=self.func_name
        )
        self.fit_data = df_temp.join(self.fit_data)
        keys = list(self.result.result.params.keys())
        self.result_stderr = dict.fromkeys(keys)
        for k in keys:
            self.result_stderr[k] = self.result.result.params[k].stderr

        # フィッティング結果を保存する関数

    def save_result(self, num=None):
        if num is not None:
            name_params = self.path + "best_params_" + str(num) + ".json"
            name_report = self.path + "fit_report_" + str(num) + ".txt"
            name_fitting = self.path + "fitting_data" + str(num) + ".csv"
        else:
            name_params = self.path + "best_params_0.json"
            name_report = self.path + "fit_report_0.txt"
            name_fitting = self.path + "fitting_data_0.csv"

        with open(name_params, "w") as f:
            json.dump(self.best_params, f, indent=2)
        with open(name_report, "w") as f:
            f.write(self.result_report)
        self.fit_data.to_csv(name_fitting, index=False)

        print("Result Saved")
