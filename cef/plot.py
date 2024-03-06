from typing import Any
from matplotlib import pyplot as plt
import pandas as pd


# グラフの描画を行うモジュール(未完成)


# 様々なグラフタイプで簡単にグラフを描画できるようにするクラス
class Plot:
    def __init__(
        self,
        data: Any,
        xlabel: str,
        ylabel: str,
        title: str = "タイトル",
        header: pd.Series = None,
    ) -> None:
        """
        self.data: データ
        self.header: データのヘッダー
        self.title: グラフのタイトル
        self.fig: グラフのインスタンス(4:3)
        self.ax: グラフの描写範囲を指定するインスタンス
        self.xlabel: x軸のラベル
        self.ylabel: y軸のラベル
        self.xlim: x軸の描写範囲
        self.ylim: y軸の描写範囲
        self.x: x軸のデータ
        self.y: y軸のデータ
        """
        self.data = data
        if header is None:
            pass
        else:
            self.header = header
        self.title = title
        self.fig = plt.figure(figsize=(4, 3))
        self.ax = self.fig.add_subplot(111)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = (0, 0)
        self.ylim = (0, 0)
        self.x = []
        self.y = []

    # プロットデータの選択
    def select_data(self, xname: str = None, yname: str = None) -> None:
        if isinstance(self.data, pd.DataFrame):
            self.x = self.data[xname]
            self.y = self.data[yname]
            print("Datatype is Dataframe")
        else:
            self.x = self.data[0]
            self.y = self.data[1]
            print("Datatype is List")

    # プロットデータの選択(0:平滑線, 1:散布図)
    def plot_type(self, c="red", type=0, mark="o", line="-", label="グラフ1"):
        if type == 0:
            self.ax.plot(self.x, self.y, linestyle=line, color=c, label=label)
        elif type == 1:
            self.ax.scatter(self.x, self.y, marker=mark, color=c)

    # プロット範囲の指定(min,max)
    def set_lim(self, xlim: tuple, ylim: tuple) -> None:
        self.xlim = xlim
        self.ylim = ylim

    # グラフの描画
    def plot(self, loc: str = "best") -> None:
        spines = 1
        self.ax.spines["top"].set_linewidth(spines)
        self.ax.spines["left"].set_linewidth(spines)
        self.ax.spines["bottom"].set_linewidth(spines)
        self.ax.spines["right"].set_linewidth(spines)
        self.ax.xaxis.set_ticks_position("both")
        self.ax.yaxis.set_ticks_position("both")

        self.ax.set_xlabel(self.xlabel, fontsize=20)
        self.ax.set_ylabel(self.ylabel, fontsize=20)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.tick_params(
            axis="both", which="both", direction="in", length=8, width=1
        )
        self.ax.legend(loc=loc, fontsize=10)
        plt.show()
