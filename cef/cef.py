import torch
import matrix

# 結晶場計算及び物理量の計算を行うモジュール


# 最大のLの値を返す
def max_l(num: int) -> int:
    sum = 0
    for i in range(3, 3 - num, -1):
        sum += i

    return sum


# {name:(L,S)}で表されるL,Sの値の辞書を作成
def make_lslist() -> dict:
    e_num = 14
    s = 1 / 2
    materials = [
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
    ]
    ls_list = []
    for i in range(e_num):
        if i <= 7:
            ls_list.append((max_l(i), s * i))
        else:
            ls_list.append((max_l(abs(i - e_num)), s * abs(i - e_num)))

    ls_dict = dict(zip(materials, ls_list))

    return ls_dict


# rareearth{name:(J多重項の大きさ、縮退度)}
def make_rareearth_dict() -> dict:
    value = [
        (5 / 2.0, 6),
        (4, 9),
        (9 / 2.0, 10),
        (4, 9),
        (5 / 2.0, 7),
        (7 / 2.0, 8),
        (6, 13),
        (15 / 2.0, 16),
        (8, 17),
        (15 / 2.0, 16),
        (6, 13),
        (7 / 2.0, 8),
    ]
    materials = ["Ce", "Pr", "Nd", "Pm", "Sm", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb"]
    rareearth = dict(zip(materials, value))
    return rareearth


# 行列の累乗計算(t:tensor,n:指数)
def pow(t: torch.Tensor, n: int = 2) -> torch.Tensor:
    result = torch.mm(t, t)
    for i in range(2, n):
        result = torch.mm(result, t)

    return result


# 結晶場計算を行うクラス
class CrystalField:
    """
    クラス変数

    k_b:ボルツマン定数
    mu_b:ボーア磁子
    rareearth:レアアースの名前とJの大きさ、縮退度の辞書
    ls_number:レアアースの名前とL,Sの値の辞書

    """

    # k_B:erg/K
    k_b = 1.380649e-16
    # mu_B:(erg/Oe^-1)
    mu_b = 9.27401008e-21
    rareearth = make_rareearth_dict()
    ls_number = make_lslist()

    def __init__(self, r: str) -> None:
        # r=rare earth name
        """
        インスタンス変数

        self.ls_number:レアアースrのL,Sの値
        self.rareearth:レアアースrのJのタプル(大きさ、縮退度)

        以下は全てtorch.Tensor型
        {
        self.identity:レアアースrのJの行列表現と同じサイズの単位行列
        self.zero:レアアースrのJの行列表現と同じサイズの零行列
        self.j_x:レアアースrのJxの行列表現
        self.j_y:レアアースrのJyの行列表現
        self.j_z:レアアースrのJzの行列表現
        self.j:レアアースrのJの行列表現
        self.j_plus:レアアースrのJ+の行列表現
        self.j_minus:レアアースrのJ-の行列表現
        self.o:レアアースrのスティーブンス演算子の辞書
        self.h:レアアースrのハミルトニアン
        }
        self.temp:温度
        self.mag_field:外部磁場(大きさ、方向:torch.Tensor型)
        self.diaghamiltonian:レアアースrのハミルトニアンの対角化を行うクラスのインスタンス
        self.energy:レアアースrのハミルトニアンの固有値
        self.gfactor:レアアースrのg因子
        self.z:分配関数
        self.mu:磁気モーメント
        self.m:磁化
        """

        self.ls_number = CrystalField.ls_number[r]
        self.rareearth = CrystalField.rareearth[r]
        self.identity = torch.eye(self.rareearth[1], dtype=torch.float64)
        self.zero = torch.zeros(
            self.rareearth[1], self.rareearth[1], dtype=torch.float64
        )

        self.j_x = torch.zeros(
            self.rareearth[1], self.rareearth[1], dtype=torch.float64
        )
        self.j_y = torch.zeros(
            self.rareearth[1], self.rareearth[1], dtype=torch.complex128
        )
        self.j_z = torch.eye(self.rareearth[1], dtype=torch.float64)
        self.j = torch.eye(self.rareearth[1], dtype=torch.float64)
        self.j_plus = torch.zeros(
            self.rareearth[1], self.rareearth[1], dtype=torch.float64
        )
        self.j_minus = torch.zeros(
            self.rareearth[1], self.rareearth[1], dtype=torch.float64
        )

        self.o = dict.fromkeys(
            [
                "O20",
                "O22",
                "O40",
                "O42",
                "O43",
                "O44",
                "O60",
                "O62",
                "O63",
                "O64",
                "O66",
            ],
            0,
        )

        self.h = torch.zeros(self.rareearth[1], self.rareearth[1], dtype=torch.float64)

        self.temp = 0
        self.mag_field = (0, torch.tensor([0, 0, 0], dtype=torch.float64))

        self.diag_hamiltonian = None
        self.energy = []

        self.gfactor = 0

        self.z = 0
        self.mu = []

        self.m = 0

    # レアアースのJzの行列表現を作成する関数
    def make_jz(self) -> None:
        j, n = self.rareearth
        for i in range(n):
            self.j_z[i][i] = j
            j -= 1

    # レアアースのJの行列表現を作成する関数
    def make_j(self) -> None:
        j, n = self.rareearth
        self.j = j * self.j

    # レアアースのJ+の行列表現を作成する関数
    def make_jplus(self) -> None:
        j, n = self.rareearth
        m = j - 1
        for i in range(n - 1):
            self.j_plus[i][i + 1] = torch.sqrt(
                torch.tensor(((j - m) * (j + m + 1)), dtype=torch.float64)
            )
            m -= 1

    # レアアースのJ-の行列表現を作成する関数
    def make_jminus(self) -> None:
        j, n = self.rareearth
        m = j
        for i in range(1, n):
            self.j_minus[i][i - 1] = torch.sqrt(
                torch.tensor(((j + m) * (j - m + 1)), dtype=torch.float64)
            )
            m -= 1

    # レアアースのJxの行列表現を作成する関数
    def make_jx(self) -> None:
        self.j_x = (1 / 2.0) * (self.j_plus + self.j_minus)

    # レアアースのJyの行列表現を作成する関数
    def make_jy(self) -> None:
        self.j_y = -(1 / 2.0) * 1j * (self.j_plus - self.j_minus)

    # 全てのjの行列表現を作成
    def make_allj(self) -> None:
        self.make_j()
        self.make_jz()
        self.make_jplus()
        self.make_jminus()
        self.make_jx()
        self.make_jy()

    # レアアースのスティーブンス演算子の行列表現を作成する関数(O20~O60)O62以降は未実装
    def o20(self) -> None:
        identity = self.identity
        j = self.j
        jz2 = pow(self.j_z)
        self.o["O20"] = 3 * jz2 - torch.mm(j, j + identity)

    def o22(self) -> None:
        j_plus2 = pow(self.j_plus)
        j_minus2 = pow(self.j_minus)
        self.o["O22"] = 1 / 2.0 * (j_plus2 + j_minus2)

    def o40(self) -> None:
        j = self.j
        j2 = pow(j)
        j2_plus1 = pow(j + self.identity)
        jz2 = pow(self.j_z)
        jz4 = pow(self.j_z, n=4)
        # jj=j*(j+1)
        jj = torch.mm(j, j + self.identity)
        res = (
            35 * jz4
            - 30 * torch.mm(jj, jz2)
            + 25 * jz2
            - 6 * torch.mm(j, j + self.identity)
            + 3 * torch.mm(j2, j2_plus1)
        )
        self.o["O40"] = res

    def o42(self) -> None:
        identity = self.identity
        j = self.j
        jz2 = pow(self.j_z)
        j_plus2 = pow(self.j_plus)
        j_minus2 = pow(self.j_minus)
        # 7*jz^2-j(j+1)-5
        temp = 7 * jz2 - torch.mm(j, j + identity) - 5
        res = (1 / 4.0) * (
            torch.mm(temp, j_plus2 + j_minus2) + torch.mm(j_plus2 + j_minus2, temp)
        )
        self.o["O42"] = res

    def o43(self) -> None:
        jz = self.j_z
        j_plus3 = torch.mm(self.j_plus, pow(self.j_plus))
        j_minus3 = torch.mm(self.j_minus, pow(self.j_minus))
        res = (1 / 4.0) * (
            torch.mm(jz, j_plus3 + j_minus3) + torch.mm(j_plus3 + j_minus3, jz)
        )
        self.o["O43"] = res

    def o44(self) -> None:
        j_plus4 = pow(self.j_plus, n=4)
        j_minus4 = pow(self.j_minus, n=4)
        self.o["O44"] = (1 / 2.0) * (j_plus4 + j_minus4)

    def o60(self) -> None:
        j = self.j
        j_plus1 = j + self.identity
        jz = self.j_z
        res = (
            231 * pow(jz, n=6)
            - torch.mm(315 * j, torch.mm((j + 1), pow(jz, n=4)))
            + 735 * pow(jz, n=4)
            + torch.mm(105 * pow(j), torch.mm(pow(j_plus1), pow(jz)))
            - torch.mm(525 * j, torch.mm(j_plus1, pow(jz)))
            + 294 * pow(jz)
            - torch.mm(5 * pow(j, n=3), pow(j_plus1, n=3))
            + torch.mm(40 * pow(j), pow(j_plus1))
            - torch.mm(60 * j, j_plus1)
        )
        self.o["O60"] = res

    # 全てのスティーブンス演算子を作成
    def stevens_factors(self):
        self.make_allj()
        self.o20()
        self.o22()
        self.o40()
        self.o42()
        self.o43()
        self.o44()
        self.o60()

    # ハミルトニアンの計算
    def hamiltonian(self, func_h: str, b: int = 1) -> None:
        """
        func_h:ハミルトニアンの式
        使える演算子:四則演算
        変数:onm,b

        """
        o20 = self.o["O20"]
        o22 = self.o["O22"]
        o40 = self.o["O40"]
        o42 = self.o["O42"]
        o43 = self.o["O43"]
        o44 = self.o["O44"]
        o60 = self.o["O60"]
        o62 = self.o["O62"]
        o63 = self.o["O63"]
        o64 = self.o["O64"]
        o66 = self.o["O66"]
        try:
            result = eval(func_h)
        except (NameError, SyntaxError) as e:
            print(e)
            result = 0

        self.h = result

    # o=(stevens factor name,Tensor)
    def print_o(o, Key=None):
        n, m = o.size()
        v = Key
        t = o[Key]
        print(v)
        for i in range(n):
            print("|", end=" ")
            for j in range(m):
                print(t[i][j].item(), end=" ")
            print("|")
        print()

    # 温度と外部磁場の設定/mag_field=(float,vactor:torch.Tensor)
    def set_env(self, temp: int, mag_field: tuple) -> None:
        self.temp = temp
        self.mag_field = mag_field

    # ハミルトニアンにゼーマン項を追加
    def add_zeeman_term(self) -> None:
        strength, direction = self.mag_field
        direction = matrix.calc_norm(direction)
        ex_mag_field = strength * direction
        ex_mag_field.to(torch.float64)
        j_matrix = [self.j_x, self.j_y, self.j_z]
        _, n = self.rareearth
        zeeman = torch.zeros(n, dtype=torch.cfloat)
        for j, h in zip(j_matrix, ex_mag_field):
            zeeman = zeeman + (j * h)

        # Hamiltonian:[K]
        zeeman_term = (self.gfactor * self.mu_b * zeeman) / self.k_b
        print("Zeeman Term:{}".format(zeeman_term))
        self.h = self.h - ((self.gfactor * self.mu_b * zeeman) / self.k_b)

    # g因子の計算
    def calc_gfactor(self) -> None:
        j = self.rareearth[0]
        l = self.ls_number[0]
        s = self.ls_number[1]
        g = 1 + (j * (j + 1) + s * (s + 1) - l * (l + 1)) / (2 * j * (j + 1))

        self.gfactor = g

    # ハミルトニアンの対角化/b=(torch.Tensor):結晶場パラメータB40等
    def calc_energy(self, b: torch.Tensor = torch.tensor([1], dtype=torch.float64)):
        self.diag_hamiltonian = matrix.EigenCalc(tensor=self.h)
        self.diag_hamiltonian.eig()
        self.diag_hamiltonian.diagonal()
        self.diag_hamiltonian.shift_eig_value()
        self.diag_hamiltonian.multiply_b(b)
        self.energy = self.diag_hamiltonian.eig_value

    # 対角化された基底においてのJx,Jy,Jzの表現行列
    def calc_newbase_j(self) -> None:
        self.diag_hamiltonian.tensor = self.j_x
        self.diag_hamiltonian.convert_base()
        self.j_x = self.diag_hamiltonian.tensor
        self.diag_hamiltonian.tensor = self.j_y
        self.diag_hamiltonian.convert_base()
        self.j_y = self.diag_hamiltonian.tensor
        self.diag_hamiltonian.tensor = self.j_z
        self.diag_hamiltonian.convert_base()
        self.j_z = self.diag_hamiltonian.tensor

    # 分配関数の計算
    def calc_z(self) -> None:
        for e in self.energy:
            self.z += torch.exp(-e / self.temp)

    # 磁化の計算(mu_bあたり)
    def calc_mu(self) -> None:
        mu_x = 0
        mu_y = 0
        mu_z = 0

        for i, e in enumerate(self.diag_hamiltonian.eig_value):
            mu_x = (
                mu_x
                + self.gfactor * (torch.exp(-e / self.temp) / self.z) * self.j_x[i][i]
            )
            print(mu_x)
            mu_y = (
                mu_y
                + self.gfactor * (torch.exp(-e / self.temp) / self.z) * self.j_y[i][i]
            )
            print(mu_y)
            mu_z = (
                mu_z
                + self.gfactor * (torch.exp(-e / self.temp) / self.z) * self.j_z[i][i]
            )
            print(mu_z)

        self.mu += [mu_x, mu_y, mu_z]
        self.m = sum(self.mu)


def main():
    ce_cef = CrystalField("Ce")
    ce_cef.make_j()
    ce_cef.make_jz()
    ce_cef.make_jminus()
    ce_cef.make_jplus()
    ce_cef.o40()
    ce_cef.o44()
    print(ce_cef.j_plus)
    print(ce_cef.j_minus)
    print(ce_cef.o["O40"])
    print(ce_cef.o["O44"])
    hamiltonian_func = "o40+5*o44"
    ce_cef.hamiltonian(hamiltonian_func)
    print(ce_cef.h)


if __name__ == "__main__":
    main()
