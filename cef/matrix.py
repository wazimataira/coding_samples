import torch

# 対角化などの行列計算を行うモジュール


# 対角化を行うクラス
class EigenCalc:
    def __init__(
        self, tensor: torch.Tensor = torch.zeros(3, 3, dtype=torch.float64)
    ) -> None:
        """
        self.tensor :対角化する行列
        self.tensor_size :行列のサイズ
        self.identity :対角化する行列と同じサイズの単位行列
        self.zero :対角化する行列と同じサイズの零行列
        self.eig_vec :固有ベクトルの辞書
        self.eig_value :固有値のリスト
        self.orthogonal_matrix :固有ベクトルの直行行列
        self.diagonal_matrix :固有値の対角行列
        """

        self.tensor = tensor
        self.tensor_size = tensor.size()
        n, _ = self.tensor_size
        self.identity = torch.eye(n, dtype=torch.float64)
        self.zero = torch.zeros(n, n, dtype=torch.float64)
        self.eig_vec = {}
        self.eig_value = torch.zeros(n, dtype=torch.float64)
        self.orthogonal_matrix = torch.zeros(n, n, dtype=torch.float64)
        self.diagonal_matrix = torch.eye(n, dtype=torch.float64)

    # 固有値,固有ベクトルの直行行列を計算する
    def eig(self) -> None:
        l, p = torch.linalg.eig(self.tensor)
        print(l)
        print(p)
        self.eig_value = torch.real(l)
        if torch.imag(p).sum() == 0:
            ortho = torch.real(p)
        else:
            ortho = p.to(torch.complex128)
        sorted, idx = torch.sort(self.eig_value)
        self.eig_value = sorted
        self.orthogonal_matrix = ortho[:, idx[0]].unsqueeze(dim=1)
        print(self.orthogonal_matrix)
        print(self.orthogonal_matrix.size())
        for i, c in enumerate(idx):
            if i == 0:
                pass
            else:
                self.orthogonal_matrix = torch.cat(
                    (self.orthogonal_matrix, (ortho[:, c]).unsqueeze(dim=1)), dim=1
                )
        print(self.eig_value)

    # 固有ベクトルを得る関数:list(Tensor)
    def eigenvector(self) -> None:
        _, n = self.orthogonal_matrix.size()
        eig_vec = []
        for i in range(n):
            eig_vec.append(self.orthogonal_matrix[:, i])

        for value, vec in zip(self.eig_value, eig_vec):
            self.eig_vec[value] = (value, vec)

    # 固有値の対角行列を得る関数
    def diagonal(self) -> None:
        n, _ = self.tensor_size
        for i in range(n):
            self.diagonal_matrix[i][i] = self.eig_value[i]

    # 対角化した基底においての行列表現を得る関数
    def convert_base(self) -> None:
        self.tensor = convert_complex(self.tensor)
        self.orthogonal_matrix = convert_complex(self.orthogonal_matrix)
        p = self.orthogonal_matrix
        p_t = torch.conj(torch.t(p))
        print(p.dtype, p_t.dtype, self.tensor.dtype)
        self.tensor = torch.matmul(p_t, torch.matmul(self.tensor, p))
        print(self.tensor)

    # 固有ベクトルを表示する関数
    def print_eigen_vec(self) -> None:
        judge_str = lambda x: str(x) if type(x) is not str else x
        for name, vec in self.eig_vec:
            print_vec(vec, name=judge_str(name))

    # 固有値の基準をずらす関数
    def shift_eig_value(self) -> None:
        min_value = torch.min(self.eig_value)
        self.eig_value = self.eig_value - min_value
        self.diagonal()

    # 結晶場パラメータBをかける関数
    def multiply_b(self, b: torch.Tensor) -> None:
        self.eig_value = torch.mul(self.eig_value, b)


def print_vec(v, name: str = None) -> None:
    n = len(v)
    print(name + ":")
    for i in range(n):
        print("(", end=" ")
        print(v[i].item(), end=" ")
    print(")")
    print()


def print_matrix(t, name: str = None) -> None:
    n, m = t.size()
    print(name + ":")
    for i in range(n):
        print("|", end=" ")
        for j in range(m):
            print(t[i][j].item(), end=" ")
        print("|")
    print()


# テンソルの情報を表示する関数
def parameter(tensor: torch.Tensor) -> None:
    print_matrix(tensor, name="Input Tensor")
    n, m = tensor.size()
    print("Shape : {0} × {1}".format(n, m), end="\n\n")
    print("dtype : {0}".format(tensor.dtype), end="\n\n")
    print("dimention : {0}".format(tensor.dim()), end="\n\n")


# pytorchのテンソルのデータタイプが複素数であるかを判定する関数
def is_complex(t: torch.Tensor) -> bool:
    if t.dtype == torch.complex64 or t.dtype == torch.complex128:
        return True
    else:
        return False


# pytorchのテンソルのデータタイプを複素数に変換する関数
def convert_complex(t: torch.Tensor) -> torch.Tensor:
    if is_complex(t):
        return t
    else:
        return t.to(torch.complex128)


# ベクトルを1に正規化する関数
def calc_norm(vec: torch.Tensor) -> torch.Tensor:
    norm = 0
    for v in vec:
        norm += v * v

    return vec / torch.sqrt(norm)
