import torch
import cef
import matrix
import plot

# Ceの立方晶結晶場の計算を行うメインモジュール


# 110方向に任意の大きさの外部磁場hをかけ、Ceの立方晶結晶場の計算を行う関数
def calc_mag(h: float) -> torch.Tensor:
    # rare_earth=input("Input Rare Earth Name : ")
    crystalfield = cef.CrystalField("Ce")
    crystalfield.calc_gfactor()
    crystalfield.make_allj()
    crystalfield.o40()
    crystalfield.o44()
    hamiltonian_func = "b*(o40+5*o44)"
    crystalfield.hamiltonian(hamiltonian_func, b=40)
    print(crystalfield.h)
    crystalfield.set_env(2, (h, torch.tensor([0, 0, 1])))
    crystalfield.add_zeeman_term()
    b = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float64)
    crystalfield.calc_energy(b=b)
    print(crystalfield.diag_hamiltonian.diagonal_matrix)
    print("eigvalue :{}".format(crystalfield.diag_hamiltonian.eig_value))
    crystalfield.calc_newbase_j()
    crystalfield.calc_z()
    crystalfield.calc_mu()
    print(crystalfield.mu)

    return crystalfield.m


# 110方向に様々な大きさの外部磁場をかけた時のCeの立方晶結晶場の計算を行い、プロットするメイン関数
def main():
    mag = []
    for i in range(0, 14):
        mag.append(calc_mag(i * 10000))

    field = [i for i in range(0, 14)]
    data = [field, mag]
    mag_plot = plot.Plot(data, "Magnetic Field", "Magnetic Moment", title="Ce 立方晶結晶場")
    mag_plot.select_data()
    mag_plot.plot_type(label="(0,0,1)")
    mag_plot.set_lim((0.0, 15.0), (0.0, 0.8))
    mag_plot.plot()


if __name__ == "__main__":
    main()
