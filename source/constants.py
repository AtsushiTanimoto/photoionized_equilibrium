"""
物理定数の定義 (cgs単位系) と原子データベースのパス、元素表・存在比
"""

import astropy.constants
import astropy.units

# --- 物理定数 (cgs単位系、astropy.constants の CODATA 値) ---
h = astropy.constants.h.cgs.value                      # プランク定数 [erg s]
c = astropy.constants.c.cgs.value                      # 光速度 [cm s^-1]
k_B = astropy.constants.k_B.cgs.value                  # ボルツマン定数 [erg K^-1]
eV = (1.0 * astropy.units.eV).to(astropy.units.erg).value  # 電子ボルト [erg]
m_e = astropy.constants.m_e.cgs.value                  # 電子質量 [g]
sigma_T = astropy.constants.sigma_T.cgs.value          # トムソン断面積 [cm^2]
m_e_c2_eV = (astropy.constants.m_e * astropy.constants.c**2).to(astropy.units.eV).value  # 電子の静止エネルギー [eV]
Lambda_ff_0 = 1.42e-27  # 制動放射冷却係数 [erg cm^3 s^-1 K^-0.5]
R_sun = astropy.constants.R_sun.cgs.value              # 太陽半径 [cm]
pc = astropy.constants.pc.cgs.value                    # パーセク [cm]
Ry = astropy.constants.Ryd.cgs.value * h * c           # リュードベリ [erg]
CM_INVERSE_TO_EV = (h * c / eV)                        # 1 cm^-1 のエネルギー [eV]
T_HI = 13.598434 * eV / k_B                            # 水素の電離ポテンシャル / k_B [K]

# --- 原子データベース ---
# 光電離断面積（電離限界振動数を含む）は FAC 原子データベースから読み込む (atomic_data.py)
ATOMIC_DATA_DIR = "/Users/tanimoto/software/database/atomic_data"
# 再結合係数（放射・二電子）、衝突電離断面積、電離ポテンシャル、輝線冷却のデータは CHIANTI から読み込む
CHIANTI_DIR = "/Users/tanimoto/software/database/chianti"

# --- 元素 ---
ELEMENT_SYMBOLS = [
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn"]
ATOMIC_NUMBER = {symbol: z + 1 for z, symbol in enumerate(ELEMENT_SYMBOLS)}

# 太陽組成 (Anders & Grevesse 1989、XSTAR の abundtbl=angr) の H に対する数密度比
ABUNDANCES_ANGR = {
    "H": 1.000e+00, "He": 9.770e-02, "Li": 1.450e-11, "Be": 1.410e-11, "B": 3.980e-10,
    "C": 3.630e-04, "N": 1.120e-04, "O": 8.510e-04, "F": 3.630e-08, "Ne": 1.230e-04,
    "Na": 2.140e-06, "Mg": 3.800e-05, "Al": 2.950e-06, "Si": 3.550e-05, "P": 2.820e-07,
    "S": 1.620e-05, "Cl": 3.160e-07, "Ar": 3.630e-06, "K": 1.320e-07, "Ca": 2.290e-06,
    "Sc": 1.260e-09, "Ti": 9.770e-08, "V": 1.000e-08, "Cr": 4.680e-07, "Mn": 2.450e-07,
    "Fe": 4.680e-05, "Co": 8.320e-08, "Ni": 1.780e-06, "Cu": 1.620e-08, "Zn": 3.980e-08}

# MONACO の PhotoionizedPlasma 物理リストが扱う元素 (XSTAR ベンチマークの比較対象)
BENCHMARK_ELEMENTS = ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "S", "Ar", "Ca", "Cr", "Mn", "Fe", "Ni"]

# --- 従来の HII 領域モデル (H + He + C) の存在比 ---
Y_He        = 0.1            # ヘリウム存在比 (n_He / n_H)
A_C         = 10**(8.46 - 12.0)  # 炭素存在比 (n_C / n_H): 太陽組成 Asplund et al. (2021)
A_N         = 10**(7.83 - 12.0)  # 窒素存在比 (n_N / n_H): 太陽組成 Asplund et al. (2021)
