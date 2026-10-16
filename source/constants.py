"""
原子データベースのパス、元素表・存在比、制動放射の係数（物理定数は各モジュールで astropy.constants から取る）
"""

# --- 制動放射 ---
Lambda_ff_0 = 1.426e-27  # 制動放射冷却係数 [erg cm^3 s^-1 K^-0.5] = (32π e⁶ / 3 h m_e c³) √(2π k / 3 m_e)。Λ = Λ_ff_0 √T Σ Z² ⟨g_ff⟩ n_e n_ion

import os

# --- 原子データベース ---
# 光電離断面積（電離限界振動数を含む）は FAC 原子データベースから読み込む (atomic_data.py)。電子数 ≤ 10 は ~/github/atomicdatabase の
# 全パイプライン、電子数 > 10 は同リポジトリの ground_photoionization.py (基底準位のみ) が同じディレクトリに置く
ATOMIC_DATA_DIR = "/Users/tanimoto/software/database/atomic_data"
# 再結合係数（放射・二電子）、衝突電離断面積、電離ポテンシャル、輝線冷却のデータは CHIANTI から読み込む
CHIANTI_DIR = "/Users/tanimoto/software/database/chianti"
# 中性原子の光吸収断面積 (pe-cs-<Z>.dat: 列は エネルギー [MeV]、断面積 [barn]。Geant4 の EPDL97 と同じ形式)。
# FAC の .pi がない中性原子 (Mg I など) に使う。
# 既定は ~/github/neutral_matter が EPICS2025 (LLNL の最新の EPDL) から作った表 (閾値が実験値で、EPDL97 と同じ値)。
# なければ MONACO の中性物質と同じ ~/software/database/neutral_matter/absorption の表を使う (geant4_epdl97 は閾値から、
# xraylib_rtab は 10 eV からしかないので閾値 < 10 eV の原子には使えない)
NEUTRAL_ABSORPTION_DIRS = ["/Users/tanimoto/github/neutral_matter/database/absorption/epics2025",
                           "/Users/tanimoto/software/database/neutral_matter/absorption/geant4_epdl97"]
# H 原子との電荷交換の係数 (Kingdon & Ferland 1996 のフィット、Cloudy のデータファイルの写し) はこのリポジトリの database/charge_exchange に同梱
CHARGE_EXCHANGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "charge_exchange")
# 周波数積分した制動放射の Gaunt 因子 (van Hoof et al. 2015、Z = 1〜30) はこのリポジトリの database/gaunt_ff に同梱
GAUNT_FF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "gaunt_ff")

# --- 元素 ---
ELEMENT_SYMBOLS = [
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn"]
ATOMIC_NUMBER = {symbol: z + 1 for z, symbol in enumerate(ELEMENT_SYMBOLS)}
# 標準原子量 [u] (輝線の熱的なドップラー幅に使う)
ATOMIC_MASS = dict(zip(ELEMENT_SYMBOLS, [1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180,
                                        22.990, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.948, 39.098, 40.078,
                                        44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933, 58.693, 63.546, 65.38]))

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

# HII 領域モデル (run_simulation.py) の元素。存在比は ABUNDANCES_ANGR をそのまま使う
HII_REGION_ELEMENTS = ["H", "He", "C", "N", "O", "Ne", "Mg", "Si", "S", "Ar"]   # 中性原子 (Mg I, Si I, S I, Ar I) は EPICS2025 の表
