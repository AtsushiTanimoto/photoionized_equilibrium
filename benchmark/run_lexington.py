"""
Lexington 2000 の標準 HII 領域ベンチマーク (HII40; Ferland et al. 1995, Péquignot et al. 2001) を計算し、
各コードの公表値 (Ercolano et al. 2003, MNRAS 340, 1136 の Table 4) と比較する。
  uv run python run_lexington.py [--n-cells 250] [--iterations 10] [--case B]

条件 (同論文の Table 1): 黒体 T = 4×10⁴ K、光度 L = 3.082×10³⁹ erg/s (全波長)、内径 R_in = 3×10¹⁸ cm、
n_H = 100 cm⁻³ (一様)、球対称、存在比 He 0.10, C 2.2e-4, N 4.0e-5, O 3.3e-4, Ne 5.0e-5, S 9.0e-6。

出力する量 (ベンチマークの定義):
  Hβ 光度、Hβ に対する輝線比、T_inner (内縁の電子温度)、⟨T[Np Ne]⟩ = ∫ Ne Np Te dV / ∫ Ne Np dV、
  R_out (H⁺ の割合が 0.5 になる半径)、⟨He⁺⟩/⟨H⁺⟩ = [n(H)/n(He)] ∫ Ne N(He⁺) dV / ∫ Ne Np dV。
Hβ は Case B の放射率 (Osterbrock & Ferland 2006 の表 4.4、n_e = 10² cm⁻³) を log-log で内挿し、
金属の禁制線・半禁制線は輝線冷却の CHIANTI の多準位モデル (line_cooling.LineCoolingIon.line_emission) から求める。
He I 5876 などの再結合線と C II 1335 (再結合と蛍光が寄与) は扱わない。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source"))  # source/ のモジュールを使う
import argparse
import time

import astropy.constants
import astropy.units
import numpy

import equilibrium
import source


eV = astropy.units.eV.to(astropy.units.erg)
hc_eV_angstrom = (astropy.constants.h * astropy.constants.c).to(astropy.units.eV * astropy.units.angstrom).value   # 12398.4 eV Å

ELEMENTS = ["H", "He", "C", "N", "O", "Ne", "S"]
ABUNDANCES = {"H": 1.0, "He": 0.10, "C": 2.2e-4, "N": 4.0e-5, "O": 3.3e-4, "Ne": 5.0e-5, "S": 9.0e-6}

# Hβ の Case B 放射率 4π j / (n_e n_p) [erg cm³ s⁻¹] (Osterbrock & Ferland 2006、表 4.4)
HBETA_T = numpy.array([2500.0, 5000.0, 10000.0, 20000.0])
HBETA_EMISSIVITY = numpy.array([3.72e-25, 2.20e-25, 1.24e-25, 0.658e-25])

# 比較する輝線: (表の名前, イオン記号, 真空波長の範囲 [Å] のリスト)。範囲内の CHIANTI の全遷移を足す
LINES = [
    ("C II] 2325+",        "C05", [(2320.0, 2331.0)]),
    ("C III] 1907+1909",   "C04", [(1905.0, 1910.0)]),
    ("[N II] 122um",       "N06", [(1.20e6, 1.23e6)]),
    ("[N II] 6584+6548",   "N06", [(6546.0, 6552.0), (6582.0, 6588.0)]),
    ("[N II] 5755",        "N06", [(5753.0, 5758.0)]),
    ("[N III] 57.3um",     "N05", [(5.70e5, 5.76e5)]),
    ("[O I] 6300+6363",    "O08", [(6298.0, 6304.0), (6361.0, 6367.0)]),
    ("[O II] 7320+7330",   "O07", [(7316.0, 7336.0)]),
    ("[O II] 3726+3729",   "O07", [(3725.0, 3731.5)]),
    ("[O III] 51.8um",     "O06", [(5.15e5, 5.21e5)]),
    ("[O III] 88.3um",     "O06", [(8.80e5, 8.86e5)]),
    ("[O III] 5007+4959",  "O06", [(4958.0, 4963.0), (5005.0, 5011.0)]),
    ("[O III] 4363",       "O06", [(4362.0, 4366.0)]),
    ("[O IV] 25.9um",      "O05", [(2.585e5, 2.595e5)]),
    ("[Ne II] 12.8um",     "Ne09", [(1.278e5, 1.284e5)]),
    ("[Ne III] 15.5um",    "Ne08", [(1.553e5, 1.558e5)]),
    ("[Ne III] 3869+3968", "Ne08", [(3867.0, 3872.0), (3966.0, 3971.0)]),
    ("[S II] 6716+6731",   "S15", [(6714.0, 6737.0)]),
    ("[S II] 4068+4076",   "S15", [(4066.0, 4080.0)]),
    ("[S III] 18.7um",     "S14", [(1.867e5, 1.875e5)]),
    ("[S III] 33.6um",     "S14", [(3.33e5, 3.37e5)]),
    ("[S III] 9532+9069",  "S14", [(9067.0, 9074.0), (9529.0, 9536.0)]),
    ("[S IV] 10.5um",      "S13", [(1.049e5, 1.053e5)]),
]

# Ercolano et al. (2003) の Table 4 から: 中央値 (Median)、Cloudy (GF)、XSTAR (TK)、MOCASSIN 1-D (BE)
REFERENCE = {
    "Hbeta/1e37":          (2.05, 2.06, 2.10, 2.10),
    "C II] 2325+":         (0.144, 0.157, 0.110, 0.126),
    "C III] 1907+1909":    (0.070, 0.071, 0.091, 0.041),
    "[N II] 122um":        (0.034, 0.027, None, 0.034),
    "[N II] 6584+6548":    (0.730, 0.669, 0.69, 0.786),
    "[N II] 5755":         (0.0054, 0.0050, None, 0.0054),
    "[N III] 57.3um":      (0.292, 0.306, None, 0.229),
    "[O I] 6300+6363":     (0.0086, 0.0094, 0.012, 0.0080),
    "[O II] 7320+7330":    (0.029, 0.029, 0.023, 0.022),
    "[O II] 3726+3729":    (2.03, 1.94, 1.6, 1.75),
    "[O III] 51.8um":      (1.06, 1.23, 0.99, 1.09),
    "[O III] 88.3um":      (1.22, 1.12, 1.18, 1.26),
    "[O III] 5007+4959":   (2.18, 2.21, 3.27, 1.70),
    "[O III] 4363":        (0.0037, 0.0035, 0.0070, 0.0023),
    "[O IV] 25.9um":       (0.0010, 0.0010, 0.0013, 0.0010),
    "[Ne II] 12.8um":      (0.195, 0.177, None, 0.209),
    "[Ne III] 15.5um":     (0.322, 0.294, 0.35, 0.269),
    "[Ne III] 3869+3968":  (0.085, 0.084, 0.092, 0.055),
    "[S II] 6716+6731":    (0.147, 0.137, 0.315, 0.138),
    "[S II] 4068+4076":    (0.0080, 0.0093, 0.026, 0.0057),
    "[S III] 18.7um":      (0.577, 0.627, 0.535, 0.569),
    "[S III] 33.6um":      (0.937, 1.24, 0.86, 0.932),
    "[S III] 9532+9069":   (1.22, 1.13, 1.25, 1.19),
    "[S IV] 10.5um":       (0.359, 0.176, 0.56, 0.539),
    "T_inner/K":           (7653, 7719, 8318, 7480),
    "<T[NpNe]>/K":         (8026, 7940, 8199, 7722),
    "R_out/1e19cm":        (1.46, 1.46, 1.45, 1.49),
    "<He+>/<H+>":          (0.767, 0.787, 0.77, 0.686),
}


def line_luminosities(model):
    """全セルを体積で積分した Hβ と各輝線の光度 [erg/s]、および温度などの平均量"""
    volume = 4.0 / 3.0 * numpy.pi * (model.radius_edges[1:]**3 - model.radius_edges[:-1]**3)
    index = {symbol: k for k, symbol in enumerate(model.ion_symbols)}
    L_line = {name: 0.0 for name, _, _ in LINES}
    L_Hbeta = 0.0
    integral_NeNp = integral_NeNpT = integral_NeHeplus = 0.0
    for i in range(model.N_r):
        x = model._ion_fraction_vector(i)
        n_ion = model.ion_element_density * x
        n_e = model.electron_density_array[i]
        T = model.temperature_array[i]
        n_p, n_HI, n_Heplus = n_ion[index["H00"]], n_ion[index["H01"]], n_ion[index["He01"]]
        emissivity = numpy.exp(numpy.interp(numpy.log(T), numpy.log(HBETA_T), numpy.log(HBETA_EMISSIVITY)))
        L_Hbeta += emissivity * n_e * n_p * volume[i]
        integral_NeNp += n_e * n_p * volume[i]
        integral_NeNpT += n_e * n_p * T * volume[i]
        integral_NeHeplus += n_e * n_Heplus * volume[i]
        for name, symbol, ranges in LINES:
            k = index[symbol]
            ion_model = model.line_cooling.ions[k]
            if ion_model is None or n_ion[k] <= 0.0:
                continue
            _, _, energy_eV, photons = ion_model.line_emission(n_e, n_HI, T)
            wavelength = hc_eV_angstrom / energy_eV
            selected = numpy.zeros(len(wavelength), dtype=bool)
            for lo, hi in ranges:
                selected |= (wavelength >= lo) & (wavelength <= hi)
            L_line[name] += numpy.sum(photons[selected] * energy_eV[selected]) * eV * n_ion[k] * volume[i]
    x_Hplus = numpy.array([model._ion_fraction_vector(i)[index["H00"]] for i in range(model.N_r)])
    k_out = numpy.argmax(x_Hplus < 0.5) if numpy.any(x_Hplus < 0.5) else model.N_r - 1
    results = {"Hbeta/1e37": L_Hbeta / 1.0e37}
    for name, _, _ in LINES:
        results[name] = L_line[name] / L_Hbeta
    results["T_inner/K"] = model.temperature_array[0]
    results["<T[NpNe]>/K"] = integral_NeNpT / integral_NeNp
    results["R_out/1e19cm"] = model.radius_edges[k_out] / 1.0e19
    results["<He+>/<H+>"] = ABUNDANCES["H"] / ABUNDANCES["He"] * integral_NeHeplus / integral_NeNp
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cells", type=int, default=250)
    parser.add_argument("--r-max", type=float, default=1.8e19, help="計算する外縁 [cm]")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--n-packets", type=int, default=200000)
    parser.add_argument("--case", default="B", choices=["A", "B"])
    args = parser.parse_args()

    time_start = time.time()
    radius_edges = numpy.linspace(3.0e18, args.r_max, args.n_cells + 1)
    T_BB = 4.0e4
    kT_eV = astropy.constants.k_B.cgs.value * T_BB / eV
    light_source = source.BlackbodySource(T_BB=T_BB, luminosity=3.082e39, band_eV=(1.0e-4 * kT_eV, 40.0 * kT_eV))   # 全波長の光度
    model = equilibrium.PhotoionizationModel(
        radius_edges=radius_edges, hydrogen_number_density=100.0, source=light_source,
        elements=ELEMENTS, abundances=ABUNDANCES, T_gas=8.0e3, N_nu=1001, recombination_case=args.case)
    model.setup_grids()
    model.calculate_equilibrium(N_packets=args.n_packets, max_global_iterations=args.iterations)
    print(model.save_tsv("lexington_hii40.tsv"))
    results = line_luminosities(model)

    print(f"\nLexington HII40 (this code: Case {args.case}, {args.n_cells} cells, {model.global_iterations_done} global iterations)")
    print(f"{'quantity':22s} {'This code':>10s} {'Median':>9s} {'Cloudy':>9s} {'XSTAR':>9s} {'MOCASSIN':>9s} {'This/Median':>11s}")
    for name, value in results.items():
        median, cloudy, xstar, mocassin = REFERENCE[name]
        f = lambda v: f"{v:9.4g}" if v is not None else f"{'-':>9s}"
        print(f"{name:22s} {value:10.4g} {f(median)} {f(cloudy)} {f(xstar)} {f(mocassin)} {value / median:11.3f}")
    print(f"\nComputation time: {(time.time() - time_start) / 60.0:.2f} min")
