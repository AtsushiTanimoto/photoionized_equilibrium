"""
XSTAR の出力と本コードの結果 (data/benchmark.tsv, data/benchmark_heating_cooling.tsv) を比較する
  python3 compare_xstar.py <XSTAR の実行ディレクトリ> [--row -1] [--cell -1]
XSTAR の ABUNDANCES は先頭 2 行が同じ半径 (delta_r = 0) なので、既定では最後の行を使う。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source"))  # source/ のモジュールを使う
import argparse
import numpy

import astropy.io.fits

import visualization
import constants

ROMAN = visualization.ROMAN
ELEMENT_NAMES = {"H": "hydrogen", "He": "helium", "Li": "lithium", "Be": "beryllium", "B": "boron", "C": "carbon",
                 "N": "nitrogen", "O": "oxygen", "F": "fluorine", "Ne": "neon", "Na": "sodium", "Mg": "magnesium",
                 "Al": "aluminum", "Si": "silicon", "P": "phosphoru", "S": "sulfur", "Cl": "chlorine", "Ar": "argon",
                 "K": "potassium", "Ca": "calcium", "Sc": "scandium", "Ti": "titanium", "V": "vanadium",
                 "Cr": "chromium", "Mn": "manganese", "Fe": "iron", "Co": "cobalt", "Ni": "nickel", "Cu": "copper", "Zn": "zinc"}


def read_xstar_fractions(fits_path, row=-1):
    """
    xout_abund1.fits の ABUNDANCES から {元素: 長さ Z+1 の配列（列 j = 電荷 j）} と温度 [K]、n_e を読む。
    列名は <元素小文字>_<ローマ数字> で、裸のイオンの列はない（1 - Σ とする）。
    """
    data = astropy.io.fits.open(fits_path)["ABUNDANCES"].data
    fractions = {}
    for symbol in constants.ELEMENT_SYMBOLS:
        Z = constants.ATOMIC_NUMBER[symbol]
        fraction = numpy.zeros(Z + 1)
        for charge in range(Z):
            name = f"{symbol.lower()}_{ROMAN[charge].lower()}"
            if name in data.columns.names:
                fraction[charge] = data[name][row]
        fraction[Z] = max(1.0 - numpy.sum(fraction[:Z]), 0.0)
        fractions[symbol] = fraction
    return fractions, data["temperature"][row] * 1.0e4, data["x_e"][row] * data["n_p"][row]


def read_xstar_heating_cooling(fits_path, row=-1):
    heating = {c: float(astropy.io.fits.open(fits_path)["HEATING"].data[c][row]) for c in astropy.io.fits.open(fits_path)["HEATING"].columns.names[8:]}
    cooling = {c: float(astropy.io.fits.open(fits_path)["COOLING"].data[c][row]) for c in astropy.io.fits.open(fits_path)["COOLING"].columns.names[8:]}
    return heating, cooling


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-xi", type=float, default=3.5, help="比較する log ξ（xstar/logxi=<値>/ と data/benchmark_logxi=<値>*.tsv を使う）")
    parser.add_argument("--xstar-dir", default=None, help="XSTAR の実行ディレクトリ（既定: xstar/logxi=<値>）")
    parser.add_argument("--tsv", default=None)
    parser.add_argument("--heating", default=None)
    parser.add_argument("--row", type=int, default=-1, help="XSTAR の行")
    parser.add_argument("--cell", type=int, default=-1, help="本コードのセル")
    parser.add_argument("--prefix", default=None)
    args = parser.parse_args()
    name = f"benchmark_logxi={args.log_xi:.3e}"
    if args.xstar_dir is None:
        args.xstar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xstar", f"logxi={args.log_xi:.3e}")
    if args.tsv is None:
        args.tsv = os.path.join(visualization.DATA_DIR, f"{name}.tsv")
    if args.heating is None:
        args.heating = os.path.join(visualization.DATA_DIR, f"{name}_heating_cooling.tsv")
    if args.prefix is None:
        args.prefix = name

    fits_path = os.path.join(args.xstar_dir, "xout_abund1.fits")
    xstar, T_xstar, n_e_xstar = read_xstar_fractions(fits_path, args.row)
    table = visualization.read_tsv(args.tsv)
    ours = visualization._fractions_by_element(table)
    cell = args.cell
    print(f"temperature: XSTAR = {T_xstar:.4e} K, this code = {table['Temperature'][cell]:.4e} K")
    print(f"n_e:         XSTAR = {n_e_xstar:.4e},   this code = {table['n_e'][cell]:.4e} cm^-3")
    print()
    print("ion fractions (stages with fraction > 1e-4 in either code)")
    for symbol in sorted(ours.keys(), key=lambda s: constants.ATOMIC_NUMBER[s]):
        Z = constants.ATOMIC_NUMBER[symbol]
        lines = []
        for j in range(Z + 1):
            a, b = ours[symbol][cell, j], xstar[symbol][j]
            if max(a, b) > 1e-4:
                ratio = a / b if b > 0 else numpy.inf
                lines.append(f"    {symbol} {ROMAN[j]:7s}  this = {a:.3e}  XSTAR = {b:.3e}  ratio = {ratio:7.3f}")
        print(f"  {symbol}:")
        print("\n".join(lines))

    if os.path.exists(args.heating):
        heating_xstar, cooling_xstar = read_xstar_heating_cooling(fits_path, args.row)
        hc = visualization.read_tsv(args.heating)
        print()
        print("heating [erg s^-1 cm^-3]           this code      XSTAR")
        for symbol in sorted(ours.keys(), key=lambda s: constants.ATOMIC_NUMBER[s]):
            print(f"  {symbol:8s} {hc['heating_' + symbol][cell]:12.3e} {heating_xstar.get(ELEMENT_NAMES[symbol], numpy.nan):12.3e}")
        print(f"  {'compton':8s} {hc['heating_compton'][cell]:12.3e} {heating_xstar['compton']:12.3e}")
        print(f"  {'total':8s} {hc['heating_total'][cell]:12.3e} {heating_xstar['total']:12.3e}")
        print("cooling [erg s^-1 cm^-3]           this code      XSTAR")
        for symbol in sorted(ours.keys(), key=lambda s: constants.ATOMIC_NUMBER[s]):
            print(f"  {symbol:8s} {hc['cooling_' + symbol][cell]:12.3e} {cooling_xstar.get(ELEMENT_NAMES[symbol], numpy.nan):12.3e}")
        for key in ("line", "recombination_RR", "recombination_DR", "collisional_ionization"):
            print(f"  {key:24s} {hc['cooling_' + key][cell]:12.3e}")
        print(f"  {'compton':8s} {hc['cooling_compton'][cell]:12.3e} {cooling_xstar['compton']:12.3e}")
        print(f"  {'brems':8s} {hc['cooling_bremsstrahlung'][cell]:12.3e} {cooling_xstar['brems']:12.3e}")
        print(f"  {'total':8s} {hc['cooling_total'][cell]:12.3e} {cooling_xstar['total']:12.3e}")

    visualization.plot_ion_fractions(table=table, prefix=args.prefix, radius=table["Radius_cm"], temperature=table["Temperature"], n_e=table["n_e"], xstar=xstar, cell=cell)
    print(f"\nfigures: {os.path.join(visualization.FIGURE_DIR, args.prefix + '_<element>_fraction.png')}")
