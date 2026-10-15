"""
XSTAR を黒体光源・一様密度の設定で実行し、出力を xstar/<名前>/ に置く。
  python3 run_xstar.py --log-xi 2.5                      # ベンチマーク (kT = 2.5 keV, L = 10^44 erg/s, n = 10^9, N_H = 10^21, angr) → xstar/logxi=2.500e+00/
  python3 run_xstar.py --name hii_region --trad 4e4 --luminosity 3.6e38 --density 1e4 --column 3.1e22 --log-xi -1.9 --abundances H=1,He=1.02,C=0.79,N=0.6
xstar (HEASoft) が PATH にある必要がある。ベンチマークで 1 回約 3 分、低温 (10^4 K) の HII 領域では数十分以上。
--abundances は太陽組成 (angr) に対する比で、書かなかった元素は 0 (ベンチマークの既定は全元素 1)。
--bbody-table を付けると、XSTAR 内蔵の spectrum=bbody の代わりに Planck 関数の表 (bbody.dat) を spectrum=file で渡す。
内蔵の bbody は xstarlib/src/starf.f90 で E/kT > 60 の指数関数を頭打ちにし (expo の上限)、E/kT > 150 では 1/E の尾を足すため、
高エネルギー側に物理的でない床 (10^-9〜10^-19 × 10^38 erg/s/erg) が残り、低温黒体 (HII 領域) では電離境界の外側を
部分電離させてしまう。表で渡すと ispecg の下限 1e-49 まで下がる。表は E/kT > BBODY_TABLE_CUTOFF で 0 にする。
"""
import argparse
import os
import subprocess
import time

import numpy


ELEMENTS = ["h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg", "al", "si", "p", "s", "cl", "ar",
            "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn"]


BBODY_TABLE_CUTOFF = 60.0     # E/kT がこれを超える点は 0 にする (exp(-60) ≈ 1e-26)
BBODY_TABLE_FILE = "bbody.dat"


def write_bbody_table(path, T_BB, E_min=0.1, E_max=1.0e6, n_points=5000, E_zero_below=None, E_zero_range=None):
    """
    黒体 Planck 関数 (形だけ。規格化は XSTAR が rlrad38 で行う) を XSTAR の spectrum=file 用の表に書く。
    書式: 1 行目に点数、以降「エネルギー [eV]  値 (spectun=0: エネルギー単位)」。
    E_zero_below を与えると、それより低いエネルギーの値を 0 にする（診断用。例: 13.6 で Lyman 連続光だけを残す）。
    E_zero_range = (lo, hi) を与えると、その範囲の値を 0 にする（診断用。例: (10.0, 10.4) で Lyα 付近を消す）。
    """
    kT = 8.617333262e-5 * T_BB
    E = numpy.geomspace(E_min, E_max, n_points)
    x = E / kT
    F = numpy.where(x < BBODY_TABLE_CUTOFF, E**3 / numpy.expm1(numpy.minimum(x, BBODY_TABLE_CUTOFF)), 0.0)
    if E_zero_below is not None:
        F = numpy.where(E < E_zero_below, 0.0, F)
    if E_zero_range is not None:
        F = numpy.where((E >= E_zero_range[0]) & (E <= E_zero_range[1]), 0.0, F)
    F /= numpy.max(F)
    with open(path, "w") as f:
        f.write(f"{len(E)}\n")
        for e, y in zip(E, F):
            f.write(f"{e:.6e} {y:.6e}\n")
    return path


def xstar_directory(log_xi):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "xstar", f"logxi={log_xi:.3e}")


def make_command(log_xi, density=1.0e9, column=1.0e21, trad=2.9011e7, luminosity=1.0e44, temperature=1.0e6, abundances=None,
                 spectrum_file=None, xeemin=1.0e-2, cfrac=0.0):
    """
    trad: 黒体温度 [K] (XSTAR には 10^7 K 単位で渡す)、luminosity: 1-1000 Ry の光度 [erg/s] (10^38 単位で渡す)、
    temperature: 温度の初期値 [K] (10^4 K 単位で渡す)、abundances: {元素: angr に対する比}（None なら全元素 1）、
    spectrum_file: None なら spectrum=bbody、指定すればそのファイルを spectrum=file (spectun=0) で渡す
    """
    if spectrum_file is None:
        spectrum = [("spectrum", "bbody")]
    else:
        spectrum = [("spectrum", "file"), ("spectrum_file", spectrum_file), ("spectun", "0")]
    parameters = [
        ("temperature", f"{temperature / 1.0e4:.1e}"), ("pressure", "3.0e-02"), ("density", f"{density:.1e}")]
    parameters += spectrum
    parameters += [("trad", f"{trad / 1.0e7:.4e}"), ("rlrad38", f"{luminosity / 1.0e38:.3e}"),
        ("column", f"{column:.3e}"), ("rlogxi", f"{log_xi:.3e}"), ("abundtbl", "angr"),
        ("modelname", "three_code_benchmark"), ("cfrac", f"{cfrac:.1e}"), ("lcpres", "0"), ("nsteps", "5"),
        ("niter", "99"), ("lwrite", "0"), ("lprint", "0"), ("emult", "5.0e-01"), ("taumax", "2.0e+00"),
        ("xeemin", f"{xeemin:.1e}"), ("critf", "1.0e-07"), ("vturbi", "1.4142e+02"), ("radexp", "0.0e+00"),
        ("ncn2", "99999"), ("loopcontrol", "0"), ("npass", "1")]
    if abundances is None:
        abundances = {element: 1.0 for element in ELEMENTS}
    parameters += [(f"{element}abund", f"{abundances.get(element, abundances.get(element.capitalize(), 0.0)):.4f}") for element in ELEMENTS]
    return "xstar " + " ".join(f"{key}={value}" for key, value in parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-xi", type=float, required=True)
    parser.add_argument("--name", default=None, help="出力ディレクトリ xstar/<name> (既定: logxi=<値>)")
    parser.add_argument("--trad", type=float, default=2.9011e7, help="黒体温度 [K]")
    parser.add_argument("--luminosity", type=float, default=1.0e44, help="1-1000 Ry の光度 [erg/s]")
    parser.add_argument("--density", type=float, default=1.0e9, help="n_H [cm^-3]")
    parser.add_argument("--column", type=float, default=1.0e21, help="N_H [cm^-2]")
    parser.add_argument("--temperature", type=float, default=1.0e6, help="温度の初期値 [K]")
    parser.add_argument("--abundances", default=None, help="angr に対する比。例: H=1,He=1.02,C=0.79,N=0.6（書かない元素は 0）")
    parser.add_argument("--bbody-table", action="store_true", help="内蔵 bbody の代わりに Planck 関数の表を spectrum=file で渡す（高エネルギーの床を避ける）")
    parser.add_argument("--bbody-zero-below", type=float, default=None, help="--bbody-table の表でこのエネルギー [eV] より下を 0 にする（診断用）")
    parser.add_argument("--bbody-zero-range", default=None, help="--bbody-table の表でこの範囲 [eV] を 0 にする。例: 10.0,10.4（診断用）")
    parser.add_argument("--xeemin", type=float, default=1.0e-2, help="XSTAR の電子割合の下限 xeemin")
    parser.add_argument("--cfrac", type=float, default=0.0, help="XSTAR の covering fraction cfrac（1 にすると ucalc の線の連続光ポンピング ans2 が 0 になる）")
    args = parser.parse_args()
    directory = xstar_directory(args.log_xi) if args.name is None else os.path.join(os.path.dirname(os.path.abspath(__file__)), "xstar", args.name)
    os.makedirs(directory, exist_ok=True)
    abundances = None
    if args.abundances:
        abundances = {item.split("=")[0]: float(item.split("=")[1]) for item in args.abundances.split(",")}
    spectrum_file = None
    if args.bbody_table:
        spectrum_file = BBODY_TABLE_FILE
        zero_range = tuple(float(v) for v in args.bbody_zero_range.split(",")) if args.bbody_zero_range else None
        write_bbody_table(os.path.join(directory, spectrum_file), args.trad, E_zero_below=args.bbody_zero_below, E_zero_range=zero_range)
    command = make_command(args.log_xi, density=args.density, column=args.column, trad=args.trad,
                           luminosity=args.luminosity, temperature=args.temperature, abundances=abundances,
                           spectrum_file=spectrum_file, xeemin=args.xeemin, cfrac=args.cfrac)
    with open(os.path.join(directory, "run_xstar.command"), "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n" + command.replace(" ", " \\\n  ", 1).replace("= ", "=") + "\n")
    print(command)
    time_start = time.time()
    with open(os.path.join(directory, "xout_step.log"), "w") as log:
        status = subprocess.run(command, shell=True, cwd=directory, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"xstar status: {status}  ({(time.time() - time_start) / 60.0:.2f} min)  -> {directory}")
