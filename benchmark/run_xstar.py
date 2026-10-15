"""
XSTAR を黒体光源・一様密度の設定で実行し、出力を xstar/<名前>/ に置く。
  python3 run_xstar.py --log-xi 2.5                      # ベンチマーク (kT = 2.5 keV, L = 10^44 erg/s, n = 10^9, N_H = 10^21, angr) → xstar/logxi=2.500e+00/
  python3 run_xstar.py --name hii_region --trad 4e4 --luminosity 3.6e38 --density 1e4 --column 3.1e22 --log-xi -1.9 --abundances H=1,He=1.02,C=0.79,N=0.6
xstar (HEASoft) が PATH にある必要がある。ベンチマークで 1 回約 3 分、低温 (10^4 K) の HII 領域では数十分以上。
--abundances は太陽組成 (angr) に対する比で、書かなかった元素は 0 (ベンチマークの既定は全元素 1)。
"""
import argparse
import os
import subprocess
import time


ELEMENTS = ["h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg", "al", "si", "p", "s", "cl", "ar",
            "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn"]


def xstar_directory(log_xi):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "xstar", f"logxi={log_xi:.3e}")


def make_command(log_xi, density=1.0e9, column=1.0e21, trad=2.9011e7, luminosity=1.0e44, temperature=1.0e6, abundances=None):
    """
    trad: 黒体温度 [K] (XSTAR には 10^7 K 単位で渡す)、luminosity: 1-1000 Ry の光度 [erg/s] (10^38 単位で渡す)、
    temperature: 温度の初期値 [K] (10^4 K 単位で渡す)、abundances: {元素: angr に対する比}（None なら全元素 1）
    """
    parameters = [
        ("temperature", f"{temperature / 1.0e4:.1e}"), ("pressure", "3.0e-02"), ("density", f"{density:.1e}"),
        ("spectrum", "bbody"), ("trad", f"{trad / 1.0e7:.4e}"), ("rlrad38", f"{luminosity / 1.0e38:.3e}"),
        ("column", f"{column:.3e}"), ("rlogxi", f"{log_xi:.3e}"), ("abundtbl", "angr"),
        ("modelname", "three_code_benchmark"), ("cfrac", "0.0e+00"), ("lcpres", "0"), ("nsteps", "5"),
        ("niter", "99"), ("lwrite", "0"), ("lprint", "0"), ("emult", "5.0e-01"), ("taumax", "2.0e+00"),
        ("xeemin", "1.0e-02"), ("critf", "1.0e-07"), ("vturbi", "1.4142e+02"), ("radexp", "0.0e+00"),
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
    args = parser.parse_args()
    directory = xstar_directory(args.log_xi) if args.name is None else os.path.join(os.path.dirname(os.path.abspath(__file__)), "xstar", args.name)
    os.makedirs(directory, exist_ok=True)
    abundances = None
    if args.abundances:
        abundances = {item.split("=")[0]: float(item.split("=")[1]) for item in args.abundances.split(",")}
    command = make_command(args.log_xi, density=args.density, column=args.column, trad=args.trad,
                           luminosity=args.luminosity, temperature=args.temperature, abundances=abundances)
    with open(os.path.join(directory, "run_xstar.command"), "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n" + command.replace(" ", " \\\n  ", 1).replace("= ", "=") + "\n")
    print(command)
    time_start = time.time()
    with open(os.path.join(directory, "xout_step.log"), "w") as log:
        status = subprocess.run(command, shell=True, cwd=directory, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"xstar status: {status}  ({(time.time() - time_start) / 60.0:.2f} min)  -> {directory}")
