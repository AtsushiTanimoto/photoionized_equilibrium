"""
XSTAR をベンチマークと同じ設定（黒体 kT = 2.5 keV、L = 10^44 erg/s、n = 10^9 cm^-3、N_H = 10^21 cm^-2、太陽組成 angr）で
与えた log ξ について実行し、出力を xstar/logxi=<値>/ に置く。
  python3 run_xstar.py --log-xi 2.5
xstar (HEASoft) が PATH にある必要がある。1 回約 3 分。
"""
import argparse
import os
import subprocess
import time


ELEMENTS = ["h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg", "al", "si", "p", "s", "cl", "ar",
            "k", "ca", "sc", "ti", "v", "cr", "mn", "fe", "co", "ni", "cu", "zn"]


def xstar_directory(log_xi):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "xstar", f"logxi={log_xi:.3e}")


def make_command(log_xi, density=1.0e9, column=1.0e21, trad=2.9011, luminosity38=1.0e6):
    parameters = [
        ("temperature", "1.0e+02"), ("pressure", "3.0e-02"), ("density", f"{density:.1e}"),
        ("spectrum", "bbody"), ("trad", f"{trad:.4e}"), ("rlrad38", f"{luminosity38:.1e}"),
        ("column", f"{column:.1e}"), ("rlogxi", f"{log_xi:.1e}"), ("abundtbl", "angr"),
        ("modelname", "three_code_benchmark"), ("cfrac", "0.0e+00"), ("lcpres", "0"), ("nsteps", "5"),
        ("niter", "99"), ("lwrite", "0"), ("lprint", "0"), ("emult", "5.0e-01"), ("taumax", "2.0e+00"),
        ("xeemin", "1.0e-02"), ("critf", "1.0e-07"), ("vturbi", "1.4142e+02"), ("radexp", "0.0e+00"),
        ("ncn2", "99999"), ("loopcontrol", "0"), ("npass", "1")]
    parameters += [(f"{element}abund", "1.00") for element in ELEMENTS]
    return "xstar " + " ".join(f"{key}={value}" for key, value in parameters)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-xi", type=float, required=True)
    args = parser.parse_args()
    directory = xstar_directory(args.log_xi)
    os.makedirs(directory, exist_ok=True)
    command = make_command(args.log_xi)
    with open(os.path.join(directory, "run_xstar.command"), "w") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n" + command.replace(" ", " \\\n  ", 1).replace("= ", "=") + "\n")
    print(command)
    time_start = time.time()
    with open(os.path.join(directory, "xout_step.log"), "w") as log:
        status = subprocess.run(command, shell=True, cwd=directory, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"xstar status: {status}  ({(time.time() - time_start) / 60.0:.2f} min)  -> {directory}")
