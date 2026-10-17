"""
輝線冷却で考慮する準位数をイオンごとに決め、database/line_cooling/n_levels.tsv に書く (line_cooling.default_number_of_levels が読む)。
  uv run python make_line_cooling_levels.py [--elements Fe,Ni] [--tolerance 0.02] [--max-levels 300]

各イオンを CHIANTI の全準位で読み、低い方から N 準位に切り詰めた冷却率を全準位の値と比べ、
光電離プラズマでそのイオンが存在しうる温度 (3000 K 〜 max(IP/10k, 2×10⁴ K) の 6 点) と
n_e = 10², 10⁶, 10¹⁰ cm⁻³ のすべてで差が tolerance 以内になる最小の N (候補 CANDIDATES、上限 max_levels) を選ぶ。
"""
import argparse
import os
import time

import numpy

import atomic_data
import constants
import line_cooling

CANDIDATES = [20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300, 400, 500]
DENSITIES = [1.0e2, 1.0e6, 1.0e10]
k_B_eV = 8.617333262e-5   # [eV K^-1]


def truncated_cooling(ion, N, n_e):
    """ion (全準位で読んだ LineCoolingIon) を低い方から N 準位に切り詰めたときの 1 イオンあたりの冷却率 (温度グリッド上)"""
    A = ion.A[:N, :N]
    radiated_power = numpy.sum(A * ion.delta_E[:N, :N], axis=1)
    R = A[None, :, :] + n_e * ion.C_e[:, :N, :N].astype(float)
    R[:, 1:, 0] += 1e-30
    L = numpy.swapaxes(R, 1, 2).copy()
    diagonal = numpy.arange(N)
    L[:, diagonal, diagonal] -= numpy.sum(R, axis=2)
    L[:, 0, :] = 1.0
    b = numpy.zeros((R.shape[0], N, 1))
    b[:, 0, 0] = 1.0
    return numpy.maximum(numpy.linalg.solve(L, b)[:, :, 0], 0.0) @ radiated_power


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", default=",".join(constants.ELEMENT_SYMBOLS))
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--max-levels", type=int, default=300)
    args = parser.parse_args()
    start = time.time()
    rows = []
    for element in args.elements.split(","):
        Z = constants.ATOMIC_NUMBER[element]
        for charge in range(Z):
            name = f"{element.lower()}_{charge + 1}"
            if not line_cooling.line_cooling_data_available(name):
                continue
            try:
                ip = atomic_data.ionization_potential(Z, charge)
            except KeyError:
                continue
            T_max = max(ip / 10.0 / k_B_eV, 2.0e4)
            ion = line_cooling.LineCoolingIon(name, N_levels_max=100000, log_T_grid=numpy.log10(numpy.geomspace(3.0e3, T_max, 6)))
            N_all = ion.N_levels
            reference = [truncated_cooling(ion, N_all, n_e) for n_e in DENSITIES]
            cooling_20 = [truncated_cooling(ion, min(20, N_all), n_e) for n_e in DENSITIES]
            chosen, error = None, None
            for N in [c for c in CANDIDATES if c < N_all and c <= args.max_levels] + [min(N_all, args.max_levels)]:
                values = [truncated_cooling(ion, N, n_e) for n_e in DENSITIES]
                error = max(float(numpy.max(numpy.abs(v - r) / (r + 1e-300))) for v, r in zip(values, reference))
                if error <= args.tolerance or N == min(N_all, args.max_levels):
                    chosen = N
                    break
            ratio_20 = max(float(numpy.max((r + 1e-300) / (c + 1e-300))) for r, c in zip(reference, cooling_20))
            rows.append((name, N_all, chosen, error, ratio_20))
            print(f"{name:7s} N_all = {N_all:4d}  N = {chosen:4d}  error = {error:.3f}  all/N20 = {ratio_20:.3g}  ({(time.time() - start) / 60:.1f} min)", flush=True)
    directory = os.path.dirname(line_cooling.N_LEVELS_FILE)
    os.makedirs(directory, exist_ok=True)
    existing = {}
    if os.path.exists(line_cooling.N_LEVELS_FILE):
        with open(line_cooling.N_LEVELS_FILE) as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                values = line.rstrip("\n").split("\t")
                existing[values[0]] = values
    for name, N_all, chosen, error, ratio_20 in rows:
        existing[name] = [name, str(N_all), str(chosen), f"{error:.4f}", f"{ratio_20:.4g}"]
    order = {f"{e.lower()}_{j + 1}": (constants.ATOMIC_NUMBER[e], j) for e in constants.ELEMENT_SYMBOLS for j in range(constants.ATOMIC_NUMBER[e])}
    with open(line_cooling.N_LEVELS_FILE, "w") as f:
        f.write("ion\tN_levels_all\tN_levels\tmax_relative_error\tcooling_all_over_N20\n")
        for name in sorted(existing, key=lambda n: order.get(n, (99, 99))):
            f.write("\t".join(existing[name]) + "\n")
    print(f"{line_cooling.N_LEVELS_FILE}: {len(rows)} ions in {(time.time() - start) / 60:.1f} min")
