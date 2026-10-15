"""
HII 領域モデル (source/run_simulation.py の data/hii_region.tsv) と XSTAR (xstar/hii_region/xout_abund1.fits) の
半径分布を比較する。イオンごとに figures/hii_region_<イオン記号>_comparison.png (本コードと XSTAR を hls の 2 色で区別、対数軸) と
figures/hii_region_temperature_comparison.png を描く。
  python3 compare_hii_region.py [--xstar-dir xstar/hii_region] [--tsv ../data/hii_region.tsv]
XSTAR の ABUNDANCES は半径ごとの行（先頭 2 行は同じ半径）で、裸のイオンの列はない (1 - Σ とする)。
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source"))  # source/ のモジュールを使う
import astropy.constants
import astropy.io.fits
import matplotlib.pyplot
import numpy
import seaborn

import constants
import visualization

pc = astropy.constants.pc.cgs.value  # パーセク [cm]
ROMAN = visualization.ROMAN
FRACTION_FLOOR = visualization.FRACTION_FLOOR   # 図の縦軸の下限 (これ未満の割合はこの値で描く)


def read_xstar_profiles(fits_path, elements):
    """XSTAR の ABUNDANCES から半径 [cm]、温度 [K]、{元素: (N_row, Z+1) の割合} を読む"""
    data = astropy.io.fits.open(fits_path)["ABUNDANCES"].data
    radius = data["radius"].astype(float)
    temperature = data["temperature"].astype(float) * 1.0e4
    fractions = {}
    for symbol in elements:
        Z = constants.ATOMIC_NUMBER[symbol]
        fraction = numpy.zeros((len(radius), Z + 1))
        for charge in range(Z):
            name = f"{symbol.lower()}_{ROMAN[charge].lower()}"
            if name in data.columns.names:
                fraction[:, charge] = data[name].astype(float)
        fraction[:, Z] = numpy.maximum(1.0 - numpy.sum(fraction[:, :Z], axis=1), 0.0)
        fractions[symbol] = fraction
    return radius, temperature, fractions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xstar-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "xstar", "hii_region"))
    parser.add_argument("--tsv", default=os.path.join(visualization.DATA_DIR, "hii_region.tsv"))
    parser.add_argument("--prefix", default="hii_region")
    parser.add_argument("--table", default=None, help="この半径 [pc] (カンマ区切り) で全元素・全電離段階の割合を表にして表示する。例: 0.05,0.1,0.14")
    args = parser.parse_args()

    table = visualization.read_tsv(args.tsv)
    ours = visualization._fractions_by_element(table)
    elements = sorted(ours.keys(), key=lambda s: constants.ATOMIC_NUMBER[s])
    r_ours = table["Radius_cm"] / pc
    r_xstar, T_xstar, xstar = read_xstar_profiles(os.path.join(args.xstar_dir, "xout_abund1.fits"), elements)
    r_xstar = r_xstar / pc

    seaborn.set_theme()
    seaborn.set_context("poster")
    seaborn.set_style("ticks")
    os.makedirs(visualization.FIGURE_DIR, exist_ok=True)
    # イオンごとに 1 枚: figures/<prefix>_<イオン記号>_comparison.png (本コードと XSTAR を色で区別、縦軸は対数)
    colors = seaborn.color_palette("hls", 2)
    for symbol in elements:
        Z = constants.ATOMIC_NUMBER[symbol]
        for j in range(Z + 1):
            label = visualization.ion_label(symbol, j)
            fig, ax = matplotlib.pyplot.subplots(figsize=(16, 9))
            ax.plot(r_ours, numpy.maximum(ours[symbol][:, j], FRACTION_FLOOR), color=colors[0], linewidth=2, label=f"{label} (This code)")
            ax.plot(r_xstar, numpy.maximum(xstar[symbol][:, j], FRACTION_FLOOR), color=colors[1], linewidth=2, label=f"{label} (XSTAR)")
            ax.set_xlabel("Distance from Star [pc]")
            ax.set_ylabel("Ionization Fraction")
            ax.set_yscale("log")
            ax.set_ylim([FRACTION_FLOOR, 3.0])
            ax.set_xlim([0.0, max(r_ours[-1], r_xstar[-1])])
            ax.legend(loc="upper right")
            matplotlib.pyplot.tight_layout()
            matplotlib.pyplot.savefig(os.path.join(visualization.FIGURE_DIR, f"{args.prefix}_{label}_comparison.png"), dpi=150)
            matplotlib.pyplot.close(fig)

    fig, ax = matplotlib.pyplot.subplots(figsize=(16, 9))
    colors = seaborn.color_palette("hls", 2)
    ax.plot(r_ours, table["Temperature"], color=colors[0], linewidth=2, label="This code")
    ax.plot(r_xstar, T_xstar, color=colors[1], linewidth=2, label="XSTAR")
    ax.set_xlabel("Distance from Star [pc]")
    ax.set_ylabel("Temperature [K]")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(os.path.join(visualization.FIGURE_DIR, f"{args.prefix}_temperature_comparison.png"), dpi=150)
    matplotlib.pyplot.close(fig)

    # 要約: 両コードの電離境界 (x_HII = 0.5 になる半径) と HII 領域内の温度
    def edge(r, x):
        k = numpy.argmax(x < 0.5) if numpy.any(x < 0.5) else len(x) - 1
        return r[k]
    print(f"HII edge (x_HII = 0.5): This code {edge(r_ours, ours['H'][:, 1]):.3f} pc, XSTAR {edge(r_xstar, xstar['H'][:, 1]):.3f} pc")
    inside = ours["H"][:, 1] > 0.5
    inside_x = xstar["H"][:, 1] > 0.5
    print(f"mean T inside HII region: This code {numpy.mean(table['Temperature'][inside]):.0f} K, XSTAR {numpy.mean(T_xstar[inside_x]):.0f} K")
    print(f"figures: {os.path.join(visualization.FIGURE_DIR, args.prefix + '_<ion>_comparison.png')} (one file per ion)")

    # 全電離段階の割合の表 (最も近いセル / ゾーンの値。両コードとも 1e-12 未満の段階は省く)
    if args.table:
        for r_pc in [float(v) for v in args.table.split(",")]:
            i = numpy.argmin(numpy.abs(r_ours - r_pc))
            k = numpy.argmin(numpy.abs(r_xstar - r_pc))
            print(f"\n=== r = {r_pc:.3f} pc  (This code: cell {i}, r = {r_ours[i]:.4f} pc, T = {table['Temperature'][i]:.0f} K | "
                  f"XSTAR: zone {k}, r = {r_xstar[k]:.4f} pc, T = {T_xstar[k]:.0f} K)")
            print(f"  {'ion':6s} {'This code':>11s} {'XSTAR':>11s} {'ratio':>8s}")
            for symbol in elements:
                Z = constants.ATOMIC_NUMBER[symbol]
                for j in range(Z + 1):
                    a, b = ours[symbol][i, j], xstar[symbol][k, j]
                    if a < 1e-12 and b < 1e-12:
                        continue
                    ratio = f"{a / b:8.3f}" if b > 0.0 else "     inf"
                    print(f"  {visualization.ion_label(symbol, j):6s} {a:11.3e} {b:11.3e} {ratio}")
