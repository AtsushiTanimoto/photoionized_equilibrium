"""
計算結果の可視化
  plot_ion_fractions: 元素ごとの電離状態の割合（最内セルと最外セル）と温度・電子密度の半径分布
  単独で実行すると data/*.tsv を読んで再描画する
"""
import matplotlib.pyplot
import numpy
import os
import seaborn

import constants

# figureディレクトリのパス（プロジェクトルート直下）
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI",
         "XVII", "XVIII", "XIX", "XX", "XXI", "XXII", "XXIII", "XXIV", "XXV", "XXVI", "XXVII", "XXVIII", "XXIX", "XXX", "XXXI"]


def _setup():
    seaborn.set_theme()
    seaborn.set_context("poster")
    seaborn.set_style("ticks")
    os.makedirs(FIGURE_DIR, exist_ok=True)


def _fractions_by_element(table):
    """
    イオン記号 <元素><電子数 2 桁> をキーとする列の辞書から、元素ごとの (N_r, Z+1) 配列（列 j = 電荷 j）を作る
    """
    result = {}
    for key, values in table.items():
        symbol = key.rstrip("0123456789")
        if symbol in constants.ATOMIC_NUMBER and key[len(symbol):].isdigit():
            Z = constants.ATOMIC_NUMBER[symbol]
            charge = Z - int(key[len(symbol):])
            if symbol not in result:
                result[symbol] = numpy.zeros((len(values), Z + 1))
            result[symbol][:, charge] = values
    return result


def plot_ion_fractions(model=None, prefix="simulation", table=None, radius=None, temperature=None, n_e=None, xstar=None, cell=-1):
    """
    model（PhotoionizationModel）または TSV から作った辞書 table から図を描く。
      <prefix>_<元素>_fraction.png: 元素ごとに 1 枚。横軸は電離段階（分光学的記法）、縦軸はイオン割合（対数）。
                                    本コード（セル cell）と、xstar が与えられていれば XSTAR の値を重ねる。
      <prefix>_temperature.png:     温度と電子密度の半径分布
    xstar: {元素: 長さ Z+1 の配列} (compare_xstar.read_xstar_fractions の戻り値)
    """
    _setup()
    if model is not None:
        radius = model.r_cm
        temperature = model.temperature_array
        n_e = model.electron_density_array
        fractions = {element.element: model.fraction_array[e] for e, element in enumerate(model.elements)}
    else:
        fractions = _fractions_by_element(table)

    seaborn.set_palette("hls", 2)
    for symbol in sorted(fractions.keys(), key=lambda s: constants.ATOMIC_NUMBER[s]):
        fraction = fractions[symbol][cell]
        Z = len(fraction) - 1
        stages = numpy.arange(Z + 1)
        labels = [f"{symbol} {ROMAN[j]}" for j in stages]
        fig, ax = matplotlib.pyplot.subplots(figsize=(max(8, 0.6 * (Z + 1) + 4), 7))
        width = 0.38
        ax.bar(stages - 0.5 * width, numpy.maximum(fraction, 1e-12), width=width, label="this code")
        if xstar is not None and symbol in xstar:
            ax.bar(stages + 0.5 * width, numpy.maximum(xstar[symbol], 1e-12), width=width, label="XSTAR")
        ax.set_yscale("log")
        ax.set_ylim(1e-8, 2.0)
        ax.set_xticks(stages)
        ax.set_xticklabels(labels, rotation=90, fontsize="x-small")
        ax.set_ylabel("ion fraction")
        ax.set_title(symbol)
        ax.legend(loc="upper left", fontsize="small")
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.savefig(os.path.join(FIGURE_DIR, f"{prefix}_{symbol.lower()}_fraction.png"), dpi=120)
        matplotlib.pyplot.close(fig)

    fig, ax1 = matplotlib.pyplot.subplots(figsize=(16, 9))
    ax1.plot(radius, temperature, color="tab:orange", linewidth=2, marker="o", label="temperature")
    ax1.set_xlabel("radius [cm]")
    ax1.set_ylabel("temperature [K]")
    ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.plot(radius, n_e, color="tab:blue", linewidth=2, marker="s", label="$n_e$")
    ax2.set_ylabel("$n_e$ [cm$^{-3}$]")
    ax2.set_yscale("log")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(os.path.join(FIGURE_DIR, f"{prefix}_temperature.png"), dpi=120)
    matplotlib.pyplot.close(fig)


def plot_radial_profiles(model=None, prefix="simulation", table=None, radius=None, temperature=None, elements=None, R_s_cm=None):
    """
    半径方向の電離状態の分布（HII 領域モデル向け）。元素ごとに 1 枚の図を描く。
    """
    _setup()
    if model is not None:
        radius = model.r_cm
        temperature = model.temperature_array
        fractions = {element.element: model.fraction_array[e] for e, element in enumerate(model.elements)}
    else:
        fractions = _fractions_by_element(table)
    r_pc = radius / 3.0856775814913673e18
    symbols = elements if elements is not None else sorted(fractions.keys(), key=lambda s: constants.ATOMIC_NUMBER[s])
    for symbol in symbols:
        fraction = fractions[symbol]
        Z = fraction.shape[1] - 1
        seaborn.set_palette("hls", Z + 1)
        fig, ax = matplotlib.pyplot.subplots(figsize=(16, 9))
        for j in range(Z + 1):
            if numpy.max(fraction[:, j]) > 1e-6:
                ax.plot(r_pc, fraction[:, j], linewidth=2, label=f"{symbol} {ROMAN[j]}")
        if R_s_cm is not None:
            ax.axvline(R_s_cm / 3.0856775814913673e18, linestyle="--", color="gray", label="$R_s$")
        ax.set_xlabel("Distance from Star [pc]")
        ax.set_ylabel("Ionization Fraction")
        ax.set_ylim([-0.05, 1.40])
        ax.legend(loc="upper left", ncol=2, fontsize="small")
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.savefig(os.path.join(FIGURE_DIR, f"{prefix}_{symbol.lower()}_fraction.png"), dpi=150)
        matplotlib.pyplot.close(fig)

    fig, ax = matplotlib.pyplot.subplots(figsize=(16, 9))
    ax.plot(r_pc, temperature, color="orange", linewidth=2, label="Gas Temperature")
    ax.set_xlabel("Distance from Star [pc]")
    ax.set_ylabel("Temperature [K]")
    ax.legend(loc="upper right")
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(os.path.join(FIGURE_DIR, f"{prefix}_temperature.png"), dpi=150)
    matplotlib.pyplot.close(fig)


def read_tsv(filepath):
    """save_tsv の出力を {列名: 配列} の辞書として読む"""
    with open(filepath) as f:
        header = f.readline().split()
    data = numpy.loadtxt(filepath, skiprows=1, ndmin=2)
    return {name: data[:, k] for k, name in enumerate(header)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=os.path.join(DATA_DIR, "benchmark.tsv"))
    parser.add_argument("--prefix", default="benchmark")
    parser.add_argument("--radial", action="store_true", help="半径分布の図を描く（HII 領域モデル）")
    args = parser.parse_args()
    if not os.path.exists(args.file):
        print(f"Error: data file not found. Run python3 run_benchmark.py first.\nPath: {args.file}")
    else:
        table = read_tsv(args.file)
        if args.radial:
            plot_radial_profiles(table=table, prefix=args.prefix, radius=table["Radius_cm"], temperature=table["Temperature"])
        else:
            plot_ion_fractions(table=table, prefix=args.prefix, radius=table["Radius_cm"], temperature=table["Temperature"], n_e=table["n_e"])
        print("Finished redrawing figures.")
