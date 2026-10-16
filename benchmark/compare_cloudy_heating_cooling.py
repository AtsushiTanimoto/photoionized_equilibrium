"""
Lexington ベンチマークの加熱・冷却の内訳を、本コード (run_lexington.py の output/<output>_heating_cooling.tsv) と
Cloudy C25 (run_cloudy.py の benchmark/cloudy/<model>/<model>.cole と .het) で比べる。
  uv run python compare_cloudy_heating_cooling.py --model HII40 [--output lexington_HII40_diffuse]

冷却は元素ごと (輝線 + 再結合 + 衝突電離。両コードとも同じ定義) と制動放射・Compton、加熱は元素ごとの光電離と Compton。
体積で積分した光度 [erg/s] と全体に対する割合を表にし、
figures/<output>_heating_cooling_vs_cloudy.png に (上) 温度、(中) 元素ごとの冷却率の半径分布、(下) 積分した内訳の棒グラフを描く。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source"))  # source/ のモジュールを使う
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
import numpy
import seaborn

import lexington_reference
import run_cloudy
import visualization


def shell_volumes(radius_centers, R_in):
    """ゾーンの中心半径から殻の体積 (内縁は R_in、外縁は最後の間隔を外挿)"""
    r = numpy.asarray(radius_centers, dtype=float)
    edges = numpy.concatenate([[R_in], 0.5 * (r[1:] + r[:-1]), [r[-1] + 0.5 * (r[-1] - r[-2])]])
    return 4.0 / 3.0 * numpy.pi * (edges[1:]**3 - edges[:-1]**3)


def this_code(output, model_name):
    """本コードの半径 [cm]、温度、{冷却の要素: 率}、{加熱の要素: 率} [erg s^-1 cm^-3]"""
    table = visualization.read_tsv(os.path.join(visualization.OUTPUT_DIR, f"{output}_heating_cooling.tsv"))
    elements = list(lexington_reference.MODELS[model_name]["abundances"])
    cooling = {symbol: table[f"cooling_{symbol}"] for symbol in elements}
    cooling["free-free"] = table["cooling_bremsstrahlung"]
    cooling["Compton"] = table["cooling_compton"]
    heating = {symbol: table[f"heating_{symbol}"] for symbol in elements}
    heating["Compton"] = table["heating_compton"]
    return table["Radius_cm"], table["Temperature"], cooling, heating


def cloudy(model_name):
    """Cloudy の半径 [cm]、温度、{冷却の要素: 率}、{加熱の要素: 率} [erg s^-1 cm^-3]"""
    reference = lexington_reference.MODELS[model_name]
    directory, name = run_cloudy.cloudy_directory(model_name), model_name.lower()
    path = os.path.join(directory, f"{name}.cole")
    with open(path) as f:
        header = f.readline().lstrip("#").rstrip("\n").split("\t")
    data = numpy.loadtxt(path, comments="#")
    column = {key.strip(): data[:, k] for k, key in enumerate(header)}
    radius = reference["R_in"] + column["depth(cm)"]
    cooling = {symbol: column[symbol] for symbol in reference["abundances"]}
    cooling["free-free"] = column["FF_H"] + column["FF_M"]
    cooling["Compton"] = column["Comp"]
    known = set(cooling) | {"depth(cm)", "Temp(K)", "Ctot(erg/cm3/s)", "FF_H", "FF_M", "Comp"}
    cooling["other"] = sum(v for k, v in column.items() if k not in known)
    # 加熱: 行ごとに "Htot と (ラベル, 割合) の並び"
    heating = {symbol: numpy.zeros(len(radius)) for symbol in reference["abundances"]}
    heating["Compton"] = numpy.zeros(len(radius))
    heating["other"] = numpy.zeros(len(radius))
    with open(os.path.join(directory, f"{name}.het")) as f:
        rows = [line.rstrip("\n").split("\t") for line in f if not line.startswith("#")]
    for k, row in enumerate(rows[:len(radius)]):
        H_total = float(row[2])
        items = [item.strip() for item in row[5:] if item.strip()]
        for label, fraction in zip(items[0::2], items[1::2]):
            element = label[:2].strip()
            key = element if element in heating else ("Compton" if label.startswith("Comp") else "other")
            heating[key][k] += H_total * float(fraction)
    return radius, column["Temp(K)"], cooling, heating


def integrate(radius, rates, R_in):
    volume = shell_volumes(radius, R_in)
    return {key: float(numpy.sum(value * volume)) for key, value in rates.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HII40", choices=list(lexington_reference.MODELS))
    parser.add_argument("--output", default=None, help="本コードの出力名 (既定: lexington_<model>_diffuse)")
    args = parser.parse_args()
    output = args.output or f"lexington_{args.model}_diffuse"
    R_in = lexington_reference.MODELS[args.model]["R_in"]
    r_this, T_this, cool_this, heat_this = this_code(output, args.model)
    r_cl, T_cl, cool_cl, heat_cl = cloudy(args.model)
    L = {"cooling": (integrate(r_this, cool_this, R_in), integrate(r_cl, cool_cl, R_in)),
         "heating": (integrate(r_this, heat_this, R_in), integrate(r_cl, heat_cl, R_in))}

    for kind, (mine, theirs) in L.items():
        total_mine, total_theirs = sum(mine.values()), sum(theirs.values())
        print(f"\n{kind} (volume-integrated) [erg/s]   total: This code {total_mine:.4e}, Cloudy C25 {total_theirs:.4e}, ratio {total_mine / total_theirs:.3f}")
        print(f"  {'agent':10s} {'This code':>11s} {'frac':>7s} {'Cloudy C25':>11s} {'frac':>7s} {'ratio':>7s}")
        for key in sorted(set(mine) | set(theirs), key=lambda k: -max(mine.get(k, 0.0), theirs.get(k, 0.0))):
            a, b = mine.get(key, 0.0), theirs.get(key, 0.0)
            if max(a / total_mine, b / total_theirs) < 1e-4:
                continue
            ratio = f"{a / b:7.3f}" if b > 0 else "      -"
            print(f"  {key:10s} {a:11.4e} {a / total_mine:7.4f} {b:11.4e} {b / total_theirs:7.4f} {ratio}")

    # 図
    seaborn.set_theme()
    seaborn.set_context("talk")
    seaborn.set_style("ticks")
    mine_c, theirs_c = L["cooling"]
    agents = [k for k in sorted(set(mine_c) | set(theirs_c), key=lambda k: -max(mine_c.get(k, 0.0), theirs_c.get(k, 0.0)))
              if max(mine_c.get(k, 0.0) / sum(mine_c.values()), theirs_c.get(k, 0.0) / sum(theirs_c.values())) > 0.01]
    colors = dict(zip(agents, seaborn.color_palette("hls", len(agents))))
    unit = lexington_reference.MODELS[args.model]["R_out_unit"]
    fig, axes = matplotlib.pyplot.subplots(3, 1, figsize=(12, 15), gridspec_kw={"height_ratios": [1, 2, 1.4]})
    palette2 = seaborn.color_palette("hls", 2)
    axes[0].plot(r_this / unit, T_this, color=palette2[0], linewidth=2, label="This code")
    axes[0].plot(r_cl / unit, T_cl, color=palette2[1], linewidth=2, label="Cloudy C25")
    axes[0].set_ylabel("Temperature [K]")
    axes[0].legend(fontsize="small")
    for key in agents:
        if key in cool_this:
            axes[1].plot(r_this / unit, numpy.maximum(cool_this[key], 1e-40), color=colors[key], linewidth=2, label=f"{key} (This code)")
        if key in cool_cl:
            axes[1].plot(r_cl / unit, numpy.maximum(cool_cl[key], 1e-40), color=colors[key], linewidth=2, linestyle="--", label=f"{key} (Cloudy C25)")
    axes[1].set_yscale("log")
    top = max(numpy.max(sum(cool_this.values())), numpy.max(sum(cool_cl.values())))
    axes[1].set_ylim(top * 1e-4, top * 2.0)
    axes[1].set_ylabel("Cooling rate [erg s$^{-1}$ cm$^{-3}$]")
    axes[1].set_xlabel(f"Radius [{unit:.0e} cm]")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="x-small", ncol=1, borderaxespad=0.0)
    x = numpy.arange(len(agents))
    axes[2].bar(x - 0.2, [mine_c.get(k, 0.0) / sum(mine_c.values()) for k in agents], width=0.4, color=palette2[0], label="This code")
    axes[2].bar(x + 0.2, [theirs_c.get(k, 0.0) / sum(theirs_c.values()) for k in agents], width=0.4, color=palette2[1], label="Cloudy C25")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(agents, fontsize="small")
    axes[2].set_ylabel("Fraction of total cooling")
    axes[2].legend(fontsize="small")
    for ax in axes[:2]:
        ax.set_xlim(R_in / unit, max(r_this[-1], r_cl[-1]) / unit)
    axes[0].set_title(f"Lexington 2000 {args.model}: heating and cooling (This code vs Cloudy C25)")
    matplotlib.pyplot.tight_layout()
    path = os.path.join(visualization.FIGURE_DIR, f"{output}_heating_cooling_vs_cloudy.png")
    matplotlib.pyplot.savefig(path, dpi=130)
    print(f"\n{path}")
