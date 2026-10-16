"""
Lexington 2000 ベンチマーク (Ferland et al. 1995, Péquignot et al. 2001) を計算し、各コードの公表値と比較する。
  uv run python run_lexington.py --model HII40 [--n-cells 250] [--iterations 10] [--case B | --diffuse] [--plot-only]
結果は output/<output>_results.tsv、図は figures/<output>.png (各量の中央値に対する比) と
figures/<output>_vs_cloudy.png (run_cloudy.py の Cloudy C25 に対する比)。
モデル (入力は lexington_reference.MODELS、Ercolano et al. 2003 の Table 1):
  HII40: 黒体 4×10⁴ K の標準 HII 領域、HII20: 2×10⁴ K の低励起 HII 領域、
  PN150: 1.5×10⁵ K の光学的に厚い惑星状星雲、PN75: 7.5×10⁴ K の光学的に薄い (外径 7.5×10¹⁷ cm で打ち切る) 惑星状星雲。
いずれも一様密度の球殻で、内側は空洞。

出力する量 (ベンチマークの定義):
  Hβ 光度、Hβ に対する輝線比、T_inner (内縁の電子温度)、⟨T[Np Ne]⟩ = ∫ Ne Np Te dV / ∫ Ne Np dV、
  R_out (H⁺ の割合が 0.5 になる半径。PN75 は外径で固定なので出さない)、⟨He⁺⟩/⟨H⁺⟩ = [n(H)/n(He)] ∫ Ne N(He⁺) dV / ∫ Ne Np dV。
Hβ は Case B の放射率 (Osterbrock & Ferland 2006 の表 4.4、n_e = 10² cm⁻³) を log-log で内挿し、
金属の輝線は輝線冷却の CHIANTI の多準位モデル (line_cooling.LineCoolingIon.line_emission) から求める (光学的に薄いとして)。
He I・He II の再結合線と C II 1335 (再結合と蛍光が寄与) は扱わない。
比較する中央値は MOCASSIN 以外のコードの値から計算する (lexington_reference.median)。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source"))  # source/ のモジュールを使う
import argparse
import time

import astropy.constants
import astropy.units
import numpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
import matplotlib.ticker
import seaborn

import equilibrium
import lexington_reference
import source
import visualization


eV = astropy.units.eV.to(astropy.units.erg)
hc_eV_angstrom = (astropy.constants.h * astropy.constants.c).to(astropy.units.eV * astropy.units.angstrom).value   # 12398.4 eV Å

# Hβ の Case B 放射率 4π j / (n_e n_p) [erg cm³ s⁻¹] (Osterbrock & Ferland 2006、表 4.4)
HBETA_T = numpy.array([2500.0, 5000.0, 10000.0, 20000.0])
HBETA_EMISSIVITY = numpy.array([3.72e-25, 2.20e-25, 1.24e-25, 0.658e-25])

# 比較する輝線: 名前 → (イオン記号, 真空波長の範囲 [Å] のリスト)。範囲内の CHIANTI の全遷移を足す
LINES = {
    "C II] 2325+":        ("C05",  [(2320.0, 2331.0)]),
    "C III] 1907+1909":   ("C04",  [(1905.0, 1910.0)]),
    "C IV 1549+":         ("C03",  [(1547.0, 1552.0)]),
    "[N I] 5200+5198":    ("N07",  [(5196.0, 5204.0)]),
    "[N II] 122um":       ("N06",  [(1.20e6, 1.23e6)]),
    "[N II] 6584+6548":   ("N06",  [(6546.0, 6552.0), (6582.0, 6588.0)]),
    "[N II] 5755":        ("N06",  [(5753.0, 5758.0)]),
    "N III] 1749+":       ("N05",  [(1745.0, 1755.0)]),
    "[N III] 57.3um":     ("N05",  [(5.70e5, 5.76e5)]),
    "N IV] 1487+":        ("N04",  [(1482.0, 1488.0)]),
    "N V 1240+":          ("N03",  [(1237.0, 1244.0)]),
    "[O I] 63.1um":       ("O08",  [(6.30e5, 6.33e5)]),
    "[O I] 6300+6363":    ("O08",  [(6298.0, 6304.0), (6361.0, 6367.0)]),
    "[O II] 7320+7330":   ("O07",  [(7316.0, 7336.0)]),
    "[O II] 3726+3729":   ("O07",  [(3725.0, 3731.5)]),
    "[O III] 51.8um":     ("O06",  [(5.15e5, 5.21e5)]),
    "[O III] 88.3um":     ("O06",  [(8.80e5, 8.86e5)]),
    "[O III] 5007+4959":  ("O06",  [(4958.0, 4963.0), (5005.0, 5011.0)]),
    "[O III] 4363":       ("O06",  [(4362.0, 4366.0)]),
    "[O IV] 25.9um":      ("O05",  [(2.585e5, 2.595e5)]),
    "O IV] 1403+":        ("O05",  [(1396.0, 1409.0)]),
    "O V] 1218+":         ("O04",  [(1212.0, 1220.0)]),
    "O VI 1034+":         ("O03",  [(1030.0, 1040.0)]),
    "[Ne II] 12.8um":     ("Ne09", [(1.278e5, 1.284e5)]),
    "[Ne III] 15.5um":    ("Ne08", [(1.553e5, 1.558e5)]),
    "[Ne III] 3869+3968": ("Ne08", [(3867.0, 3872.0), (3966.0, 3971.0)]),
    "[Ne IV] 2423+":      ("Ne07", [(2420.0, 2428.0)]),
    "[Ne V] 3426+3346":   ("Ne06", [(3343.0, 3349.0), (3424.0, 3430.0)]),
    "[Ne V] 24.2um":      ("Ne06", [(2.42e5, 2.44e5)]),
    "[Ne VI] 7.63um":     ("Ne05", [(7.62e4, 7.67e4)]),
    "Mg II 2798+":        ("Mg11", [(2795.0, 2805.0)]),
    "[Mg IV] 4.49um":     ("Mg09", [(4.48e4, 4.50e4)]),
    "[Mg V] 5.61um":      ("Mg08", [(5.60e4, 5.62e4)]),
    "[Si II] 34.8um":     ("Si13", [(3.47e5, 3.49e5)]),
    "Si II] 2335+":       ("Si13", [(2328.0, 2352.0)]),
    "Si III] 1892+":      ("Si12", [(1882.0, 1894.0)]),
    "Si IV 1397+":        ("Si11", [(1393.0, 1404.0)]),
    "[S II] 6716+6731":   ("S15",  [(6714.0, 6737.0)]),
    "[S II] 4068+4076":   ("S15",  [(4066.0, 4080.0)]),
    "[S III] 18.7um":     ("S14",  [(1.867e5, 1.875e5)]),
    "[S III] 33.6um":     ("S14",  [(3.33e5, 3.37e5)]),
    "[S III] 9532+9069":  ("S14",  [(9067.0, 9074.0), (9529.0, 9536.0)]),
    "[S IV] 10.5um":      ("S13",  [(1.049e5, 1.053e5)]),
}


def save_results(results, model_name, output):
    """結果を output/<output>_results.tsv に保存する (列: quantity, this_code, median)"""
    path = os.path.join(visualization.OUTPUT_DIR, f"{output}_results.tsv")
    with open(path, "w") as f:
        f.write("quantity\tthis_code\tmedian\n")
        for name, value in results.items():
            median = lexington_reference.median(model_name, name)
            f.write(f"{name}\t{value:.6e}\t{median if median is not None else 'nan'}\n")
    return path


def read_cloudy(model_name):
    """run_cloudy.py の結果 (output/cloudy_<model>_results.tsv) を読む。なければ空の辞書"""
    path = os.path.join(visualization.OUTPUT_DIR, f"cloudy_{model_name}_results.tsv")
    values = {}
    if os.path.exists(path):
        with open(path) as f:
            next(f)
            for line in f:
                name, value, _ = line.rstrip("\n").split("\t")
                values[name] = float(value)
    return values


def plot_results(results, model_name, output, title=None):
    """
    各量の中央値に対する比 (対数軸) を図にして figures/<output>.png に保存する。
    灰色の点は MOCASSIN 以外の各コード、記号は Cloudy (GF)・XSTAR (TK) の 2001 年の値と、手元で同じ条件で計算した Cloudy C25
    (run_cloudy.py の結果があれば)、赤の丸が本コード。
    灰色の帯は MOCASSIN 以外のコードの最小〜最大の範囲。
    """
    reference = lexington_reference.MODELS[model_name]
    names = [name for name in results if lexington_reference.median(model_name, name)]
    y = numpy.arange(len(names))[::-1]
    seaborn.set_theme()
    seaborn.set_context("talk")
    seaborn.set_style("ticks")
    colors = seaborn.color_palette("hls", 4)
    fig, ax = matplotlib.pyplot.subplots(figsize=(11, 0.42 * len(names) + 2.0))
    for yy, name in zip(y, names):
        median = lexington_reference.median(model_name, name)
        others = [v / median for code, v in zip(reference["columns"], reference["rows"][name]) if v is not None and not code.startswith("BE")]
        ax.plot([min(others), max(others)], [yy, yy], color="0.85", linewidth=8, solid_capstyle="round", zorder=1)
        ax.scatter(others, [yy] * len(others), color="0.55", s=12, zorder=2)
    for code, marker, color, label in [("GF", "s", colors[2], "Cloudy (2001)"), ("TK", "^", colors[1], "XSTAR (2001)")]:
        xs, ys = [], []
        for yy, name in zip(y, names):
            v = lexington_reference.value(model_name, name, code)
            if v is not None:
                xs.append(v / lexington_reference.median(model_name, name))
                ys.append(yy)
        if xs:
            ax.scatter(xs, ys, marker=marker, facecolors="none", edgecolors=[color], s=50, linewidths=1.5, label=label, zorder=3)
    cloudy = read_cloudy(model_name)
    if cloudy:
        xs = [cloudy[n] / lexington_reference.median(model_name, n) for n in names if cloudy.get(n)]
        ys = [yy for yy, n in zip(y, names) if cloudy.get(n)]
        ax.scatter(xs, ys, marker="*", color=colors[3], s=130, label="Cloudy C25", zorder=4)
    ax.scatter([results[n] / lexington_reference.median(model_name, n) for n in names], y, color=colors[0], s=70, label="This code", zorder=5)
    ax.axvline(1.0, color="0.3", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 5.0)
    # 対数軸の補助目盛りのラベル (2×10⁻¹ など) が重なるので、主目盛りを明示して数値だけを書く
    ticks = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize="small")
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlabel("Ratio to the median of the benchmark codes")
    ax.set_title(title or f"Lexington 2000 {model_name}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize="small", borderaxespad=0.0)
    matplotlib.pyplot.tight_layout()
    os.makedirs(visualization.FIGURE_DIR, exist_ok=True)
    path = os.path.join(visualization.FIGURE_DIR, f"{output}.png")
    matplotlib.pyplot.savefig(path, dpi=150)
    matplotlib.pyplot.close(fig)
    return path


def plot_versus_cloudy(results, model_name, output, title=None):
    """本コード / Cloudy C25 の比 (対数軸) を figures/<output>_vs_cloudy.png に保存する (Cloudy の結果がなければ None)"""
    cloudy = read_cloudy(model_name)
    names = [name for name in results if cloudy.get(name)]
    if not names:
        return None
    y = numpy.arange(len(names))[::-1]
    seaborn.set_theme()
    seaborn.set_context("talk")
    seaborn.set_style("ticks")
    colors = seaborn.color_palette("hls", 4)
    fig, ax = matplotlib.pyplot.subplots(figsize=(10, 0.42 * len(names) + 2.0))
    ax.axvspan(0.9, 1.1, color="0.9", zorder=0)
    ax.scatter([results[n] / cloudy[n] for n in names], y, color=colors[0], s=70, zorder=3)
    ax.axvline(1.0, color="0.3", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(0.2, 5.0)
    ticks = [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize="small")
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlabel("This code / Cloudy C25 (grey band: ±10%)")
    ax.set_title(title or f"Lexington 2000 {model_name}")
    matplotlib.pyplot.tight_layout()
    path = os.path.join(visualization.FIGURE_DIR, f"{output}_vs_cloudy.png")
    matplotlib.pyplot.savefig(path, dpi=150)
    matplotlib.pyplot.close(fig)
    return path


def read_results(output):
    """save_results() の TSV を読む"""
    results = {}
    with open(os.path.join(visualization.OUTPUT_DIR, f"{output}_results.tsv")) as f:
        next(f)
        for line in f:
            name, value, _ = line.rstrip("\n").split("\t")
            results[name] = float(value)
    return results


def benchmark_quantities(model, model_name):
    """全セルを体積で積分した Hβ と各輝線の光度 (Hβ 比)、および温度などの平均量"""
    reference = lexington_reference.MODELS[model_name]
    abundances = reference["abundances"]
    volume = 4.0 / 3.0 * numpy.pi * (model.radius_edges[1:]**3 - model.radius_edges[:-1]**3)
    index = {symbol: k for k, symbol in enumerate(model.ion_symbols)}
    line_names = [name for name in reference["rows"] if name in LINES and LINES[name][0] in index]
    L_line = {name: 0.0 for name in line_names}
    L_Hbeta = 0.0
    integral_NeNp = integral_NeNpT = integral_NeHeplus = 0.0
    for i in range(model.N_r):
        n_ion = model.ion_element_density * model._ion_fraction_vector(i)
        n_e = model.electron_density_array[i]
        T = model.temperature_array[i]
        n_p, n_HI, n_Heplus = n_ion[index["H00"]], n_ion[index["H01"]], n_ion[index["He01"]]
        emissivity = numpy.exp(numpy.interp(numpy.log(T), numpy.log(HBETA_T), numpy.log(HBETA_EMISSIVITY)))
        L_Hbeta += emissivity * n_e * n_p * volume[i]
        integral_NeNp += n_e * n_p * volume[i]
        integral_NeNpT += n_e * n_p * T * volume[i]
        integral_NeHeplus += n_e * n_Heplus * volume[i]
        for name in line_names:
            symbol, ranges = LINES[name]
            k = index[symbol]
            ion_model = model.line_cooling.ions[k]
            if ion_model is None or n_ion[k] <= 0.0:
                continue
            betas = model._cell_betas(i)
            _, _, energy_eV, photons = ion_model.line_emission(n_e, n_HI, T, None if betas is None else betas[k])
            wavelength = hc_eV_angstrom / energy_eV
            selected = numpy.zeros(len(wavelength), dtype=bool)
            for lo, hi in ranges:
                selected |= (wavelength >= lo) & (wavelength <= hi)
            L_line[name] += numpy.sum(photons[selected] * energy_eV[selected]) * eV * n_ion[k] * volume[i]
    results = {"Hbeta": L_Hbeta / reference["Hbeta_unit"]}
    for name in line_names:
        results[name] = L_line[name] / L_Hbeta
    results["T_inner"] = model.temperature_array[0]
    results["<T[NpNe]>"] = integral_NeNpT / integral_NeNp
    if not reference["matter_bounded"]:
        x_Hplus = numpy.array([model._ion_fraction_vector(i)[index["H00"]] for i in range(model.N_r)])
        k_out = numpy.argmax(x_Hplus < 0.5) if numpy.any(x_Hplus < 0.5) else model.N_r - 1
        results["R_out"] = model.radius_edges[k_out] / reference["R_out_unit"]
    results["<He+>/<H+>"] = abundances["H"] / abundances["He"] * integral_NeHeplus / integral_NeNp
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HII40", choices=list(lexington_reference.MODELS))
    parser.add_argument("--n-cells", type=int, default=250)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--n-packets", type=int, default=200000)
    parser.add_argument("--case", default="B", choices=["A", "B"])
    parser.add_argument("--diffuse", action="store_true", help="H, He の拡散光を外向き近似で輸送する (Case A になる)")
    parser.add_argument("--line-transfer", action="store_true", help="輝線の光学的厚さを脱出確率で入れる (2 回目以降の反復で効く)")
    parser.add_argument("--output", default=None, help="output/<output>.tsv (既定: lexington_<model>[_diffuse])")
    parser.add_argument("--plot-only", action="store_true", help="計算せず output/<output>_results.tsv から図だけを描き直す")
    args = parser.parse_args()
    reference = lexington_reference.MODELS[args.model]
    if args.output is None:
        args.output = f"lexington_{args.model}" + ("_diffuse" if args.diffuse else "") + ("_lines" if args.line_transfer else "")
    mode = ("diffuse field (outward-only), Case A" if args.diffuse else f"Case {args.case}") + (", line transfer" if args.line_transfer else "")
    if args.plot_only:
        results = read_results(args.output)
        print(plot_results(results, args.model, args.output, f"Lexington 2000 {args.model} ({mode})"))
        print(plot_versus_cloudy(results, args.model, args.output, f"Lexington 2000 {args.model} ({mode})"))
        sys.exit(0)

    time_start = time.time()
    radius_edges = numpy.linspace(reference["R_in"], reference["r_max"], args.n_cells + 1)
    T_BB = reference["T_BB"]
    kT_eV = astropy.constants.k_B.cgs.value * T_BB / eV
    light_source = source.BlackbodySource(T_BB=T_BB, luminosity=reference["luminosity"], band_eV=(1.0e-4 * kT_eV, 40.0 * kT_eV))   # 全波長の光度
    elements = list(reference["abundances"])
    model = equilibrium.PhotoionizationModel(
        radius_edges=radius_edges, hydrogen_number_density=reference["n_H"], source=light_source,
        elements=elements, abundances=reference["abundances"], T_gas=1.0e4, N_nu=1001,
        recombination_case="A" if args.diffuse else args.case, diffuse=args.diffuse, line_transfer=args.line_transfer)
    model.setup_grids()
    model.calculate_equilibrium(N_packets=args.n_packets, max_global_iterations=args.iterations)
    print(model.save_tsv(f"{args.output}.tsv"))
    print(model.save_heating_cooling_tsv(f"{args.output}_heating_cooling.tsv"))
    results = benchmark_quantities(model, args.model)

    print(save_results(results, args.model, args.output))
    print(plot_results(results, args.model, args.output, f"Lexington 2000 {args.model} ({mode})"))
    print(plot_versus_cloudy(results, args.model, args.output, f"Lexington 2000 {args.model} ({mode})"))
    print(f"\nLexington {args.model} (this code: {mode}, {args.n_cells} cells, {model.global_iterations_done} global iterations)")
    print(f"  Hbeta in units of {reference['Hbeta_unit']:.0e} erg/s, R_out in {reference['R_out_unit']:.0e} cm, lines relative to Hbeta")
    cloudy = read_cloudy(args.model)
    print(f"{'quantity':22s} {'This code':>10s} {'Median':>9s} {'Cloudy01':>9s} {'XSTAR01':>9s} {'CloudyC25':>9s} {'This/Median':>11s} {'This/C25':>9s}")
    for name, value in results.items():
        median = lexington_reference.median(args.model, name)
        f = lambda v: f"{v:9.4g}" if v is not None else f"{'-':>9s}"
        ratio = f"{value / median:11.3f}" if median else f"{'-':>11s}"
        print(f"{name:22s} {value:10.4g} {f(median)} {f(lexington_reference.value(args.model, name, 'GF'))} "
              f"{f(lexington_reference.value(args.model, name, 'TK'))} {f(cloudy.get(name))} {ratio} "
              f"{(value / cloudy[name]) if cloudy.get(name) else float('nan'):9.3f}")
    print(f"\nComputation time: {(time.time() - time_start) / 60.0:.2f} min")
