"""
XSTAR ベンチマーク (three_code_benchmark) と同じ設定で光電離平衡を計算するスクリプト
  黒体 kT = 2.5 keV (T = 2.9011×10^7 K)、L(1-1000 Ry) = 10^44 erg/s、n_H = 10^9 cm^-3、
  log ξ = 3.5 (r_in = 5.6234×10^15 cm)、N_H = 10^21 cm^-2 (厚さ 10^12 cm)、太陽組成 (angr)
計算後、XSTAR の出力 (xout_abund1.csv, xout_abund1.fits) があれば compare_xstar.py で比較できる。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source"))  # source/ のモジュールを使う
import argparse
import numpy
import time

import equilibrium
import source
import visualization
import constants


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xstar-spectrum", default=None, help="XSTAR の xout_spect1.fits を光源に使う（省略時は --source）")
    parser.add_argument("--source", default="bbody", choices=["bbody", "diskbb"], help="光源: 黒体 (bbody) か標準円盤の多温度黒体 (diskbb、温度は内縁温度)")
    parser.add_argument("--T-source", type=float, default=2.9011e+07, help="黒体温度 / 円盤の内縁温度 [K] (既定 kT = 2.5 keV)")
    parser.add_argument("--no-ea", action="store_true", help="衝突電離に励起自動電離を入れない (比較用)")
    parser.add_argument("--no-auger", action="store_true", help="内殻光電離に続く Auger 電離を入れない (比較用)")
    parser.add_argument("--n-cells", type=int, default=2)
    parser.add_argument("--n-packets", type=int, default=200000)
    parser.add_argument("--n-energy", type=int, default=2000)
    parser.add_argument("--iterations", type=int, default=10, help="グローバル反復の最大回数 (収束したらそこで終える)")
    parser.add_argument("--elements", default=",".join(constants.BENCHMARK_ELEMENTS))
    parser.add_argument("--T-init", type=float, default=1.0e6)
    parser.add_argument("--fix-temperature", type=float, default=None, help="熱収支を解かず、この温度 [K] で電離平衡だけを解く（収支の診断用）")
    parser.add_argument("--log-xi", type=float, default=3.5, help="電離パラメータ log ξ (内径 r_in = √(L/(n ξ)) で指定する)")
    parser.add_argument("--xstar-dir", default=None,
                        help="XSTAR の実行ディレクトリ (xout_abund1.fits)。図に XSTAR のイオン割合を重ねる。既定は xstar/logxi=<値>、\"\" で無効")
    parser.add_argument("--output", default=None, help="output/<output>.tsv などの名前。既定は benchmark_logxi=<値>")
    args = parser.parse_args()

    time_start = time.time()
    luminosity = 1.000e+44             # erg/s (1-1000 Ry)
    hydrogen_number_density = 1.000e+09   # cm^-3
    log_xi = args.log_xi
    if args.output is None:
        args.output = f"benchmark_logxi={log_xi:.3e}"
    if args.xstar_dir is None:
        args.xstar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xstar", f"logxi={log_xi:.3e}")
    column_density = 1.000e+21         # cm^-2
    radius_inner = numpy.sqrt(luminosity / (hydrogen_number_density * 10**log_xi))
    thickness = column_density / hydrogen_number_density
    radius_edges = numpy.linspace(radius_inner, radius_inner + thickness, args.n_cells + 1)

    if args.xstar_spectrum:
        light_source = source.TabulatedSource.from_xstar(args.xstar_spectrum)
    elif args.source == "diskbb":
        light_source = source.DiskBlackbodySource(T_in=args.T_source, luminosity=luminosity)
    else:
        light_source = source.BlackbodySource(T_BB=args.T_source, luminosity=luminosity)

    model = equilibrium.PhotoionizationModel(
        radius_edges=radius_edges, hydrogen_number_density=hydrogen_number_density, source=light_source,
        elements=args.elements.split(","), abundances=constants.ABUNDANCES_ANGR, T_gas=args.T_init, N_nu=args.n_energy,
        auger=not args.no_auger, excitation_autoionization=not args.no_ea)
    model.setup_grids()
    model.calculate_equilibrium(N_packets=args.n_packets, max_global_iterations=args.iterations, fixed_temperature=args.fix_temperature)
    print("Saving results to TSV...", flush=True)
    print(model.save_tsv(f"{args.output}.tsv"))
    print(model.save_heating_cooling_tsv(f"{args.output}_heating_cooling.tsv"))
    print(model.save_line_emission_tsv(f"{args.output}_line_emission.tsv"))
    print(model.save_level_population_tsv(f"{args.output}_level_populations"))
    print("Plotting results...", flush=True)
    xstar = None
    if args.xstar_dir:
        fits_path = os.path.join(args.xstar_dir, "xout_abund1.fits")
        if os.path.exists(fits_path):
            import compare_xstar
            xstar, _, _ = compare_xstar.read_xstar_fractions(fits_path)
        else:
            print(f"XSTAR の出力が見つからないので XSTAR を重ねずに描きます: {fits_path}")
    visualization.plot_ion_fractions(model, prefix=args.output, xstar=xstar)
    print(f"Computation time: {(time.time() - time_start) / 60.0:.3e} min", flush=True)
