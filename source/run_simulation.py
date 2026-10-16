"""
HII 領域モデル（黒体星に照らされた一様密度ガス球、元素は constants.HII_REGION_ELEMENTS）の光電離平衡計算のメインスクリプト
  uv run python run_simulation.py [--elements H,He] [--output hii_region]
"""
import argparse
import astropy.constants
import astropy.units
import numpy
import time

import equilibrium
import source
import constants

# 物理定数 (cgs、astropy.constants の CODATA 値)
R_sun = astropy.constants.R_sun.cgs.value  # 太陽半径 [cm]
pc = astropy.constants.pc.cgs.value  # パーセク [cm]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", default=",".join(constants.HII_REGION_ELEMENTS), help="元素記号をカンマ区切り")
    parser.add_argument("--output", default="hii_region", help="output/<output>.tsv などの名前")
    parser.add_argument("--case", default="B", choices=["A", "B"], help="H, He の再結合 (B: 拡散光子がその場で吸収される近似)")
    args = parser.parse_args()
    elements = [symbol.strip() for symbol in args.elements.split(",")]
    time_start = time.time()
    radius_edges            = numpy.linspace(0.0, 1.0, 101) * pc      # セル境界 [cm]
    radius_edges[0]         = 1.0e-3 * pc                             # 中心は特異点なので小さくずらす
    hydrogen_number_density = 1.000e+04                               # [cm^-3]
    light_source            = source.BlackbodySource(T_BB=4.0e4, radius_cm=10.0 * R_sun)  # O 型星

    model = equilibrium.PhotoionizationModel(
        radius_edges=radius_edges, hydrogen_number_density=hydrogen_number_density, source=light_source,
        elements=elements,
        abundances={symbol: constants.ABUNDANCES_ANGR[symbol] for symbol in elements},   # 太陽組成 angr (XSTAR と同じ)
        T_gas=1e4,                  # ガスの初期温度 [K]
        N_nu=1001,                  # エネルギービン数
        recombination_case=args.case)   # H, He は既定で Case B
    model.setup_grids()
    model.calculate_equilibrium(N_packets=200000, max_global_iterations=1, max_local_iterations=100)
    # 図は benchmark/compare_hii_region.py (XSTAR との比較図) で描く。本コードだけの図が必要なら
    # cd source && python3 visualization.py --file ../output/hii_region.tsv --radial
    print("Saving results to TSV...", flush=True)
    print(model.save_tsv(f"{args.output}.tsv"))
    print(model.save_heating_cooling_tsv(f"{args.output}_heating_cooling.tsv"))
    print(model.save_level_population_tsv(f"{args.output}_level_populations"))
    print("Done.", flush=True)
    print(f"Computation time: {(time.time() - time_start) / 60.0:.3e} min", flush=True)
