"""
HII 領域モデル（黒体星に照らされた一様密度ガス球、H + He + C）の光電離平衡計算のメインスクリプト
"""
import numpy
import time

import equilibrium
import source
import visualization
from constants import pc, R_sun, Y_He, A_C


if __name__ == "__main__":
    time_start = time.time()
    radius_edges            = numpy.linspace(0.0, 1.0, 101) * pc      # セル境界 [cm]
    radius_edges[0]         = 1.0e-3 * pc                             # 中心は特異点なので小さくずらす
    hydrogen_number_density = 1.000e+04                               # [cm^-3]
    light_source            = source.BlackbodySource(T_BB=4.0e4, radius_cm=10.0 * R_sun)  # O 型星

    model = equilibrium.PhotoionizationModel(
        radius_edges=radius_edges, hydrogen_number_density=hydrogen_number_density, source=light_source,
        elements=["H", "He", "C"], abundances={"H": 1.0, "He": Y_He, "C": A_C},
        T_gas=1e4,                  # ガスの初期温度 [K]
        N_nu=1001,                  # エネルギービン数
        recombination_case="B")     # H, He は Case B
    model.setup_grids()
    model.calculate_equilibrium(N_packets=200000, max_global_iterations=1, max_local_iterations=100)
    print("Plotting results...", flush=True)
    visualization.plot_radial_profiles(model, prefix="hii_region")
    print("Saving results to TSV...", flush=True)
    print(model.save_tsv("hii_region.tsv"))
    print(model.save_heating_cooling_tsv("hii_region_heating_cooling.tsv"))
    print("Done.", flush=True)
    print(f"Computation time: {(time.time() - time_start) / 60.0:.3e} min", flush=True)
