import matplotlib.pyplot
import numpy
import os
import seaborn

# figureディレクトリのパス（プロジェクトルート直下）
FIGURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")

def plot_results(model):
    """計算結果をプロットして保存する（水素・ヘリウムのイオン割合と温度）"""
    seaborn.set_theme()
    seaborn.set_context("poster")
    seaborn.set_style("ticks")

    # --- 水素イオン割合の図 ---
    seaborn.set_palette("hls", 2)
    fig1, ax1 = matplotlib.pyplot.subplots(figsize=(16, 9))
    ax1.plot(model.r_pc, model.H00_fraction_array, label="H00_fraction", linewidth=2, marker="o", markersize=4)
    ax1.plot(model.r_pc, model.H01_fraction_array, label="H01_fraction", linewidth=2, marker="o", markersize=4)
    ax1.axvline(model.R_s_pc, linestyle='--', color='gray', label=f'Self-consistent $R_s$ = {model.R_s_pc:.2f} pc')
    ax1.set_xlabel('Distance from Star [pc]')
    ax1.set_ylabel('Ionization Fraction')
    ax1.legend(loc="upper left")
    ax1.set_ylim([-0.05, 1.40])
    ax1.set_xlim([0, model.r_pc[-1]])
    matplotlib.pyplot.tight_layout()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    ionization_filename = os.path.join(FIGURE_DIR, "hydrogen.png")
    matplotlib.pyplot.savefig(ionization_filename, dpi=200)
    matplotlib.pyplot.close(fig1)

    # --- ヘリウムイオン割合の図 ---
    seaborn.set_palette("Set2", 3)
    fig3, ax3 = matplotlib.pyplot.subplots(figsize=(16, 9))
    x_HeI = numpy.maximum(1.0 - model.He01_fraction_array - model.He00_fraction_array, 0.0)
    ax3.plot(model.r_pc, model.He02_fraction_array, linewidth=2, label="He02_fraction")
    ax3.plot(model.r_pc, model.He01_fraction_array, linewidth=2, label="He01_fraction")
    ax3.plot(model.r_pc, model.He00_fraction_array, linewidth=2, label="He00_fraction")
    ax3.axvline(model.R_s_pc, linestyle='--', color='gray', label=f'$R_s$ = {model.R_s_pc:.2f} pc')
    ax3.set_xlabel('Distance from Star [pc]')
    ax3.set_ylabel('Ionization Fraction')
    ax3.legend(loc="upper left")
    ax3.set_ylim([-0.05, 1.40])
    ax3.set_xlim([0, model.r_pc[-1]])
    matplotlib.pyplot.tight_layout()
    he_filename = os.path.join(FIGURE_DIR, 'helium_ionization.png')
    matplotlib.pyplot.savefig(he_filename, dpi=200)
    matplotlib.pyplot.close(fig3)

    # --- 温度の図 ---
    fig2, ax2 = matplotlib.pyplot.subplots(figsize=(16, 9))
    ax2.plot(model.r_pc, model.T_gas, color='orange', linewidth=2, label='Gas Temperature')
    ax2.set_xlabel('Distance from Star [pc]')
    ax2.set_ylabel('Temperature [K]')
    ax2.legend(loc="upper right")
    ax2.set_xlim([0, model.r_pc[-1]])
    matplotlib.pyplot.tight_layout()
    temperature_filename = os.path.join(FIGURE_DIR, 'temperature.png')
    matplotlib.pyplot.savefig(temperature_filename, dpi=150)
    matplotlib.pyplot.close(fig2)


if __name__ == "__main__":
    # TSVからデータを読み込んでプロットするためのダミークラス
    class MockModel:
        pass
    
    data_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "simulation.tsv")
    
    if not os.path.exists(data_file):
        print(f"エラー: データファイルが見つかりません。先に python3 photoionization.py を実行してください。\nパス: {data_file}")
    else:
        print(f"データファイル {data_file} を読み込んでいます...")
        data = numpy.loadtxt(data_file, skiprows=1)
        model = MockModel()
        model.r_pc = data[:, 0]
        model.H00_fraction_array = data[:, 1]
        model.H01_fraction_array = data[:, 2]
        model.He00_fraction_array = data[:, 3]
        model.He01_fraction_array = data[:, 4]
        model.He02_fraction_array = data[:, 5]
        model.T_gas = data[:, 6]
        
        # タイトル等に使うパラメータ（TSVに含まれないためデフォルト値を設定）
        model.T_BB = 40000
        model.hydrogen_number_density = 10000
        model.R_s_pc = numpy.nan # TSVからは求まらないため描画を省略する
        
        plot_results(model)
        print("図の再描画が完了しました。")