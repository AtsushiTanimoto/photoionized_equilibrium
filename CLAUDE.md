# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概要

光電離プラズマの電離平衡 + 熱平衡をモンテカルロ輻射輸送（連続吸収法）で計算する研究用 Python コード。
全元素（H〜Zn のうち原子データがあるもの）を共通の枠組みで扱い、XSTAR の計算（`benchmark/xstar/` に出力を同梱。
kT = 2.5 keV の黒体、log ξ = 3.5、n = 10⁹ cm⁻³、N_H = 10²¹ cm⁻²、`run_xstar.command` が入力パラメータ）との比較を主目的とする。
将来は同じ計算を MONACO（`~/software/monaco`）の中に C++ で実装する予定で、このコードはその試作。
単位系は cgs、エネルギーは eV で扱う。物理定数は各モジュールの先頭で `astropy.constants` / `astropy.units` から cgs 値を取って定義する（`constants.py` には置かない）。コメントや出力メッセージは日本語。

## 実行方法

ビルドシステム、テスト、リンタ、依存関係ファイルはない。必要なパッケージは numpy（`numpy.trapezoid` を使うため 2.0 以上）、scipy、astropy、matplotlib、seaborn。

原子データは外部データベースから import 時／`setup_grids()` 時に読む（`constants.py` のパス）:
- FAC（`ATOMIC_DATA_DIR` = `/Users/tanimoto/software/database/atomic_data`、MONACO と同じ DB）: 光電離断面積 `<元素>/<元素><電子数>.pi`。**電子数 ≤ 10 のイオンしかない**（H, He, C, N, O, Ne は全段階）。生成元は `~/github/atomicdatabase`。
- CHIANTI（`CHIANTI_DIR` = `/Users/tanimoto/software/database/chianti`、v11）: `ip/chianti.ip`（電離ポテンシャル）、`*.rrparams`/`*.drparams`（再結合）、`*.diparams`（衝突電離）、`*.elvlc`/`*.wgfa`/`*.scups`（輝線冷却）。

モジュールはフラットな import なので、`source/` のスクリプトは **`source/` ディレクトリ内から実行する**。
ベンチマーク関係 (`benchmark/`) は先頭で `source/` を import パスに加えているので `benchmark/` 内から実行する:

```sh
cd benchmark
python3 run_benchmark.py [--n-cells 1] [--n-packets 200000] [--iterations 2] [--elements H,He,O,Fe] [--fix-temperature 2.9e6] [--xstar-spectrum xstar/xout_spect1.fits] [--xstar-dir DIR]
    # → data/benchmark.tsv, data/benchmark_heating_cooling.tsv, data/benchmark_line_emission.tsv, figures/benchmark_<元素>_fraction.png（既定で benchmark/xstar の XSTAR を重ねる。全元素・2 セルで約 30 秒）
python3 compare_xstar.py [XSTAR_DIR]   # 既定で benchmark/xstar の xout_abund1.fits と温度・イオン割合・加熱冷却を比較
cd ../source
python3 run_simulation.py        # 従来の HII 領域モデル（黒体星 4×10⁴ K、H + He + C + N、Case B）→ figures/hii_region_*.png
cd ../benchmark
python3 run_xstar.py --name hii_region --log-xi 3.5832 --trad 4.0e4 --luminosity 3.6469e38 --density 1.0e4 --column 3.0826e22 --temperature 1.0e4 --abundances H=1,He=1.0235,C=0.7945,N=0.6036   # HII 領域と同じ設定で XSTAR（約 4 分）
python3 compare_hii_region.py    # 半径分布を XSTAR と重ねて figures/hii_region_<元素>_comparison.png, hii_region_temperature_comparison.png
python3 visualization.py --file ../data/benchmark.tsv [--radial]   # TSV から図だけを再描画
```

`--fix-temperature` は熱収支を解かずに与えた温度で電離平衡だけを解く（XSTAR の温度での収支の診断用）。
`benchmark/xstar/xout_spect1.fits`（入射スペクトル、6.6 MB）は `.gitignore` で除外しており、`--xstar-spectrum` を使うときだけ必要（既定の黒体光源なら不要。同じ黒体なので結果の差は 0.5% 程度）。

## アーキテクチャ

依存関係は `constants.py` → `atomic_data.py`, `line_cooling.py`, `recombination.py`, `bremsstrahlung.py`, `source.py` → `equilibrium.py` → `run_simulation.py` / `benchmark/run_benchmark.py` / `benchmark/compare_xstar.py` の一方向。`visualization.py` はモデルの属性か TSV の辞書を読むだけ。

- **`constants.py`**: DB のパス、制動放射の係数 `Lambda_ff_0`、元素表 `ELEMENT_SYMBOLS`/`ATOMIC_NUMBER`、太陽組成 `ABUNDANCES_ANGR`（XSTAR の angr）、`BENCHMARK_ELEMENTS`（MONACO の PhotoionizedPlasma と同じ 15 元素）。
- **`atomic_data.py`**: `get_element(symbol)` → `ElementData`（`ions[j]` は電荷 j の `IonData`、j = 0〜Z）。`IonData` は光電離断面積（MONACO と同じ式 σ = Σ Σ0 (E/I)^Γ exp(−E/τ)、閾値は CHIANTI の実験値に合わせて全チャネルを同じ量だけずらす）、加熱断面積 Σ σ_k (E − I_k)、このイオンから 1 階下への再結合 α_RR（Badnell type 1/2、Shull & Van Steenberg type 3）・α_DR（type 1、type 2 は 2 項に変換）、再結合冷却（RR は Hui & Gnedin 1997 の水素様 Case A の ⟨E⟩/kT を T/z² でスケール、DR は共鳴エネルギー k E_i で重み付け）、衝突電離 q_CI（Dere 2007 の diparams を Maxwell 分布で数値積分。励起自動電離 easplups は未実装）を与える。`ElementData.min_charge` は FAC データのある最低電荷で、それより低い電離段階は扱わない（鉄なら Fe XVII 以上）。
- **`line_cooling.py`**: CHIANTI の多準位モデルによる衝突励起冷却（低い方から `N_LEVELS_MAX` = 20 準位、温度グリッド 1〜10⁸ K）。`LineCoolingSet(names)` でイオンをまとめ、`.table(n_e, n_HI, fractions)` で表を作り `.rate(T, table)` で内挿する。`.scups` がないイオン（Cr, Mn の低電離段階など）は冷却ゼロ。C I/C II の微細構造準位は H 原子衝突も含む。C IV の `.elvlc` は `,` 以降に余分な列があるのでパーサは `,` の前だけを使う。
- **`source.py`**: `BlackbodySource(T_BB, radius_cm=… | luminosity=…)`（光度指定は XSTAR の bbody と同じ 1〜1000 Ry で規格化）と `TabulatedSource.from_xstar(xout_spect1.fits)`。ビンごとの光子放出率とパケットの多項分布サンプリングを持つ。
- **`equilibrium.py`** の `PhotoionizationModel(radius_edges, n_H, source, elements, abundances, …)`:
  - `setup_grids()`: エネルギービン（最低閾値〜光源の上限、対数）、全イオンの断面積行列 (N_ions × N_nu)、Compton の重み、再結合・衝突電離係数の温度表（log T = 0〜9）、輝線冷却のデータを準備する。
  - `calculate_equilibrium()`: 3 重のループ。(1) グローバル反復: パケットをビンにサンプリングし重み `W` を持たせる。(2) 半径セルを内側から外側へ: 連続吸収法で `W` が exp(−τ) 減衰。(3) セル内の局所反復（緩和係数 0.5）: パスレングス・エスティメータ φ(τ) = (1 − e^{−τ})/τ で局所の光子フラックス c_E を求め、光電離率 Γ と加熱率 → 熱収支（輝線 + RR + DR + 衝突電離 + 制動放射 + Compton 冷却 = 光電離加熱 + Compton 加熱）を `_solve_thermal_balance()` で解く（温度グリッド上で符号変化をすべて探し、現在の温度に最も近い根を `brentq` で求める。光電離プラズマは高温解と低温解が共存することがあり、log ξ = 3.0 では初期温度 ≥ 10⁶ K なら 2.1×10⁶ K、5×10⁴ K なら 8.4×10⁴ K に収束する。局所反復の最初に初期温度と整合する電離状態を作ってから始める）→ `_solve_all_elements()` で電離平衡（上向き Γ + n_e q_CI、下向き n_e (α_RR + α_DR)、対数で連鎖）。散乱による光子の輸送は入れていない。
  - 状態は `fraction_array[e]`（形状 (N_r, Z+1)、列 j が電荷 j）、`temperature_array`、`electron_density_array`、加熱・冷却の内訳 `heating_array`/`cooling_array`（元素ごと + Compton/制動放射など）。
  - `save_tsv()` の列は Radius_cm, Temperature, n_e, 各イオン（記号は `<元素><電子数 2 桁>`、裸は `<元素>00`。XSTAR CSV/MONACO と同じ規約）。`save_heating_cooling_tsv()` は内訳。`data/` は自動作成される。
  - `save_line_emission_tsv()`: 衝突励起による輝線をセルごと・イオンごと・線ごとに出力する（CHIANTI の準位番号、エネルギー、体積放射率 [photons s⁻¹ cm⁻³]、**光電離 1 回あたりの光子数** = 放射率/(Γ n_ion)）。後者は MONACO が光電離イベントごとに輝線を抽選する規格化で、MONACO の `.ln`（FAC の再結合カスケード、ce=0 で再生成する予定）の衝突励起成分に相当する。旧 `.ln` はイオン割合 1/(Z+1) を仮定して衝突励起を混ぜていたため光電離プラズマで過大だった。MONACO 側でセル（material）ごとの表を読む仕組みはまだない。
  - `recombination_case="B"` は H, He の再結合だけ `recombination.py` の Case B 近似式に置き換える（HII 領域向け）。
- **`benchmark/compare_xstar.py`**: XSTAR の `xout_abund1.fits`（ABUNDANCES の最終行。裸のイオンの列はなく 1 − Σ で求める。温度は 10⁴ K 単位、HEATING/COOLING は erg s⁻¹ cm⁻³）と比較する。**XSTAR の元素ごとの heating/cooling は共鳴線による連続光の散乱（吸収を加熱、再放射を冷却に計上）を含むため、元素ごとの値は本コードの光電離加熱・輝線冷却と直接は比べられず、差し引き（net）で比べる。**

## ベンチマークの現状 (2026-10-14)

- XSTAR (HEASoft 6.37.1) は `xstar.f90` の `tinf`（温度下限）が未初期化で segfault することがある。ローカルの `~/software/heasoft/.../xstar/src/xstar/xstar.f90` に `tinf=1.d-3` を入れて再ビルド済み（`xstar.f90.orig` が元）。`benchmark/run_xstar.py` は黒体温度・光度・密度・柱密度・存在比を指定できる。
- HII 領域モデルと XSTAR の比較（`compare_hii_region.py`）: 電離境界は本コード 0.165 pc、XSTAR 0.181 pc、HII 領域内の温度はどちらも約 1.3×10⁴ K。XSTAR は星の直近 (< 0.01 pc、log ξ > 3) で温度が 10 K の下限まで落ち、境界の外側でも H が 5〜25% 電離したまま温度 ≈ 10⁴ K を保つ（本コードは中性・18 K）。この外側の違いは XSTAR の入射黒体スペクトルの数値的な床が原因 (2026-10-15 に確認): `xout_spect1.fits` の incident は 200 eV までは Planck 関数に一致するが、それより上で減衰せず 300〜700 eV で約 10⁻⁹、1〜10 keV で約 10⁻¹⁹ (10³⁸ erg s⁻¹ erg⁻¹ 単位) の床が残る（真の Planck の 10¹¹〜10³⁷ 倍以上）。この 100 eV〜10 keV の光子 (約 3×10⁴⁹ s⁻¹) が電離境界を透過し、外側で必要な H 電離率 (約 10⁻¹⁰〜10⁻¹² s⁻¹) を桁で説明する。拡散放射（He II Lyα、再結合連続光）は中性 H に対する平均自由行程が短く原因ではない。物理的には本コードの「境界の外側は中性」が正しく、この領域の一致は比較の目標にしない。星直近の 10 K は別の問題で未調査。

1 セル・2×10⁵ パケットで、イオン割合は XSTAR と主要な段階で 10〜20% 以内（Fe XXIII〜XXVII は 5% 以内、Cr/Mn/Ni の低い段階は約 2 倍）。温度は本コード 3.9×10⁶ K、XSTAR 2.9×10⁶ K。XSTAR の温度で収支を評価すると本コードは加熱 1.17×10⁻⁵ > 冷却 0.95×10⁻⁵ erg s⁻¹ cm⁻³。主な差: XSTAR の O VIII Lyα 衝突強度が過大（約 19 倍、XSTAR チームに報告済み）で XSTAR の O 冷却が約 2.6×10⁻⁶ 過大、本コードの制動放射が XSTAR より約 20% 大きい（Gaunt 因子 1.35 vs 約 1.1）、Fe/O の光電離加熱が本コードでやや大きい。未実装: Auger 電離（内殻光電離後の多重電離）、励起自動電離、電荷交換、散乱の輸送、ダスト。

## 実装上の注意

- `from module import name` は使わず、`import module` して `module.name` で参照する（ユーザの方針）。

- 新しい冷却・加熱過程を加えるときは、`calculate_equilibrium()` 内の `cooling_terms()` と `_store_heating_cooling()`、`save_heating_cooling_tsv()`、`compare_xstar.py` の表示をすべて揃える。
- 温度に依存する係数は `_setup_rate_tables()` で温度表にして `rates_at(T)` で内挿している（`brentq` の中で毎回 CHIANTI の式を評価しないため）。新しい係数も同じ仕組みに載せる。
- FAC の `.pi` にないイオン（電子数 > 10）を扱うには `~/github/atomicdatabase` の `config.py`（`MAX_ELECTRON_NUMBERS`）でデータを生成する必要がある。
- グローバル反復の収束判定値は表示されるだけで、早期終了は実装されていない。
