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
- FAC（`ATOMIC_DATA_DIR` = `/Users/tanimoto/software/database/atomic_data`、MONACO と同じ DB）: 光電離断面積 `<元素>/<元素><電子数>.pi`。**電子数 ≤ 10 のイオンしかない**（H, He, C, N, O, Ne は全段階）。生成元は `~/github/atomicdatabase`。電子数 > 10 のイオンは `source/data/fac_supplement/`（`FAC_SUPPLEMENT_DIR`、`make_pi_from_fac.py` で FAC の `.rr` から同じ形式に変換。現状 Mg II）で補い、FAC データのない中性原子は EPICS2025 の表（`NEUTRAL_ABSORPTION_DIRS`、`~/github/neutral_matter/database/absorption/epics2025/pe-cs-<ZZ>.dat`、MeV と barn。閾値が実験値で、値は EPDL97 と同じ。なければ `~/software/database/neutral_matter/absorption/geant4_epdl97`。MONACO 既定の `xraylib_rtab` は 10 eV からしかなく閾値 < 10 eV の原子に使えない）を使う（現状 Mg I。FAC の Mg I は閾値付近で Verner の 1/100 で使えない）。
- CHIANTI（`CHIANTI_DIR` = `/Users/tanimoto/software/database/chianti`、v11）: `ip/chianti.ip`（電離ポテンシャル）、`*.rrparams`/`*.drparams`（再結合）、`*.diparams`（衝突電離）、`*.elvlc`/`*.wgfa`/`*.scups`（輝線冷却）。

モジュールはフラットな import なので、`source/` のスクリプトは **`source/` ディレクトリ内から実行する**。
ベンチマーク関係 (`benchmark/`) は先頭で `source/` を import パスに加えているので `benchmark/` 内から実行する:

```sh
cd benchmark
python3 run_benchmark.py [--n-cells 1] [--n-packets 200000] [--iterations 2] [--elements H,He,O,Fe] [--fix-temperature 2.9e6] [--xstar-spectrum xstar/xout_spect1.fits] [--xstar-dir DIR]
    # → data/benchmark.tsv, data/benchmark_heating_cooling.tsv, data/benchmark_line_emission.tsv, data/benchmark_level_populations/<イオン>.tsv, figures/benchmark_<元素>_fraction.png（既定で benchmark/xstar の XSTAR を重ねる。全元素・2 セルで約 30 秒）
python3 compare_xstar.py [XSTAR_DIR]   # 既定で benchmark/xstar の xout_abund1.fits と温度・イオン割合・加熱冷却を比較
cd ../source
python3 run_simulation.py        # HII 領域モデル（黒体星 4×10⁴ K、H + He + C + N + O + Ne + Mg、太陽組成 angr、Case B）→ figures/hii_region_*.png, data/hii_region{,_heating_cooling}.tsv, data/hii_region_level_populations/<イオン>.tsv
cd ../benchmark
python3 run_xstar.py --name hii_region --log-xi 3.5832 --trad 4.0e4 --luminosity 3.6469e38 --density 1.0e4 --column 3.0826e22 --temperature 1.0e4 --abundances H=1,He=1,C=1,N=1,O=1,Ne=1,Mg=1 --bbody-table --cfrac 1 --name hii_region_cfrac1   # HII 領域と同じ設定で XSTAR（約 2 分。cfrac=1 の理由は下記）
python3 compare_hii_region.py --xstar-dir xstar/hii_region_cfrac1   # 半径分布を XSTAR と重ねて figures/hii_region_<元素>_comparison.png, hii_region_temperature_comparison.png
python3 visualization.py --file ../data/benchmark.tsv [--radial]   # TSV から図だけを再描画
```

`--fix-temperature` は熱収支を解かずに与えた温度で電離平衡だけを解く（XSTAR の温度での収支の診断用）。
`benchmark/xstar/xout_spect1.fits`（入射スペクトル、6.6 MB）は `.gitignore` で除外しており、`--xstar-spectrum` を使うときだけ必要（既定の黒体光源なら不要。同じ黒体なので結果の差は 0.5% 程度）。

## アーキテクチャ

依存関係は `constants.py` → `atomic_data.py`, `line_cooling.py`, `recombination.py`, `bremsstrahlung.py`, `charge_exchange.py`, `source.py` → `equilibrium.py` → `run_simulation.py` / `benchmark/run_benchmark.py` / `benchmark/compare_xstar.py` の一方向。`visualization.py` はモデルの属性か TSV の辞書を読むだけ。

- **`constants.py`**: DB のパス、制動放射の係数 `Lambda_ff_0`、元素表 `ELEMENT_SYMBOLS`/`ATOMIC_NUMBER`、太陽組成 `ABUNDANCES_ANGR`（XSTAR の angr）、`BENCHMARK_ELEMENTS`（MONACO の PhotoionizedPlasma と同じ 15 元素）。
- **`atomic_data.py`**: `get_element(symbol)` → `ElementData`（`ions[j]` は電荷 j の `IonData`、j = 0〜Z）。`IonData` は光電離断面積（MONACO と同じ式 σ = Σ Σ0 (E/I)^Γ exp(−E/τ)、閾値は CHIANTI の実験値に合わせて全チャネルを同じ量だけずらす）、加熱断面積 Σ σ_k (E − I_k)、このイオンから 1 階下への再結合 α_RR（Badnell type 1/2、Shull & Van Steenberg type 3）・α_DR（type 1、type 2 は 2 項に変換）、再結合冷却（RR は Hui & Gnedin 1997 の水素様 Case A の ⟨E⟩/kT を T/z² でスケール、DR は共鳴エネルギー k E_i で重み付け）、衝突電離 q_CI（Dere 2007 の diparams を Maxwell 分布で数値積分。励起自動電離 easplups は未実装）を与える。`ElementData.min_charge` は光電離データ（FAC、補助 `.pi`、中性原子の EPDL97 表）のある最低電荷で、それより低い電離段階は扱わない（鉄なら Fe XVII 以上）。EPICS2025/EPDL97 の表は `IonData.table` に持ち log-log 内挿する（加熱は σ(E)(E − I)）。
- **`line_cooling.py`**: CHIANTI の多準位モデルによる衝突励起冷却（低い方から `N_LEVELS_MAX` = 20 準位、温度グリッド 1〜10⁸ K）。`LineCoolingSet(names)` でイオンをまとめ、`.table(n_e, n_HI, fractions)` で表を作り `.rate(T, table)` で内挿する。`.scups` がないイオン（Cr, Mn の低電離段階など）は冷却ゼロ。C I/C II の微細構造準位は H 原子衝突も含む。C IV の `.elvlc` は `,` 以降に余分な列があるのでパーサは `,` の前だけを使う。
- **`charge_exchange.py`**: H 原子・H⁺ との電荷交換の係数 (Kingdon & Ferland 1996 の解析フィット。データは Cloudy の `ctrecombdata.dat`/`ctiondata.dat` の写しを `source/data/charge_exchange/` に同梱、書式は同じディレクトリの README)。再結合側は親イオンの電荷 1〜4、電離側は 0〜3 のみ。反応のエネルギー差による加熱・冷却は未実装。
- **`source.py`**: `BlackbodySource(T_BB, radius_cm=… | luminosity=…)`（光度指定は XSTAR の bbody と同じ 1〜1000 Ry で規格化）と `TabulatedSource.from_xstar(xout_spect1.fits)`。ビンごとの光子放出率とパケットの多項分布サンプリングを持つ。
- **`equilibrium.py`** の `PhotoionizationModel(radius_edges, n_H, source, elements, abundances, …)`:
  - `setup_grids()`: エネルギービン（最低閾値〜光源の上限、対数）、全イオンの断面積行列 (N_ions × N_nu)、Compton の重み、再結合・衝突電離係数の温度表（log T = 0〜9）、輝線冷却のデータを準備する。
  - `calculate_equilibrium()`: 3 重のループ。(1) グローバル反復: パケットをビンにサンプリングし重み `W` を持たせる。(2) 半径セルを内側から外側へ: 連続吸収法で `W` が exp(−τ) 減衰。(3) セル内の局所反復（緩和係数 0.5）: パスレングス・エスティメータ φ(τ) = (1 − e^{−τ})/τ で局所の光子フラックス c_E を求め、光電離率 Γ と加熱率 → 熱収支（輝線 + RR + DR + 衝突電離 + 制動放射 + Compton 冷却 = 光電離加熱 + Compton 加熱）を `_solve_thermal_balance()` で解く（温度グリッド上で符号変化をすべて探し、現在の温度に最も近い根を `brentq` で求める。光電離プラズマは高温解と低温解が共存することがあり、log ξ = 3.0 では初期温度 ≥ 10⁶ K なら 2.1×10⁶ K、5×10⁴ K なら 8.4×10⁴ K に収束する。局所反復の最初に初期温度と整合する電離状態を作ってから始める）→ `_solve_all_elements()` で電離平衡（上向き Γ + n_e q_CI + n_HII k_CX_ion、下向き n_e (α_RR + α_DR) + n_HI k_CX_rec、対数で連鎖。電荷交換の H の密度は直前の電離状態から渡す）。散乱による光子の輸送は入れていない。
  - 状態は `fraction_array[e]`（形状 (N_r, Z+1)、列 j が電荷 j）、`temperature_array`、`electron_density_array`、加熱・冷却の内訳 `heating_array`/`cooling_array`（元素ごと + Compton/制動放射など）。
  - `save_tsv()` の列は Radius_cm, Temperature, n_e, 各イオン（記号は `<元素><電子数 2 桁>`、裸は `<元素>00`。XSTAR CSV/MONACO と同じ規約）。`save_heating_cooling_tsv()` は内訳。`data/` は自動作成される。
  - `save_line_emission_tsv()`: 衝突励起による輝線をセルごと・イオンごと・線ごとに出力する（CHIANTI の準位番号、エネルギー、体積放射率 [photons s⁻¹ cm⁻³]、**光電離 1 回あたりの光子数** = 放射率/(Γ n_ion)）。後者は MONACO が光電離イベントごとに輝線を抽選する規格化で、MONACO の `.ln`（FAC の再結合カスケード、ce=0 で再生成する予定）の衝突励起成分に相当する。旧 `.ln` はイオン割合 1/(Z+1) を仮定して衝突励起を混ぜていたため光電離プラズマで過大だった。MONACO 側でセル（material）ごとの表を読む仕組みはまだない。
  - `save_level_population_tsv()`: 輝線冷却の多準位モデルで解いた準位占有数（衝突励起・脱励起と自発放射だけの統計平衡。光励起・再結合カスケードは含まない）をイオンごとのファイル `data/<prefix>/<イオン記号>.tsv`（行がセル、列が準位 `level_01`…、CHIANTI の準位番号順、イオン内の占有率 Σ = 1）に出力する。
  - `recombination_case="B"` は H, He の再結合だけ `recombination.py` の Case B 近似式に置き換える（HII 領域向け）。
- **`benchmark/compare_xstar.py`**: XSTAR の `xout_abund1.fits`（ABUNDANCES の最終行。裸のイオンの列はなく 1 − Σ で求める。温度は 10⁴ K 単位、HEATING/COOLING は erg s⁻¹ cm⁻³）と比較する。**XSTAR の元素ごとの heating/cooling は共鳴線による連続光の散乱（吸収を加熱、再放射を冷却に計上）を含むため、元素ごとの値は本コードの光電離加熱・輝線冷却と直接は比べられず、差し引き（net）で比べる。**

## ベンチマークの現状 (2026-10-14)

- XSTAR (HEASoft 6.37.1) は `xstar.f90` の `tinf`（温度下限）が未初期化で segfault することがある。ローカルの `~/software/heasoft/.../xstar/src/xstar/xstar.f90` に `tinf=1.d-3` を入れて再ビルド済み（`xstar.f90.orig` が元）。`benchmark/run_xstar.py` は黒体温度・光度・密度・柱密度・存在比を指定できる。
- HII 領域モデルと XSTAR の比較（`compare_hii_region.py`）: 電離境界は本コード 0.165 pc、XSTAR 0.181 pc、HII 領域内の温度はどちらも約 1.3×10⁴ K。XSTAR は星の直近 (< 0.01 pc、log ξ > 3) で温度が 10 K の下限まで落ち、境界の外側でも H が 5〜25% 電離したまま温度 ≈ 10⁴ K を保つ（本コードは中性・18 K）。この外側の違いの原因 (2026-10-15 に XSTAR の診断実行で特定): XSTAR は 13.6 eV 未満の星光（Balmer 連続光と Lyα 付近）で H を電離している。`run_xstar.py --bbody-table --bbody-zero-below 13.6`（Lyman 連続光だけを入射）にすると外側の x_e は 0.25 → 0.02 に落ち、`--bbody-zero-range 10.0,10.4`（Lyα 付近だけを消す）では x_e が半分、温度が 10⁴ K → 10²〜10³ K に落ちる。XSTAR では電離境界の外の中性 H 柱密度 2×10²² cm⁻² を 10.0〜13.5 eV の星光が 20〜90% 透過しており（物理的には Lyα の減衰翼だけで薄い層で吸収されるはず）、脱出確率法で星の Lyα が全ゾーンの n = 2 を励起し、Balmer 連続光（光子数は Lyman 連続光と同程度）が n = 2 から光電離する。本コードは基底状態からの光電離しか扱わないので中性になる。XSTAR 側の機構 (xstarlib/src): `heatt.f90` がマスタースペクトル `zrems(1,:)` に拡散放射（`linopac` で連続光ビンに足した線放射 `rccemis` を含む）を加え、第 1 パスでは `trnfrc.f90` がそれを局所の放射場 `bremsa` に使い、`ucalc.f90` の型 50 の連続光ポンピング `ans2 = σ bremsa(nb1) (1 − cfrac)` が拡散 Lyα で各ゾーンの n = 2 を励起し、型 12/36（励起準位の光電離）で Balmer 連続光が電離する。`run_xstar.py --cfrac 1` でこのポンピング項を切ると外側は中性・49 K になり本コードと一致する（ただし cfrac は拡散放射の内外の配分も変える）。現実の HII 領域では n = 2 の励起は境界のごく薄い層に限られるので、この領域の一致は比較の目標にしない。なお XSTAR 内蔵の bbody は `xstarlib/src/starf.f90` で E/kT > 60 の指数関数を頭打ちにし E/kT > 150 で 1/E の尾を足すため 200 eV 以上に床 (10⁻⁹〜10⁻¹⁹ × 10³⁸ erg s⁻¹ erg⁻¹) があるが、`--bbody-table`（Planck の表を `spectrum=file` で渡す。床は 10⁻³⁸ 以下になる）で消しても外側の結果は変わらず、床は原因ではない（`xstar/hii_region_table/`）。`run_xstar.py --cfrac 1` では外側が中性・49 K、星直近の 10 K の落ち込みも消え、電離境界と HII 領域内の温度が本コードと一致する（`xstar/hii_region_cfrac1/`、`figures/hii_region_*_comparison.png` はこの比較）。現在のモデル (2026-10-15、H + He + C + N + O + Ne + Mg、angr) では O III の輝線冷却が支配的（冷却の 6 割、Ne III が 1.5 割）で、電離境界は本コード 0.145 pc、XSTAR 0.151 pc、HII 領域内の温度は本コード 6600〜7400 K、XSTAR 7400〜8400 K（本コードが 10% 低い。酸素なしでは両者 1.3×10⁴ K）。O III/O II の境界は一致するが、Ne III → Ne II の移行は XSTAR の方が内側で起きる（0.31 pc で Ne II が XSTAR 91%、本コード 26%）。Mg は HII 領域内で両コードとも Mg III が 99% で、Mg II は本コードが XSTAR の 0.4 倍（0.1 pc で 0.4% vs 1%）。Mg の冷却は全体の 10⁻³ 以下で温度に効かない。XSTAR の Mg III → Mg II の RR/DR は CHIANTI と同一。この Ne の差の調査 (2026-10-15): (Ne III/Ne II)/(O III/O II) は本コードでは全域で 0.27〜0.4 と一定だが、XSTAR では内側で 0.14（本コードの半分）、外縁に向かって 10⁻³ まで急減する。Ne の存在比を 1/100 にした XSTAR でも急減は残るので Ne II の自己遮蔽ではない。FAC の光電離断面積は Verner et al. (1996) と Ne II・O II で 41〜54 eV で ±20% 以内（He I は閾値で FAC が 22% 大きい）なので断面積でもない。H との電荷交換 (Kingdon & Ferland 1996) を本コードに入れても結果はほぼ変わらない（HII 領域内は n_HI k_CX ≲ 0.05 n_e α。Ne III + H の係数は 10⁻¹⁴ で無視できる）ので XSTAR 側を疑い、`atdb.fits`（POINTERS/REALS/INTEGERS/CHARS の 4 拡張。レコードごとに 10 個のポインタ: np1, ltyp, lrtyp, lcon, nrdt, nidt, nkdt, np1r, np1i, np1k。lrtyp = 12 のイオンレコード (`ne_ii` など) の後に続くレコードがそのイオンのデータで、lrtyp = 8 はそのイオンへの再結合 (1 階上から)、lrtyp = 7・ltyp = 53 は準位別の OP 光電離断面積 [E − E_th (Ry), σ (Mb)] の組）を読んだ結果: Ne III → Ne II の RR/DR (Badnell、ltyp 38/39) は CHIANTI と同一、電荷交換は両者とも無視できる。一方 XSTAR の Ne II 基底準位の光電離断面積は終状態 Ne III ³P₂, ³P₁ の 2 チャネルしかなく（³P₀, ¹D, ¹S がない）、43〜51 eV で 3.1〜3.4 Mb と Verner (7.6〜8.5 Mb) や FAC の 0.4 倍。O II も 5 チャネルで Verner の 0.5〜0.6 倍。これが内側の 2 倍の差の原因とみられる（XSTAR チームへ報告する価値あり）。外縁に向かう急減はまだ説明できていない。XSTAR は境界直後で x_e < xeemin となり計算を打ち切る。残る差は境界位置 7% と星から 0.03 pc 以内の電離度・温度（XSTAR は O IV 優勢で 1.1〜1.3×10⁴ K、本コードは O III 優勢で 9×10³ K。本コードの最内セルは幅 0.01 pc で分解能が足りない）。図の凡例のイオン記号は `visualization.ion_label` の `<元素><電子数 2 桁>` 形式。

1 セル・2×10⁵ パケットで、イオン割合は XSTAR と主要な段階で 10〜20% 以内（Fe XXIII〜XXVII は 5% 以内、Cr/Mn/Ni の低い段階は約 2 倍）。温度は本コード 3.9×10⁶ K、XSTAR 2.9×10⁶ K。XSTAR の温度で収支を評価すると本コードは加熱 1.17×10⁻⁵ > 冷却 0.95×10⁻⁵ erg s⁻¹ cm⁻³。主な差: XSTAR の O VIII Lyα 衝突強度が過大（約 19 倍、XSTAR チームに報告済み）で XSTAR の O 冷却が約 2.6×10⁻⁶ 過大、本コードの制動放射が XSTAR より約 20% 大きい（Gaunt 因子 1.35 vs 約 1.1）、Fe/O の光電離加熱が本コードでやや大きい。未実装: Auger 電離（内殻光電離後の多重電離）、励起自動電離、電荷交換の加熱・冷却、散乱の輸送、ダスト。

## 実装上の注意

- `from module import name` は使わず、`import module` して `module.name` で参照する（ユーザの方針）。

- 新しい冷却・加熱過程を加えるときは、`calculate_equilibrium()` 内の `cooling_terms()` と `_store_heating_cooling()`、`save_heating_cooling_tsv()`、`compare_xstar.py` の表示をすべて揃える。
- 温度に依存する係数は `_setup_rate_tables()` で温度表にして `rates_at(T)` で内挿している（`brentq` の中で毎回 CHIANTI の式を評価しないため）。新しい係数も同じ仕組みに載せる。
- FAC の `.pi` にないイオン（電子数 > 10）を扱うには、`pfac.atom.atomic_data(nele=[...], asym=..., dir=..., no_ce=True, no_ci=True, no_ai=True, no_tr=True)` で RR だけ計算し（数分）、`source/make_pi_from_fac.py` で `source/data/fac_supplement/<元素>/` に `.pi` を作る（`README.md` 参照）。共有 DB を更新するなら `~/github/atomicdatabase` の `config.py`（`MAX_ELECTRON_NUMBERS`）で全パイプラインを回す。
- グローバル反復の収束判定値は表示されるだけで、早期終了は実装されていない。
