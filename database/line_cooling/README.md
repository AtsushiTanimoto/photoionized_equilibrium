# 輝線冷却の準位数

`n_levels.tsv` は `source/line_cooling.py` の多準位モデルで、イオンごとに考慮する準位数（CHIANTI の準位番号の低い方から）。
`source/make_line_cooling_levels.py` が CHIANTI v11 から作る（`source/` 内で `uv run python make_line_cooling_levels.py`、全イオンで約 2.5 分）。

各イオンを全準位で読み、低い方から N 準位に切り詰めた冷却率を全準位の値と比べる。
光電離プラズマでそのイオンが存在しうる温度（3000 K 〜 max(IP/10k, 2×10⁴ K) の 6 点）と n_e = 10², 10⁶, 10¹⁰ cm⁻³ のすべてで、
相対差が 2% 以内になる最小の N を選ぶ（候補 20, 30, 40, 50, 60, 80, 100, 150, 200, 250, 300、上限 300）。

列:
- `ion`: CHIANTI のイオン名（例 `fe_2` = Fe II）
- `N_levels_all`: CHIANTI の全準位数
- `N_levels`: 採用した準位数
- `max_relative_error`: 採用した準位数での全準位の値との最大相対差
- `cooling_all_over_N20`: 全準位の冷却率 / 20 準位の冷却率の最大値（旧既定の 20 準位の取りこぼし）

表にないイオンは 20 準位（`line_cooling.N_LEVELS_MAX`）。上限 300 で 2% に届かないのは Cr II（13%）だけ。
