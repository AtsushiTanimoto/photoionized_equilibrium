# 共有 FAC データベースにないイオンの光電離断面積

`ATOMIC_DATA_DIR`（`~/software/database/atomic_data`、電子数 ≤ 10 のイオンのみ）にない `.pi` をここに置く。
形式は共有データベースと同じ（`atomic_data.load_photoionization_channels` が読む）。

- `Mg/Mg11.pi` (Mg II): 2026-10-15 に pfac 1.1.5 で
  `pfac.atom.atomic_data(nele=[11, 12], asym='Mg', dir='Mg/', no_ce=True, no_ci=True, no_ai=True, no_tr=True)`
  を実行し（RR のみ、約 3 分）、`make_pi_from_fac.py Mg 11 <出力ディレクトリ>` で基底準位のチャネルをフィットしたもの。
  価電子 (3s) の断面積は Verner et al. (1996) と 10% 以内で一致する。66 eV 以上の 2p 内殻のチャネルも含む。
- Mg I (12 電子) は FAC の値が閾値付近で Verner の 1/100 になり使えないので、`.pi` は置かず EPICS2025/EPDL97
  (`constants.NEUTRAL_ABSORPTION_DIRS`) の表を `atomic_data.load_epdl97_table` で使う。
