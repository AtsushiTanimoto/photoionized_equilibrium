# H 原子との電荷交換の係数

Kingdon & Ferland (1996, ApJS, 106, 205) の解析フィットを Cloudy (https://gitlab.nublado.org/cloudy/cloudy, `data/`) の
データファイルからそのまま取ったもの (2026-10-15 取得)。1 行目はマジックナンバー、以降は Z = 1〜30 の各元素につき 4 行。

- `ctrecombdata.dat`: 再結合 A^(q+1)+ + H → A^q+ + H⁺ (行 i = 0〜3 が親イオンの電荷 q+1 = 1〜4)。
  列: a [10⁻⁹ cm³ s⁻¹], b, c, d, T_low [K], T_high [K], ΔE [eV]。
  k = a 10⁻⁹ (T/10⁴ K)^b [1 + c exp(d T/10⁴ K)]。T は [T_low, T_high] に切り詰める。a = 0 はデータなし。
- `ctiondata.dat`: 電離 A^q+ + H⁺ → A^(q+1)+ + H (行 i = 0〜3 が親イオンの電荷 q = 0〜3)。
  列: a, b, c, d, T_low, T_high, ΔT [10⁴ K], ΔE [eV]。k = 上と同じ式 × exp(−ΔT 10⁴ K / T)（Boltzmann 因子は切り詰めない温度で評価）。

Cloudy では He の再結合側 (Z = 2) の a に Stancil et al. (1998) の補正が入っている。
