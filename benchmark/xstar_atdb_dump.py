"""XSTAR の原子データベース atdb.fits から指定イオンのレコードを取り出して表示する（診断用）。
  python3 xstar_atdb_dump.py ~/software/heasoft/heasoft-6.37.1/ftools/xstar/data/atdb.fits ne_ii
イオン名は atdb の表記 (小文字、例 ne_ii)。lrtyp 1/6/8/15 (電離、再結合、衝突電離) と基底準位の型 53 (OP 光電離断面積) を表示する。"""
import sys, numpy as np, astropy.io.fits as f
h=f.open(sys.argv[1]); target=sys.argv[2].lower()
ptr=np.asarray(h['POINTERS'].data['pointers'][0]).reshape(-1,10)   # 列: np1, ltyp, lrtyp, lcon, nrdt, nidt, nkdt, np1r, np1i, np1k
rdat=np.asarray(h['REALS'].data['reals'][0]); idat=np.asarray(h['INTEGERS'].data['integers'][0]); kdat=np.asarray(h['CHARS'].data['char'][0])
def kstr(k0,n): return bytes(kdat[k0-1:k0-1+n]).decode('ascii','replace')
cur_ion=None; hits=[]; ion_names=[]
for rec in ptr:
    np1,ltyp,lrtyp,lcon,nrdt,nidt,nkdt,np1r,np1i,np1k=rec
    if lrtyp==12:
        cur_ion=kstr(np1k,nkdt); ion_names.append(cur_ion)
        continue
    if lrtyp==11: cur_ion=None; continue
    if cur_ion is not None and target == cur_ion.lower().strip():
        hits.append((ltyp,lrtyp,rdat[np1r-1:np1r-1+nrdt],idat[np1i-1:np1i-1+nidt],kstr(np1k,nkdt) if nkdt>0 else ''))
print('ion records matching:', [n for n in ion_names if target == n.lower().strip()][:5], ' total ion records', len(ion_names))
want={1,6,8,15}
shown53=0
from collections import Counter
print('record counts by (lrtyp,ltyp):', sorted(Counter((a,b) for a,b,*_ in hits).items()))
for ltyp,lrtyp,r,i,k in hits:
    if lrtyp==7 and ltyp==53 and shown53<2:
        shown53+=1; print(f'lrtyp=7 ltyp=53 (OP PI) nrdt={len(r)} idat={i.tolist()} rdat[:12]={np.array2string(r[:12],precision=4,max_line_width=200)}')
    if lrtyp in want:
        print(f'lrtyp={lrtyp} ltyp={ltyp} rdat={np.array2string(r,precision=4,max_line_width=200)} idat={i.tolist()} kdat={k!r}')
