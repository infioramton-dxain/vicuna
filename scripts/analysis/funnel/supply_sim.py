""
import sqlite3, sys, logging, time
from datetime import date
from pathlib import Path
import pandas as pd
sys.path.insert(0,"src")
logging.disable(logging.WARNING)
from macmini.ranking import exporter as ex
S="/tmp/vicuna_funnel_"
a=pd.read_pickle(S+"agg.pkl")
base=a[(a.trades>=20)&(a.pf>=1.1)&(a.dd>=-0.20)].sort_values("dsr",ascending=False).reset_index(drop=True)
db=Path("macmini.db").resolve()
c=sqlite3.connect(f"file:{db}?mode=ro",uri=True,timeout=60)
barred=ex._out_of_scope_symbols(c)
fs=dict(c.execute("select symbol,finnhub_sector from mm_sector_map"))
out=[]
for as_of in (date(2026,9,28),date(2026,9,21),date(2026,9,14)):
    t0=time.time(); veto={}; dq={}
    for _,r in base.iterrows():
        sym=r.sym; st="ok"
        if sym in barred: st="scope"
        else:
            if sym not in veto:
                veto[sym]=ex._veto_veto(c,sym,as_of)
            if veto[sym] is not None: st="tradability"
            else:
                if sym not in dq: dq[sym]=ex._qs_disqualify(ex._model_stack(c,sym,as_of))
                if dq[sym] is not None: st="quality"
                elif ex._today_signal_fires(db,int(r.sid),sym,as_of) is None: st="no_signal"
        out.append((as_of.isoformat(),int(r.sid),sym,r.sec,fs.get(sym),r.dsr,st))
    print(as_of,"done in",round(time.time()-t0),"s",flush=True)
o=pd.DataFrame(out,columns="as_of sid sym sec fs dsr st".split())
o.to_pickle(S+"supply.pkl")
for as_of,g in o.groupby("as_of"):
    print("\n===",as_of)
    for thr in (0.10,0.05,0.02,0.0):
        h=g[g.dsr>=thr]; ok=h[h.st=="ok"]
        uniq=ok.drop_duplicates("sym")
        capped=uniq.groupby("sec").head(ex.K_PER_SECTOR)
        print(f"DSR>={thr:<4}: cells={len(h):4d} | " + " ".join(f"{k}={v}" for k,v in h.st.value_counts().items())
              + f" | firing cells={len(ok)} unique symbols={len(uniq)} after 5/sector cap={len(capped)} sectors={capped.sec.value_counts().to_dict()}")
    if as_of=="2026-09-28":
        print("firing symbols at DSR>=0:", " ".join(g[(g.st=='ok')].drop_duplicates('sym').sym.tolist()))
