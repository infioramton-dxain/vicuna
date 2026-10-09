""
import sqlite3, sys, numpy as np, pandas as pd
sys.path.insert(0,"src")
from macmini.ranking import exporter as ex
c=sqlite3.connect("file:macmini.db?mode=ro",uri=True,timeout=60)
run="bt_20260926T000005Z_1a877e"
df=pd.read_sql_query("select strategy_id sid,symbol sym,window_start ws,deflated_sharpe dsr,profit_factor pf,max_drawdown dd,n_trades nt,total_return ret,hit_rate hr from mm_backtest_runs where run_id=?",c,params=[run])
df=df.sort_values(["sid","sym","ws"],kind="stable")
df["k"]=df.groupby(["sid","sym"]).cumcount()
n=df.groupby(["sid","sym"]).k.transform("max")+1
print("windows per cell:",n.describe().round(1).to_dict())
W=ex.SURVIVAL_WINDOWS; lam=ex.RECENCY_LAMBDA
w=np.array([ex._recency_weight(a) for a in range(W)])
print("weights newest→oldest:",w.round(3))
recs=[]
for (sid,sym),g in df.groupby(["sid","sym"],sort=False):
    g=g.reset_index(drop=True); m=len(g)
    if m<W+3: continue
    dsr=g.dsr.values; pf=g.pf.values; dd=g.dd.values; nt=g.nt.values; ret=g.ret.values
    if nt.sum()==0: continue
    for k in range(W-1,m-3):
        sl=slice(k-W+1,k+1)
        t=nt[sl]
        tot=t[t>=1].sum()
        if tot<ex.MIN_TOTAL_TRADES: continue
        ww=w[::-1]
        d=(dsr[sl]*ww).sum()/ww.sum()
        mask=t>=1
        p=ex._winsor_wmedian(list(pf[sl][mask]),list(ww[mask]))
        x=dd[sl].min()
        recs.append((sid,sym,g.ws[k],d,p,x,tot,nt[k+3],ret[k+3],pf[k+3]))
r=pd.DataFrame(recs,columns="sid sym ws dsr pf dd trades f_nt f_ret f_pf".split())
r.to_pickle("/tmp/vicuna_funnel_gate_oos.pkl")
print("eval points (trade floor passed):",len(r), "distinct ws:",r.ws.nunique(), r.ws.min(), r.ws.max())
base=(r.pf>=ex.MIN_PROFIT_FACTOR)&(r.dd>=ex.MAX_DRAWDOWN_FLOOR)
def rep(name,m):
    s=r[m]; tr=s[s.f_nt>0]
    print(f"{name:34s} n={len(s):7d} fwd-traded={len(tr):7d} mean_ret={tr.f_ret.mean():+.4f} med_ret={tr.f_ret.median():+.4f} win={(tr.f_ret>0).mean():.3f} trades/win={s.f_nt.mean():.2f}")
rep("trade floor only",r.dsr>-9)
rep("trades+pf+dd (no DSR gate)",base)
for lo,hi in [(0,0.02),(0.02,0.05),(0.05,0.10),(0.10,9)]:
    rep(f"  base & DSR in [{lo},{hi})",base&(r.dsr>=lo)&(r.dsr<hi))
rep("fails pf or dd",~base)
for yr in ("2023","2024","2025"):
    m=r.ws.str[:4]==yr
    rep(f"[{yr}] base, DSR<0.10",m&base&(r.dsr<0.10)); rep(f"[{yr}] base, DSR>=0.10",m&base&(r.dsr>=0.10))
