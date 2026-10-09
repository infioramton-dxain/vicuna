import sqlite3, pandas as pd, numpy as np, sys
sys.path.insert(0,"src")
from macmini.ranking import exporter as ex
c=sqlite3.connect("file:macmini.db?mode=ro",uri=True,timeout=30)
run=c.execute("select last_run_id from mm_leaderboard limit 1").fetchone()[0]
print("leaderboard run:",run)
df=pd.read_sql_query("select strategy_id,symbol,sector,window_start,deflated_sharpe,profit_factor,max_drawdown,n_trades from mm_backtest_runs where run_id=?",c,params=[run])
print("rows",len(df),"windows",df.window_start.nunique(),"range",df.window_start.min(),df.window_start.max())
df=df.sort_values("window_start",kind="stable")
df["row_num"]=df.groupby(["strategy_id","symbol"]).cumcount(ascending=False)
r=df[df.row_num<ex.SURVIVAL_WINDOWS]
out=[]
for (sid,sym,sec),g in r.groupby(["strategy_id","symbol","sector"],sort=False):
    w=g.row_num.map(ex._recency_weight)
    t=g[g.n_trades>=1]
    out.append(dict(sid=sid,sym=sym,sec=sec,dsr=float((g.deflated_sharpe*w).sum()/w.sum()),
        dsr_plain=float(g.deflated_sharpe.mean()),dd=float(g.max_drawdown.min()),
        trades=int(t.n_trades.sum()),nwin=len(t),
        pf=ex._winsor_wmedian(t.profit_factor.tolist(),t.row_num.map(ex._recency_weight).tolist())))
a=pd.DataFrame(out)
a.to_pickle("/tmp/vicuna_funnel_agg.pkl")
n=len(a)
T=a.trades>=ex.MIN_TOTAL_TRADES; D=a.dsr>=ex.MIN_DEFLATED_SHARPE; P=a.pf>=ex.MIN_PROFIT_FACTOR; X=a.dd>=ex.MAX_DRAWDOWN_FLOOR
print(f"cells {n}")
print("zero-trade cells:",(a.trades==0).sum())
for name,m in [("trades>=20",T),("dsr>=0.10",D),("pf>=1.1",P),("dd>=-0.20",X)]:
    print(f"  pass {name:12s}: {m.sum():6d} ({m.mean():.1%})")
print("sequential: trades",T.sum(),"-> +pf",(T&P).sum(),"-> +dd",(T&P&X).sum(),"-> +dsr",(T&P&X&D).sum())
print("all but one gate:")
print("  w/o trades:",(D&P&X).sum(),"  w/o dsr:",(T&P&X).sum(),"  w/o pf:",(T&D&X).sum(),"  w/o dd:",(T&D&P).sum())
print("trade count distribution (cells with >=1 trade):",a[a.trades>0].trades.describe().round(1).to_dict())
print("trades thresholds:",{k:int((a.trades>=k).sum()) for k in (5,10,15,20,30)})
base=T&P&X
print("DSR dist among cells passing trades+pf+dd:",a[base].dsr.describe().round(3).to_dict())
print("DSR threshold sweep (with other gates):",{k:int((base&(a.dsr>=k)).sum()) for k in (0.0,0.02,0.05,0.08,0.10,0.15)})
print("plain-mean DSR>=0.10 w/ other gates:",int((base&(a.dsr_plain>=0.10)).sum()))
print("raw DSR distribution all window rows:",df.deflated_sharpe.describe().round(4).to_dict())
print("raw DSR ==0 share:",(df.deflated_sharpe==0).mean().round(3), " n_trades==0 share:",(df.n_trades==0).mean().round(3))
print("sectors (cells / qualifying):")
a["q"]=T&D&P&X
print(a.groupby("sec").agg(cells=("q","size"),syms=("sym","nunique"),qual=("q","sum")).sort_values("cells",ascending=False).to_string())
