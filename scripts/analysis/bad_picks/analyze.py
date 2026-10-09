""
import sys
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
e = pd.read_pickle(sys.argv[1])
e = e[e.br.notna()].copy()
e["yr"] = e.date.dt.year
e["stays_down"] = (e.how == "stop") & (e.f40 < 0)
COST = 0.0015

def line(g):
    return pd.Series({
        "n": len(g), "native_mean": g.nat.mean() - COST, "native_win": (g.nat > COST).mean(),
        "native_days": g.nat_days.median(),
        "br_mean": g.br.mean() - COST, "br_win": (g.br > COST).mean(), "br_days": g.br_days.median(),
        "stop%": (g.how == "stop").mean(), "target%": (g.how == "target").mean(),
        "stays_down%": g.stays_down.mean(), "f20": g.f20.mean()})

def cut(df, col, title):
    print(f"\n== {title}")
    print(df.groupby(col, observed=True).apply(line).round(4).to_string())

print("events:", len(e), "| cost per round trip assumed:", COST)
print("\n== ALL EVENTS"); print(line(e).round(4).to_string())
e["vol_b"] = pd.cut(e.vol60, [0, .25, .35, .45, .55, .65, .80, 9], labels=["<25", "25-35", "35-45", "45-55", "55-65", "65-80", ">80"])
cut(e, "vol_b", "by 60d realized vol (annualized %)")
cut(e, "yr", "by year")
print("\n== vol>65% vs <=65%, by year (bracket mean, net)")
e["hv"] = np.where(e.vol60 > .65, "vol>65", "vol<=65")
print(e.pivot_table(index="yr", columns="hv", values="br", aggfunc=["mean", "count"]).round(4).to_string())
e["trend"] = np.where((e.vs_sma200 > 0) & (e.mom63 > 0), "up", np.where((e.vs_sma200 < 0) & (e.mom63 < 0), "down", "mixed"))
cut(e, "trend", "by trend state (SMA200 + 3-month)")
e["m21"] = pd.cut(e.mom21, [-9, -.15, -.05, .05, .15, 9], labels=["<-15%", "-15..-5", "-5..5", "5..15", ">15%"])
cut(e, "m21", "by 1-month momentum at entry")
e["oh"] = pd.cut(e.off_high252, [-9, -.4, -.25, -.1, .001], labels=[">40% off high", "25-40%", "10-25%", "<10%"])
cut(e, "oh", "by distance from 52w high")
s = e.groupby("sid").apply(line)
s["hold_ratio"] = s.br_days / s.native_days
print("\n== by strategy (sorted by bracket mean)")
print(s.sort_values("br_mean", ascending=False).round(4).to_string())
print("\n== strategies 14 / 17 / 27 / 11 / 7 / 55, vol split")
print(e[e.sid.isin([14, 17, 27, 11, 7, 55])].groupby(["sid", "hv"]).apply(line).round(4).to_string())
