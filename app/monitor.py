""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

PROJ_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJ_ROOT / "macmini.db"
ENV_PATH = PROJ_ROOT / ".env"
CASES_DIR = PROJ_ROOT / "data" / "investment_cases"


def _load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


ENV = _load_env(ENV_PATH)


@st.cache_data(ttl=15)
def alpaca_get(path: str) -> dict | list:
    ""
    key = ENV.get("ALPACA_API_KEY")
    secret = ENV.get("ALPACA_SECRET_KEY")
    base = ENV.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    if not key or not secret:
        return {"error": "Alpaca creds missing from .env"}
    r = requests.get(
        f"{base}{path}",
        headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
        timeout=10,
    )
    try:
        return r.json()
    except ValueError:
        return {"error": r.text}


@st.cache_data(ttl=60)
def db_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as c:
        return pd.read_sql_query(sql, c, params=params)


@st.cache_data(ttl=60)
def db_mtime() -> datetime:
    return datetime.fromtimestamp(DB_PATH.stat().st_mtime)


_PICK_HEAD_RE = re.compile(r"^## Pick #(\d+) — ([A-Z][A-Z0-9.\-]*)")
_DECISION_RE = re.compile(r"\*\*Decision\*\*:\s*(.+)")


@st.cache_data(ttl=300)
def load_investment_cases() -> tuple[pd.DataFrame, pd.DataFrame]:
    ""
    rows: list[dict] = []
    extra_rows: list[dict] = []
    files = sorted(CASES_DIR.glob("*.md")) if CASES_DIR.exists() else []
    for f in files:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
        if not m:
            continue
        day = m.group(1)
        chunks = re.split(r"(?m)^(?=## )", f.read_text())
        preamble = chunks[0].strip()
        extras_by_title: dict[str, str] = {}
        for chunk in chunks[1:]:
            head = _PICK_HEAD_RE.match(chunk)
            if head:
                dm = _DECISION_RE.search(chunk)
                raw = dm.group(1).strip() if dm else ""
                decision = ("approve" if "APPROVE" in raw.upper()
                            else "reject" if "REJECT" in raw.upper() else "other")
                rows.append({
                    "as_of_date": day,
                    "pick_rank": int(head.group(1)),
                    "symbol": head.group(2),
                    "decision": decision,
                    "decision_raw": raw,
                    "case_md": chunk,
                    "preamble": preamble,
                })
            else:
                title = chunk.splitlines()[0].lstrip("# ").strip()
                if title.lower() == "portfolio decisions log":
                    preamble = f"{preamble}\n\n{chunk.strip()}"
                else:
                    extras_by_title[title] = chunk
        for title, chunk in extras_by_title.items():
            extra_rows.append({"as_of_date": day, "title": title, "section_md": chunk})
    return pd.DataFrame(rows), pd.DataFrame(extra_rows)


INK = "#141414"
ACCENT = "#1b5fa8"
RED = "#c62828"
GAIN, LOSS = "#2e9e5b", "#e23b3b"
GAINT, LOSST = "#1e7a45", "#c0271f"
CASH_FILL = "#eef1f4"
NEU3 = "#e3e6ea"
SERIF = "'Source Serif 4', Georgia, serif"
MUTED = "rgba(20,20,20,0.55)"
DIVIDER = "rgba(20,20,20,0.15)"

PLOTLY_CFG = {"displayModeBar": False, "responsive": True}

MODEL_SERIES = ["#1b5fa8", "#b0711f", "#c23a63", "#7b4b94", "#0e8a70"]
STATUS_GREY = "#90a4ae"

SECTOR_COLORS = {
    "Software": "#1b5fa8",
    "Mining": "#b0711f",
    "Defense": "#c23a63",
    "Consumer Staples": "#7b4b94",
    "Computers": "#0e8a70",
    "Materials": "#a8552d",
    "Other": "#6b7a84",
}


def nl_layout(**extra) -> dict:
    ""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SERIF, color=INK, size=13),
        margin=dict(l=56, r=20, t=8, b=36), hovermode="closest",
        hoverlabel=dict(bgcolor="#f4f6f8", bordercolor="rgba(20,20,20,0.25)",
                        font=dict(family=SERIF, color=INK, size=13)),
    )
    base.update(extra)
    return base


def sector_symbol_picker(syms: list[str], sector_of: dict[str, str], *,
                         prefix: str, default_sym: str | None = None,
                         dot: bool = True) -> str:
    ""
    letters = [chr(c) for c in range(65, 91)]
    have_letters = {s[0].upper() for s in syms}
    letter_state = f"{prefix}_letter"
    sym_state = f"{prefix}_sym"
    if st.session_state.get(letter_state) not in letters:
        st.session_state[letter_state] = (
            default_sym[0].upper() if default_sym else "A")

    def _bkey(sym: str) -> str:
        return f"{prefix}_stk_" + "".join(
            ch if ch.isalnum() else "_" for ch in sym)

    picker_key = f"{prefix}_picker"
    css_rules = [
        f".st-key-{picker_key}{{background:#e9edf1;border-radius:8px;"
        "padding:14px 14px 8px 14px;}",
        f'.st-key-{picker_key} div[class*="st-key-{prefix}_lbtn_"] '
        'button[data-testid="stBaseButton-secondary"]{'
        "background:#ffffff;}",
        f'.st-key-{picker_key} div[class*="st-key-{prefix}_lbtn_"] button{{'
        "width:100%;min-height:30px;padding:2px 0;font-size:12px;"
        "justify-content:center;}}",
    ]
    with st.container(key=picker_key):
        lcols = st.columns(len(letters))
        for lc, L in zip(lcols, letters):
            if lc.button(
                L, key=f"{prefix}_lbtn_{L}", disabled=L not in have_letters,
                type="primary" if st.session_state[letter_state] == L
                else "secondary",
            ):
                st.session_state[letter_state] = L
                st.rerun()
        letter = st.session_state[letter_state]
        letter_syms = [s for s in syms if s[0].upper() == letter]

        if st.session_state.get(sym_state) not in syms:
            st.session_state[sym_state] = (
                default_sym if default_sym in syms else (letter_syms or syms)[0])
        sel = st.session_state[sym_state]

        per_row = 12
        for row_start in range(0, len(letter_syms), per_row):
            row = letter_syms[row_start:row_start + per_row]
            scols = st.columns(per_row)
            for sc, s in zip(scols, row):
                color = SECTOR_COLORS.get(sector_of.get(s, "Other"),
                                          SECTOR_COLORS["Other"])
                key = _bkey(s)
                selected = "box-shadow:0 0 0 2px #141414;" if s == sel else ""
                css_rules.append(
                    f".st-key-{key} button{{background:#ffffff;"
                    f"border-color:rgba(20,20,20,0.15);font-size:11.5px;"
                    f"padding:2px 3px;min-height:30px;width:100%;"
                    f"display:flex;align-items:center;justify-content:center;"
                    f"{selected}}}")
                if dot:
                    css_rules.append(
                        f".st-key-{key} button::before{{content:'';"
                        f"display:inline-block;width:8px;height:8px;"
                        f"border-radius:50%;background:{color};"
                        f"margin-right:5px;flex:0 0 auto;}}")
                css_rules.append(
                    f".st-key-{key} button p{{color:{color};margin:0;"
                    f"font-weight:600;white-space:nowrap;}}")
                if sc.button(s, key=key):
                    st.session_state[sym_state] = s
                    st.rerun()
        legend = " &nbsp; ".join(
            f'<span style="display:inline-block;width:8px;height:8px;'
            f'border-radius:50%;background:{c};margin-right:4px;"></span>'
            f'<span class="nl-muted" style="font-size:12px;">{s}</span>'
            for s, c in SECTOR_COLORS.items())
        st.markdown(f'<div style="margin:6px 0 4px 2px;">{legend}</div>',
                    unsafe_allow_html=True)
    st.markdown("<style>" + "".join(css_rules) + "</style>",
                unsafe_allow_html=True)
    return sel


def nl_axis(**extra) -> dict:
    base = dict(gridcolor="rgba(20,20,20,0.10)",
                zerolinecolor="rgba(20,20,20,0.30)",
                linecolor="rgba(20,20,20,0.25)",
                tickfont=dict(size=12), automargin=True)
    base.update(extra)
    return base


def money(n: float, d: int = 0) -> str:
    return f"${abs(n):,.{d}f}"


def smoney(n: float, d: int = 0) -> str:
    return ("−" if n < 0 else "+") + money(n, d)


def pctf(n: float, d: int = 1) -> str:
    return ("−" if n < 0 else "+") + f"{abs(n):.{d}f}%"


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _mix(a: str, b: str, t: float) -> str:
    A, B = _hex_rgb(a), _hex_rgb(b)
    return "rgb(" + ",".join(str(round(A[i] + (B[i] - A[i]) * t)) for i in range(3)) + ")"


def div_color(upl: float, max_abs: float) -> str:
    ""
    if max_abs <= 0:
        return NEU3
    t = max(-1.0, min(1.0, upl / max_abs))
    return _mix(NEU3, GAIN, t) if t >= 0 else _mix(NEU3, LOSS, -t)


def kpi_strip(metrics: list[tuple[str, str, str, str, str]]) -> None:
    ""
    _v = (f"font-family:{SERIF};font-weight:600;font-size:26px;line-height:1.05;"
          "margin-top:7px;font-variant-numeric:tabular-nums;letter-spacing:-0.02em;")
    cells = "".join(
        f'<div style="padding:0 18px;border-left:1px solid {DIVIDER};">'
        f'<div class="nl-kicker-muted">{lbl}</div>'
        f'<div style="{_v}color:{vc};">{val}</div>'
        f'<div style="font-size:11.5px;margin-top:4px;color:{sc};">{sub}</div></div>'
        for lbl, val, vc, sub, sc in metrics)
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat({len(metrics)},1fr);'
        f'gap:2px;border-top:1px solid {DIVIDER};border-bottom:1px solid {DIVIDER};'
        f'padding:14px 0;margin:6px 0 12px;">{cells}</div>',
        unsafe_allow_html=True)


_SIGNAL_TABLES = ("mm_dl_signals", "mm_kronos_signals")


def _gate_module():
    for p in (str(PROJ_ROOT / "src"), str(PROJ_ROOT / "scripts" / "dl")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import eval_dl_gate
    return eval_dl_gate


@st.cache_data(ttl=1800)
def ic_series(model_id: str, table: str, asof: str) -> pd.Series:
    ""
    if table not in _SIGNAL_TABLES:
        raise ValueError(table)
    g = _gate_module()
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as c:
        return g.daily_rank_ic(c, model_id, table=table)


@st.cache_data(ttl=1800)
def rank_cross_section(model_id: str, table: str, asof: str) -> pd.DataFrame:
    ""
    if table not in _SIGNAL_TABLES:
        raise ValueError(table)
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as c:
        sig = pd.read_sql_query(
            f"SELECT symbol, date, predicted_5d_return FROM {table} "
            "WHERE model_id = ? AND predicted_5d_return IS NOT NULL",
            c, params=(model_id,))
        if sig.empty:
            return pd.DataFrame()
        bars = pd.read_sql_query(
            "SELECT symbol, date, open FROM mm_bars WHERE date >= ?",
            c, params=(str(sig["date"].min()),))
    opens = bars.pivot(index="date", columns="symbol", values="open").sort_index()
    fwd = opens.shift(-6) / opens.shift(-1) - 1.0
    fwd_long = fwd.stack().rename("realized_5d").reset_index()
    merged = sig.merge(fwd_long, on=["date", "symbol"], how="inner").dropna()
    if merged.empty:
        return pd.DataFrame()
    sizes = merged.groupby("date").size()
    sizes = sizes[sizes >= 30]
    if sizes.empty:
        return pd.DataFrame()
    day = sizes.index.max()
    cs = merged[merged["date"] == day].copy()
    cs["pred_rank"] = cs["predicted_5d_return"].rank(pct=True)
    cs["real_rank"] = cs["realized_5d"].rank(pct=True)
    cs["day"] = day
    return cs


@st.cache_data(ttl=86400, show_spinner="Computing market structure "
                                       "(cached until tomorrow)…")
def structure_data(asof: str) -> dict | None:
    ""
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.manifold import MDS
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    universe = pd.read_csv(PROJ_ROOT / "symbols.csv")["symbol"].tolist()
    start = (pd.Timestamp(asof) - pd.Timedelta(days=420)).date().isoformat()
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as c:
        bars = pd.read_sql_query(
            "SELECT symbol, date, close FROM mm_bars WHERE date >= ?",
            c, params=(start,))
        sec_of = dict(c.execute(
            "SELECT symbol, COALESCE(sector,'Other') FROM mm_sector_map"))
    if bars.empty:
        return None

    px = bars.pivot(index="date", columns="symbol", values="close").sort_index()
    px = px[[s for s in universe if s in px.columns]].tail(253)
    rets = px.pct_change(fill_method=None).iloc[1:]
    rets = rets.clip(rets.quantile(0.01), rets.quantile(0.99), axis=1)
    n_universe = rets.shape[1]
    rets = rets.loc[:, rets.notna().mean() >= 0.90].fillna(0.0)
    syms = rets.columns.tolist()
    T, N = rets.shape
    if N < 50:
        return None

    def _residualize(r: pd.DataFrame):
        mkt = r.mean(axis=1)
        mc = mkt - mkt.mean()
        beta = r.sub(r.mean()).mul(mc, axis=0).sum() / (mc ** 2).sum()
        return r.sub(r.mean()) - np.outer(mc, beta), beta

    def _cluster(r: pd.DataFrame, k: int | None = None):
        res, _ = _residualize(r)
        corr = np.corrcoef(res.values, rowvar=False)
        dist = np.sqrt(np.clip(2.0 * (1.0 - corr), 0, None))
        np.fill_diagonal(dist, 0.0)
        Z = linkage(squareform(dist, checks=False), method="average")
        sil = None
        if k is None:
            scores = {}
            for kk in range(4, 21):
                lab = fcluster(Z, kk, criterion="maxclust")
                if len(set(lab)) < 2:
                    continue
                scores[kk] = silhouette_score(dist, lab,
                                              metric="precomputed")
            smax = max(scores.values())
            k = max(kk for kk, s in scores.items() if s >= 0.9 * smax)
            sil = scores[k]
        return fcluster(Z, k, criterion="maxclust"), corr, dist, k, sil

    lab_raw, corr, dist, k, sil = _cluster(rets)
    _, beta = _residualize(rets)

    sizes = pd.Series(lab_raw).value_counts()
    remap = {old: new + 1 for new, old in enumerate(sizes.index)}
    labels = np.array([remap[v] for v in lab_raw])

    sectors = [sec_of.get(s, "Other") for s in syms]
    cluster_meta = []
    for cid in range(1, k + 1):
        idx = np.where(labels == cid)[0]
        top = pd.Series([sectors[i] for i in idx]).value_counts()
        sub = corr[np.ix_(idx, idx)]
        intra = (float(sub[np.triu_indices(len(idx), 1)].mean())
                 if len(idx) > 1 else 0.0)
        cluster_meta.append({
            "id": cid, "n": int(len(idx)), "dom": top.index[0],
            "name": f"C{cid} · {top.index[0]} ({len(idx)})",
            "intra": intra})

    order = np.lexsort((syms, labels))
    block = np.zeros((k, k))
    for a in range(1, k + 1):
        ia = np.where(labels == a)[0]
        for b in range(a, k + 1):
            ib = np.where(labels == b)[0]
            sub = corr[np.ix_(ia, ib)]
            v = (sub[np.triu_indices(len(ia), 1)].mean()
                 if a == b and len(ia) > 1 else sub.mean())
            block[a - 1, b - 1] = block[b - 1, a - 1] = float(v)

    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
              n_init=2, normalized_stress="auto").fit_transform(dist)

    std = (rets - rets.mean()) / rets.std()
    evals = np.linalg.eigvalsh(np.corrcoef(std.values, rowvar=False))[::-1]
    mp_upper = float((1 + np.sqrt(N / T)) ** 2)
    n_real = int((evals > mp_upper).sum())

    half = T // 2
    lab1, *_ = _cluster(rets.iloc[:half], k=k)
    lab2, *_ = _cluster(rets.iloc[half:], k=k)

    return {
        "window": (str(rets.index[0]), str(rets.index[-1])),
        "syms": syms, "sectors": sectors, "labels": labels,
        "cluster_meta": cluster_meta, "order": order,
        "corr": corr.astype(np.float32), "block": block,
        "mds": mds,
        "win_ret": (px.ffill().iloc[-1] / px.bfill().iloc[0] - 1.0)[syms].values,
        "beta": beta[syms].values, "evals": evals[:40],
        "mp_upper": mp_upper, "n_real": n_real, "k": k, "sil": float(sil),
        "pc1_pct": float(evals[0] / N),
        "ari_sector": float(adjusted_rand_score(sectors, labels)),
        "ari_halves": float(adjusted_rand_score(lab1, lab2)),
        "n_dropped": n_universe - N,
    }


def _short_model(mid: str) -> str:
    ""
    m = re.match(r"(.+?)(?:_zoo)?_20(\d{2})(\d{2})(\d{2})$", mid)
    if m:
        return f"{m.group(1)} {m.group(3)}-{m.group(4)}"
    return mid


_MODEL_GLOSSARY = {
    "attention_lstm": ("ALS", "LSTM recurrent network with an attention "
                       "layer — learns which past days matter most instead "
                       "of weighting all history equally."),
    "autoencoder": ("AEC", "Autoencoder — compresses recent history into a "
                    "small code and predicts from it; good at denoising."),
    "blend_bma": ("BMA", "Bayesian model average of the whole zoo — weights "
                  "members by recent evidence instead of picking one winner."),
    "deepar": ("DPR", "DeepAR — autoregressive RNN that outputs a "
               "probability distribution per day, not just a point guess."),
    "gbdt": ("GBD", "Gradient-boosted decision trees on engineered features "
             "— the non-neural baseline; strong on tabular signals."),
    "gru": ("GRU", "Gated recurrent unit network — the lightest recurrent "
            "model; fast to train and hard to overfit."),
    "kronos_base_zs_t10p90s3": ("KRN", "Kronos foundation model run "
                                "zero-shot — pretrained on other markets' "
                                "bars, never fitted to ours."),
    "llm_arbiter_deepseek": ("LLM", "deepseek LLM arbiter — reads the "
                             "numeric evidence and takes a daily stance; "
                             "scored like any other pool member."),
    "nhits": ("NHI", "N-HiTS — hierarchical interpolation across multiple "
              "time scales; built for multi-horizon forecasts."),
    "patchtst_seq64_dm128": ("PTS", "PatchTST — transformer over patches of "
                             "the price series; the original DL line (its "
                             "sign gate is audit-quarantined)."),
    "tcn": ("TCN", "Temporal convolutional network — dilated causal "
            "convolutions see long history without recurrence."),
    "tft": ("TFT", "Temporal Fusion Transformer — attention plus gating "
            "with explicit feature selection."),
    "transformer": ("TRF", "Plain transformer encoder over the bar "
                    "sequence — the attention baseline."),
    "vae": ("VAE", "Variational autoencoder — learns a probabilistic "
            "latent space of price patterns and predicts from it."),
    "wavenet": ("WAV", "WaveNet — stacked dilated convolutions borrowed "
                "from audio modeling."),
}


def _model_family(mid: str) -> str:
    m = re.match(r"(.+?)(?:_zoo)?_20\d{6}$", mid)
    return m.group(1) if m else mid


def _model_date(mid: str) -> str:
    m = re.match(r".+?_20(\d{2})(\d{2})(\d{2})$", mid)
    return f"20{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "rolling"


def _model_codes(mids: list[str]) -> dict[str, str]:
    ""
    from collections import Counter
    fam_n = Counter(_model_family(m) for m in mids)
    out: dict[str, str] = {}
    used: set[str] = set()
    for mid in mids:
        fam = _model_family(mid)
        base = _MODEL_GLOSSARY.get(fam, (fam[:3].upper(),))[0]
        code = base
        if fam_n[fam] > 1:
            code = base[:2] + _model_date(mid).replace("-", "")[-1]
        i = 1
        while code in used:
            code = base[:2] + str(i)
            i += 1
        used.add(code)
        out[mid] = code
    return out


st.set_page_config(
    page_title="Vicuna",
    page_icon=":llama:",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,400;0,600;1,400&display=swap');
    @import url('https://unpkg.com/@phosphor-icons/web@2.1.1/src/duotone/style.css');

    html, body, [class*="css"], .stMarkdown, .stDataFrame, button, input, select {
        font-family: "Source Serif 4", Georgia, serif !important;
    }
    .stApp { background: #ffffff; }
    /* hide Streamlit's fixed chrome bar — it overlays the wordmark */
    header[data-testid="stHeader"] { display: none; }
    .block-container { max-width: 1200px; padding-top: 2.2rem; }
    h1, h2, h3, h4 { font-family: "Source Serif 4", Georgia, serif !important;
                     letter-spacing: -0.015em; color: #141414; }
    h2 { font-size: 33px !important; margin-top: 1.5rem !important; }
    h4 { font-size: 20px !important; }
    .dataframe { font-size: 0.9rem; }

    /* tab bar — serif, sits above the account strip. Inactive labels are
       accent-blue (reads as clickable); the active one is ink + underline. */
    .stTabs [data-baseweb="tab-list"] { gap: 26px; border-bottom: 1px solid rgba(20,20,20,0.15); }
    .stTabs [data-baseweb="tab"] { padding: 8px 2px 12px; background: none; }
    .stTabs [data-baseweb="tab"] [data-testid="stMarkdownContainer"] p {
        font-size: 15px !important; font-weight: 600;
        color: #1b5fa8;
        font-family: "Source Serif 4", Georgia, serif !important;
    }
    .stTabs [aria-selected="true"] { border-bottom: 2px solid #1b5fa8 !important; }
    .stTabs [aria-selected="true"] [data-testid="stMarkdownContainer"] p {
        color: #141414 !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #1b5fa8; }

    /* newsprint furniture */
    .nl-kicker { font-size: 10.5px; letter-spacing: 0.1em; text-transform: uppercase;
                 color: #1b5fa8; }
    .nl-kicker-muted { font-size: 10.5px; letter-spacing: 0.09em; text-transform: uppercase;
                 color: rgba(20,20,20,0.55); }
    .nl-h4 { font-family: "Source Serif 4", Georgia, serif; font-weight: 600;
             font-size: 20px; margin: 5px 0 0; letter-spacing: -0.015em; }
    .nl-muted { color: rgba(20,20,20,0.55); }
    .nl-rule { border: none; border-top: 1px solid rgba(20,20,20,0.15); margin: 0; }
    .nl-empty { padding: 26px; border: 1px dashed rgba(20,20,20,0.15); border-radius: 3px;
                display: flex; align-items: center; gap: 12px; font-size: 13.5px; }
    .tag-neutral { display:inline-block; padding: 1px 8px; border: 1px solid rgba(20,20,20,0.15);
                   border-radius: 2px; font-size: 11px; letter-spacing: 0.03em; }
    .tag-critical { display:inline-block; padding: 1px 8px; border-radius: 2px; font-size: 11px;
                    letter-spacing: 0.04em; background: #c62828; color: #fff; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

_hdr_l, _hdr_r = st.columns([5, 1])
with _hdr_l:
    st.markdown(
        f"""
<div style="font-family:{SERIF};font-weight:600;font-size:26px;letter-spacing:0.42em;color:{INK};">VICUNA</div>
<div class="nl-muted" style="font-size:11.5px;margin-top:5px;">DB: {DB_PATH.name} · mtime {db_mtime():%Y-%m-%d %H:%M} · read-only</div>
""",
        unsafe_allow_html=True,
    )
with _hdr_r:
    if st.button("⟳ Refresh", use_container_width=True):
        alpaca_get.clear()
        db_query.clear()
        db_mtime.clear()
        st.rerun()


_HALT_FLAG = PROJ_ROOT / "data" / ".vicuna_halt"
_halted = _HALT_FLAG.exists()
_halted_since = _HALT_FLAG.read_text().strip() if _halted else ""

if _halted:
    st.markdown(
        f"""
<div style="background:#c62828;color:#fff;padding:10px 16px;margin:8px 0 2px;border-radius:2px;
font-size:13.5px;letter-spacing:0.02em;"><b>KILL SWITCH ENGAGED</b> since {_halted_since} —
laptop pipeline halted, mini instructed to veto all new entries. Protective legs stay on.
Resume below or via <code>scripts/kill_switch.sh --resume</code>.</div>
""",
        unsafe_allow_html=True,
    )

with st.expander("⛔ Kill switch" + ("  ·  ENGAGED" if _halted else ""), expanded=_halted):
    if not _halted:
        st.markdown(
            "Halts Vicuna autonomy on **both hosts**: stops laptop crons, kills running "
            "jobs, and writes the mini's `agent/.halt` — the execution agent vetoes every new entry "
            "until resumed. Open positions and their GTC protective legs are untouched. "
            "")
        _cancel_too = st.checkbox(
            "Also cancel ALL open orders — removes protective stop/target legs; "
            "positions become unprotected", value=False, key="ks_cancel")
        _phrase = "HALT AND CANCEL" if _cancel_too else "HALT"
        _typed = st.text_input(f"Type {_phrase} to engage", key="ks_confirm")
        if st.button("Engage kill switch", type="primary", disabled=(_typed != _phrase)):
            _args = ["--cancel-orders-yes"] if _cancel_too else []
            with st.spinner("Engaging — halting laptop + mini…"):
                _proc = subprocess.run(
                    ["bash", str(PROJ_ROOT / "scripts" / "kill_switch.sh"), *_args],
                    cwd=PROJ_ROOT, capture_output=True, text=True, timeout=90)
            _out = _proc.stdout + _proc.stderr
            st.code(_out, language=None)
            if "WARN: could not" in _out:
                st.error("Laptop halted, but the mini could not be reached — "
                         "halt the mini directly.")
            elif _proc.returncode == 0:
                st.rerun()
    else:
        _typed_r = st.text_input("Type RESUME to clear the halt on both hosts", key="ks_resume")
        if st.button("Resume Vicuna", disabled=(_typed_r != "RESUME")):
            with st.spinner("Resuming — clearing laptop + mini halt…"):
                _proc = subprocess.run(
                    ["bash", str(PROJ_ROOT / "scripts" / "kill_switch.sh"), "--resume"],
                    cwd=PROJ_ROOT, capture_output=True, text=True, timeout=90)
            st.code(_proc.stdout + _proc.stderr, language=None)
            st.rerun()


(tab_overview, tab_picks, tab_cases, tab_kronos, tab_models, tab_clusters,
 tab_shadow, tab_briefing, tab_policy) = st.tabs(
    ["Overview", "Picks", "Investment cases", "Kronos", "Models", "Clusters",
     "Shadow book", "Briefing", "Policy"])


acct = alpaca_get("/v2/account")
equity = cash = bp = day_pnl = 0.0
_acct_html: str | None = None
_acct_err = ""
if isinstance(acct, dict) and "error" not in acct:
    equity = float(acct.get("equity", 0))
    cash = float(acct.get("cash", 0))
    bp = float(acct.get("buying_power", 0))
    daytrade_count = int(acct.get("daytrade_count", 0))
    status = acct.get("status", "?")
    last_equity = float(acct.get("last_equity", equity))
    day_pnl = equity - last_equity
    day_pnl_pct = (day_pnl / last_equity * 100) if last_equity else 0.0

    day_col = LOSST if day_pnl < 0 else GAINT
    _v = (f"font-family:{SERIF};font-weight:600;font-size:31px;line-height:1.05;"
          f"margin-top:7px;font-variant-numeric:tabular-nums;letter-spacing:-0.02em;")
    _s = "font-size:11.5px;margin-top:4px;"
    _metrics = [
        ("Equity", money(equity), INK, pctf(day_pnl_pct) + " today", day_col),
        ("Cash", money(cash), INK, f"{cash / equity * 100:.0f}% of equity" if equity else "—", MUTED),
        ("Buying power", money(bp), INK, "cash account", MUTED),
        ("Day P&L", smoney(day_pnl), day_col, "settled + intraday", MUTED),
        ("Status", status.upper(), GAINT if status.lower() == "active" else LOSST,
         f"{daytrade_count} day trades", MUTED),
    ]
    _cells = "".join(
        f'<div style="padding:0 18px;border-left:1px solid {DIVIDER};">'
        f'<div class="nl-kicker-muted">{lbl}</div>'
        f'<div style="{_v}color:{vc};">{val}</div>'
        f'<div style="{_s}color:{sc};">{sub}</div></div>'
        for lbl, val, vc, sub, sc in _metrics
    )
    _acct_html = (
        f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:2px;'
        f'border-top:2px solid {INK};border-bottom:1px solid {DIVIDER};'
        f'padding:16px 0;margin:2px 0 4px;">{_cells}</div>')
else:
    _acct_err = f"Alpaca account fetch failed: {acct.get('error', acct)}"


def account_strip() -> None:
    ""
    if _acct_html is not None:
        st.markdown(_acct_html, unsafe_allow_html=True)
    else:
        st.error(_acct_err)


picks_df = db_query("""
    SELECT rank, symbol, sector, strategy_name,
           ROUND(deflated_sharpe, 3) AS dsr,
           ROUND(profit_factor, 2) AS pf,
           ROUND(max_dd, 3) AS max_dd,
           ROUND(suggested_size_pct * 100, 1) AS size_pct,
           rationale, as_of_date
    FROM mm_active_picks
    ORDER BY rank
""")


with tab_picks:
    account_strip()
    st.header("Picks")

    try:
        _fn = db_query(
            "SELECT seq, stage, cells, symbols, note, as_of_date FROM mm_pick_funnel "
            "WHERE as_of_date = (SELECT MAX(as_of_date) FROM mm_pick_funnel) ORDER BY seq")
        _ft = db_query(
            "SELECT symbol, strategy_name, sector, ROUND(deflated_sharpe, 3) AS dsr, "
            "fate, reason FROM mm_pick_fates "
            "WHERE as_of_date = (SELECT MAX(as_of_date) FROM mm_pick_fates) "
            "ORDER BY (fate = 'picked') DESC, deflated_sharpe DESC")
    except Exception:
        _fn, _ft = pd.DataFrame(), pd.DataFrame()
    if _fn.empty:
        st.info("The pick funnel has not been recorded yet — it appears after "
                "the next evening export.")
    else:
        _asof = str(_fn["as_of_date"].iloc[0])
        _shipped = int(_fn["cells"].iloc[-1])
        _tested = int(_fn["cells"].iloc[0])
        _stale_days = (date.today() - date.fromisoformat(_asof)).days
        st.subheader(
            f"Why {_shipped} pick{'s' if _shipped != 1 else ''} — export of {_asof}")
        _prev = _fn["cells"].shift(1)
        _cut = (1 - _fn["cells"] / _prev.where(_prev > 0)).fillna(0)
        _worst = _fn.loc[_cut.idxmax()]
        _before = int(_prev.loc[_cut.idxmax()])
        _counts = _ft["fate"].value_counts() if not _ft.empty else pd.Series(dtype=int)
        _stopped = ", ".join(
            f"{n} by {fate}" for fate, n in _counts.items() if fate != "picked")
        st.markdown(
            f"The system tested **{_tested:,}** strategy–stock pairs and shipped "
            f"**{_shipped}**. The largest single cut was **{_worst['stage']}**: "
            f"{_before:,} pairs reached it and {int(_worst['cells']):,} passed. "
            + (f"Of the pairs that cleared every track-record gate, {_stopped}. "
               if _stopped else "")
            + "A pair with a good record still needs its entry condition to occur "
              "on the day; no signal means no pick.")
        if _stale_days > 3:
            st.warning(f"This export is {_stale_days} days old — the evening chain "
                       "has not run since. Check the laptop crons.")
        _c1, _c2 = st.columns([5, 7])
        with _c1:
            st.caption("Gate by gate — pairs and distinct stocks left after each step")
            st.dataframe(
                _fn[["stage", "cells", "symbols", "note"]].rename(columns={
                    "stage": "Step", "cells": "Pairs left",
                    "symbols": "Stocks left", "note": "What it checks"}),
                use_container_width=True, hide_index=True,
                height=40 + 35 * len(_fn))
        with _c2:
            st.caption("Every pair that cleared the track-record gates, and what "
                       "happened to it")
            if _ft.empty:
                st.write("No pair cleared the track-record gates.")
            else:
                st.dataframe(
                    _ft.rename(columns={
                        "symbol": "Stock", "strategy_name": "Strategy",
                        "sector": "Sector", "dsr": "DSR", "fate": "Outcome",
                        "reason": "Why"}),
                    use_container_width=True, hide_index=True,
                    height=min(495, 40 + 35 * len(_ft)))
        _dec = db_query(
            "SELECT entry_date AS day, symbol, status, fill_status, "
            "actual_fill_price AS fill FROM mm_recs_ledger "
            "WHERE entry_date >= ? ORDER BY entry_date DESC, rank",
            ((date.today() - timedelta(days=7)).isoformat(),))
        st.caption("What the mini did with the last week of picks"
                   + (" — none were issued." if _dec.empty else ""))
        if not _dec.empty:
            st.dataframe(_dec, use_container_width=True, hide_index=True,
                         height=min(300, 40 + 35 * len(_dec)))
        st.divider()

    _rv_path = PROJ_ROOT / "exports" / "strategy_review.json"
    if _rv_path.exists():
        _rv = json.loads(_rv_path.read_text())
        _ov, _rnd = _rv["overall"], _rv["random_entry"]
        with st.expander(
                f"Strategy review — every signal since {_rv['since']}, "
                f"as of {_rv['as_of']}", expanded=False):
            st.markdown(
                f"**{_ov['signals']:,}** buy signals on tradable stocks, each scored "
                f"as the trade we actually place. Average result "
                f"**{_ov['avg_return_pct']:+.2f}%**, which is "
                f"**{_ov['vs_spy_pct']:+.2f}%** against SPY over the same days; "
                f"{_ov['stopped_pct']:.0f}% were stopped out and "
                f"{_ov['hit_target_pct']:.0f}% reached their target. Buying the same "
                f"stocks on random days gave {_rnd['avg_return_pct']:+.2f}% "
                f"({_rnd['vs_spy_pct']:+.2f}% against SPY) — the signals did not "
                f"beat chance. The sector a pick came from mattered more than the "
                f"strategy that picked it.")
            _rc1, _rc2 = st.columns([8, 4])
            with _rc1:
                st.caption("By strategy — worst first. 'Retired' strategies no "
                           "longer produce picks.")
                _rs = pd.DataFrame(_rv["strategies"]).sort_values(
                    "vs_spy_pct", na_position="last")
                st.dataframe(
                    _rs.rename(columns={
                        "id": "#", "strategy": "Strategy", "persona": "Persona",
                        "signals": "Signals", "avg_return_pct": "Avg %",
                        "vs_spy_pct": "vs SPY %", "win_rate_pct": "Win %",
                        "stopped_pct": "Stopped %", "hit_target_pct": "Target %",
                        "verdict": "Verdict"}),
                    use_container_width=True, hide_index=True, height=460)
            with _rc2:
                st.caption("By sector — random entries, same bracket")
                st.dataframe(
                    pd.DataFrame(_rv["by_sector_random_entry"]).sort_values(
                        "vs_spy_pct").rename(columns={
                            "sector": "Sector", "trades": "Trades",
                            "avg_return_pct": "Avg %", "vs_spy_pct": "vs SPY %"}),
                    use_container_width=True, hide_index=True)
            st.caption(_rv["method"])

    _today = date.today()
    _GO_LIVE = date(2026, 7, 6)
    _PRESETS = {
        "1 Day": (_today, _today),
        "1 Week": (_today - timedelta(weeks=1), _today),
        "1 Month": (_today - timedelta(days=30), _today),
        "3 Months": (_today - timedelta(days=91), _today),
        "6 Months": (_today - timedelta(days=182), _today),
        "1 Year": (_today - timedelta(days=365), _today),
        "All Time": (_GO_LIVE, _today),
    }
    if "picks_preset" not in st.session_state:
        st.session_state["picks_preset"] = "1 Month"
    preset_cols = st.columns(len(_PRESETS))
    for _col, _label in zip(preset_cols, _PRESETS):
        if _col.button(
            _label, key=f"picks_preset_{_label}",
            type="primary" if st.session_state["picks_preset"] == _label else "secondary",
        ):
            st.session_state["picks_preset"] = _label
            st.rerun()
    start_d, end_d = _PRESETS[st.session_state["picks_preset"]]
    start_s, end_s = start_d.isoformat(), end_d.isoformat()

    ledger_range = db_query("""
        SELECT entry_date, rank, symbol, sector, strategy_name,
               status, fill_status, actual_fill_price, fill_qty,
               ROUND(suggested_size_pct * 100, 1) AS size_pct,
               rationale
        FROM mm_recs_ledger
        WHERE entry_date BETWEEN ? AND ?
        ORDER BY entry_date DESC, rank
    """, (start_s, end_s))

    active_asof = str(picks_df["as_of_date"].iloc[0]) if not picks_df.empty else None

    if not ledger_range.empty:
        view = ledger_range
        if (not picks_df.empty
                and ledger_range["entry_date"].nunique() == 1
                and set(ledger_range["symbol"]) == set(picks_df["symbol"])):
            view = ledger_range.merge(
                picks_df[["symbol", "dsr", "pf", "max_dd"]],
                on="symbol", how="left",
            )
        st.caption(
            f"{start_s} → {end_s} · {len(view)} picks from mm_recs_ledger "
            f"(decision + fill state from the mini)"
        )
        st.dataframe(
            view, use_container_width=True, hide_index=True,
            height=min(600, 40 + 35 * len(view)),
        )
    elif not picks_df.empty and active_asof is not None and end_s >= active_asof:
        st.caption(
            f"picks as_of {active_asof} · {len(picks_df)} picks · "
            f"no ledger decisions in this range yet · DL + RL annotations in rationale"
        )
        st.dataframe(
            picks_df.drop(columns=["as_of_date"]),
            use_container_width=True,
            hide_index=True,
            height=min(400, 40 + 35 * len(picks_df)),
        )
    else:
        known_days = db_query(
            "SELECT DISTINCT entry_date FROM mm_recs_ledger ORDER BY entry_date DESC"
        )["entry_date"].tolist()
        st.info(
            f"No picks recorded between {start_s} and {end_s}. "
            + (f"Days with ledger history: {', '.join(known_days)}."
               if known_days else "Ledger history starts accumulating from go-live day.")
        )


with tab_overview:
    account_strip()

    try:
        df_spy = db_query(
            "SELECT date, vicuna_ret, spy_ret, excess FROM mm_vs_spy "
            "ORDER BY date DESC LIMIT 1")
    except Exception:
        df_spy = pd.DataFrame()
    if not df_spy.empty:
        _r = df_spy.iloc[0]
        _exc = float(_r["excess"])
        _exc_col = GAINT if _exc >= 0 else LOSST
        st.markdown(
            f"""
<div style="display:flex;align-items:baseline;gap:18px;padding:10px 15px;border-left:3px solid {ACCENT};background:#f4f7fa;border-radius:2px;margin:14px 0 10px;font-size:13px;">
<span style="font-weight:600;">vs SPY</span>
<span class="nl-muted">since 2026-09-03</span>
<span>Vicuna <b>{100*float(_r['vicuna_ret']):+.2f}%</b></span>
<span>SPY <b>{100*float(_r['spy_ret']):+.2f}%</b></span>
<span style="color:{_exc_col};font-weight:600;">excess {100*_exc:+.2f} pp</span>
<span class="nl-muted">as of {_r['date']}</span>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.caption("vs SPY scoreboard: epoch starts 2026-09-03 open — no data yet.")

    df_div = db_query(
        "SELECT snapshot_at, severity, kind, symbol, rec_id, detail "
        "FROM mm_broker_divergences "
        "ORDER BY CASE severity WHEN 'critical' THEN 0 "
        "WHEN 'warn' THEN 1 ELSE 2 END")
    if not df_div.empty:
        n_crit = int((df_div["severity"] == "critical").sum())
        n_warn = int((df_div["severity"] == "warn").sum())
        checked_at = df_div["snapshot_at"].iloc[0]
        st.markdown(
            f"""
<div style="display:flex;align-items:flex-start;gap:11px;padding:12px 15px;background:#fdecec;border-left:3px solid {RED};border-radius:2px;margin:14px 0 10px;">
<i class="ph-duotone ph-warning-diamond" style="font-size:19px;color:#a01c1c;flex:none;margin-top:1px;"></i>
<div style="font-size:12.5px;line-height:1.5;">
<span style="font-weight:600;">Broker-vs-ledger divergence</span>
<span class="tag-critical" style="margin:0 6px;">{n_crit} CRITICAL</span>
<span class="tag-neutral">{n_warn} WARN</span><br>
<span class="nl-muted">Mirror snapshot {checked_at} · triangulate positions + activities + portfolio history before acting — never force-sell on one source.</span>
</div></div>
""",
            unsafe_allow_html=True,
        )
        with st.expander("Divergence findings"):
            st.dataframe(df_div[["severity", "kind", "symbol", "detail"]],
                         use_container_width=True, hide_index=True)

    df_rest = db_query("""
        SELECT o.symbol, o.qty, o.limit_price, o.submitted_at, o.id AS order_id,
               l.rec_id, l.stop_price, l.target_price
        FROM mm_alpaca_orders o
        LEFT JOIN mm_recs_ledger l ON l.alpaca_order_id = o.id
        WHERE o.side = 'buy'
          AND o.status IN ('new', 'held', 'accepted', 'partially_filled', 'pending_new')
          AND (o.parent_id IS NULL OR o.parent_id = '')
          AND o.snapshot_at = (SELECT MAX(snapshot_at) FROM mm_alpaca_orders)
        ORDER BY o.submitted_at
    """)
    st.header("Resting entry orders")
    if df_rest.empty:
        st.caption("No unfilled entry orders at the broker (latest mirror snapshot).")
    else:
        from zoneinfo import ZoneInfo as _ZI
        _et = _ZI("America/New_York")
        _today_et = datetime.now(_et).date()
        _picks_now = db_query("SELECT symbol, rank FROM mm_active_picks")
        _rank_now = dict(zip(_picks_now["symbol"], _picks_now["rank"]))
        _syms = tuple(df_rest["symbol"].unique())
        _lc = db_query(
            "SELECT b.symbol, b.close FROM mm_bars b "
            "JOIN (SELECT symbol, MAX(date) AS d FROM mm_bars GROUP BY symbol) m "
            "  ON m.symbol = b.symbol AND m.d = b.date "
            f"WHERE b.symbol IN ({','.join('?' * len(_syms))})", _syms)
        _close_of = dict(zip(_lc["symbol"], _lc["close"]))

        def _placed_day(ts: str):
            return (datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    .astimezone(_et).date())

        view = pd.DataFrame({
            "symbol": df_rest["symbol"],
            "qty": df_rest["qty"].astype(int),
            "limit": df_rest["limit_price"],
            "stop": df_rest["stop_price"],
            "target": df_rest["target_price"],
            "placed (ET)": [_placed_day(t) for t in df_rest["submitted_at"]],
        })
        view["sessions resting"] = [
            int(np.busday_count(d, _today_et)) for d in view["placed (ET)"]]
        view["last close"] = [_close_of.get(s) for s in view["symbol"]]
        view["limit vs close"] = [
            f"{(lim / c - 1) * 100:+.1f}%" if (c and lim) else ""
            for lim, c in zip(view["limit"], view["last close"])]
        view["in picks now"] = [
            f"rank {int(_rank_now[s])}" if s in _rank_now else "dropped"
            for s in view["symbol"]]
        _is_op = [((r or "").startswith("OPERATOR_") or "-op-" in (r or ""))
                  for r in df_rest["rec_id"]]
        view["source"] = ["operator" if o else "pipeline" for o in _is_op]
        view["state"] = [
            "exempt" if o else ("STALE — sweep should have canceled" if n >= 1 else "today")
            for o, n in zip(_is_op, view["sessions resting"])]
        n_stale = int(sum(1 for st_ in view["state"] if st_.startswith("STALE")))
        notional = float((view["qty"] * view["limit"]).sum())
        st.caption(
            f"{len(view)} resting entry parents · ${notional:,.0f} reserved at the limits · "
            f"{n_stale} past their session (v1.6: unfilled pipeline entries are canceled at "
            f"16:05 ET and re-priced next morning from fresh data; manually placed orders "
            f"are exempt). Mirror snapshot "
            f"{str(db_query('SELECT MAX(snapshot_at) AS s FROM mm_alpaca_orders')['s'].iloc[0])[:16]}Z."
        )
        st.dataframe(view, use_container_width=True, hide_index=True,
                     height=min(520, 40 + 35 * len(view)))


    positions = alpaca_get("/v2/positions")
    pos_list = positions if isinstance(positions, list) else []
    if not isinstance(positions, list):
        st.warning(f"positions fetch: {positions}")

    sector_map = db_query(
        "SELECT symbol, COALESCE(sector, 'Other') AS sector FROM mm_sector_map")
    _sector_of = dict(zip(sector_map["symbol"], sector_map["sector"]))
    pos_df = pd.DataFrame([{
        "symbol": p["symbol"],
        "sector": _sector_of.get(p["symbol"], "Other"),
        "qty": float(p["qty"]),
        "entry": float(p["avg_entry_price"]),
        "current": float(p["current_price"]),
        "mv": float(p["market_value"]),
        "upl": float(p["unrealized_pl"]),
        "uplpc": float(p["unrealized_plpc"]) * 100,
    } for p in pos_list])

    invested = float(pos_df["mv"].sum()) if not pos_df.empty else 0.0
    cash_val = equity - invested
    tot_upl = float(pos_df["upl"].sum()) if not pos_df.empty else 0.0
    upl_col = LOSST if tot_upl < 0 else GAINT

    sector_sel = st.session_state.get("pnl_sector", "All")
    if not pos_df.empty and sector_sel != "All" and sector_sel not in set(pos_df["sector"]):
        sector_sel = "All"
    fpos = pos_df if sector_sel == "All" else pos_df[pos_df["sector"] == sector_sel]

    def _empty_note() -> None:
        st.markdown(
            f"""
<div class="nl-empty"><i class="ph-duotone ph-wallet" style="font-size:22px;color:{ACCENT};"></i>
<span>No open positions — the account is 100% cash. Allocation and per-position views resume when a pick fills.</span></div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
<div style="display:flex;align-items:baseline;gap:14px;margin-top:14px;">
<h2 style="margin:0;">P&amp;L &amp; allocation</h2>
<span class="nl-muted" style="font-size:12.5px;">history · per-position contribution · allocation</span>
</div>
<div style="display:flex;flex-wrap:wrap;align-items:center;gap:22px 30px;margin:18px 0 8px;padding:14px 0;border-top:1px solid {DIVIDER};border-bottom:1px solid {DIVIDER};font-size:12.5px;color:{MUTED};">
<span><span style="font-family:{SERIF};font-weight:600;color:{INK};font-size:14px;">{len(pos_df)}</span> open
· invested <span style="font-family:{SERIF};font-weight:600;color:{INK};">{money(invested)}</span>
({invested / equity * 100 if equity else 0:.1f}% of equity)
· <span style="font-family:{SERIF};font-weight:600;color:{upl_col};">{smoney(tot_upl)}</span> unrealized</span>
</div>
""",
        unsafe_allow_html=True,
    )

    h_head, h_ctl = st.columns([2, 3.4])
    with h_head:
        st.markdown(
            '<div class="nl-kicker">01 · History</div>'
            '<div class="nl-h4">Equity &amp; daily P&amp;L '
            '<span class="nl-muted" style="font-weight:400;font-size:14px;">· base $100,000</span></div>',
            unsafe_allow_html=True)
    with h_ctl:
        c_per, c_view, c_pre = st.columns([1.7, 2.7, 1.9])
        period = c_per.segmented_control(
            "Period", ["1W", "1M", "All"], default="1M",
            key="hist_period", label_visibility="collapsed") or "1M"
        hist_view = c_view.segmented_control(
            "View", ["Combo", "Cumulative", "Daily"], default="Combo",
            key="hist_view", label_visibility="collapsed") or "Combo"
        drop_pre = c_pre.checkbox("Drop pre-funding zeros", value=True, key="drop_pre")

    hist = alpaca_get("/v2/account/portfolio/history?period=all&timeframe=1D")
    if not (isinstance(hist, dict) and hist.get("timestamp")):
        hist = alpaca_get("/v2/account/portfolio/history?period=3M&timeframe=1D")

    if isinstance(hist, dict) and hist.get("timestamp"):
        hdf = pd.DataFrame({
            "ts": pd.to_datetime(hist["timestamp"], unit="s"),
            "eq": [float(v or 0) for v in hist["equity"]],
            "pnl": [float(v or 0) for v in hist["profit_loss"]],
        })
        base_val = float(hist.get("base_value") or 100000)
        hdf["pre"] = hdf["eq"] == 0
        hdf["label"] = hdf["ts"].dt.strftime("%b %-d")

        rows = hdf[~hdf["pre"]] if drop_pre else hdf
        rows = rows.tail({"1W": 5, "1M": 21}.get(period, len(rows)))
        cum = [None if p else e - base_val
               for e, p in zip(rows["eq"], rows["pre"])]

        fig = go.Figure()
        if hist_view != "Cumulative":
            fig.add_trace(go.Bar(
                x=rows["label"], y=rows["pnl"], name="Daily P&L",
                marker_color=[LOSS if v < 0 else GAIN for v in rows["pnl"]],
                yaxis="y",
                hovertemplate="%{x}<br>Daily %{y:$,.2f}<extra></extra>"))
        if hist_view != "Daily":
            fig.add_trace(go.Scatter(
                x=rows["label"], y=cum, name="Cumulative",
                mode="lines+markers", connectgaps=False,
                line=dict(color=ACCENT, width=2.5),
                marker=dict(size=5, color=ACCENT),
                yaxis="y2" if hist_view == "Combo" else "y",
                hovertemplate="%{x}<br>Cumulative %{y:$,.2f}<extra></extra>"))
        lay = nl_layout(height=360, showlegend=False, bargap=0.45,
                        xaxis=nl_axis(type="category"))
        if hist_view == "Combo":
            lay["yaxis"] = nl_axis(title=dict(text="daily $", font=dict(size=11)),
                                   tickformat="$,.0f")
            lay["yaxis2"] = nl_axis(overlaying="y", side="right",
                                    title=dict(text="cumulative $", font=dict(size=11)),
                                    tickformat="$,.0f", showgrid=False,
                                    gridcolor="rgba(0,0,0,0)")
        else:
            lay["yaxis"] = nl_axis(
                tickformat="$,.0f",
                title=dict(text="cumulative $" if hist_view == "Cumulative" else "daily $",
                           font=dict(size=11)))
        fig.update_layout(**lay)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
        st.caption(
            "Each bar is one trading day: green bars are days the account "
            "made money and red bars are days it lost money, in dollars. "
            "The blue line adds those days up into a running total of "
            "profit or loss versus the $100,000 starting balance (bars read "
            "on the left axis, the line on the right). A rising blue line "
            "means the account is gaining overall; a cluster of red bars "
            "means a losing streak.")
    else:
        st.info("Alpaca portfolio_history unavailable — usually recovers once account has activity.")

    col_upl, col_pct = st.columns(2, gap="large")
    with col_upl:
        st.markdown('<div class="nl-kicker" style="margin-top:24px;">02 · Contribution</div>'
                    '<div class="nl-h4">Unrealized P&amp;L by position</div>',
                    unsafe_allow_html=True)
        if fpos.empty:
            _empty_note()
        else:
            ps = fpos.sort_values("upl")
            mn, mx = min(0.0, ps["upl"].min()), max(0.0, ps["upl"].max())
            sp = max(80.0, mx - mn)
            fig = go.Figure(go.Bar(
                x=ps["upl"], y=ps["symbol"], orientation="h",
                marker_color=[LOSS if v < 0 else GAIN for v in ps["upl"]],
                text=[f"{smoney(u)} ({pctf(p)})" for u, p in zip(ps["upl"], ps["uplpc"])],
                textposition="outside", cliponaxis=False, textfont=dict(size=12),
                hovertemplate="%{y}: %{x:$,.2f}<extra></extra>"))
            fig.update_layout(**nl_layout(
                height=340, bargap=0.4, margin=dict(l=58, r=70, t=6, b=34),
                xaxis=nl_axis(tickformat="$,.0f",
                              range=[mn - sp * 0.78, mx + sp * 0.42]),
                yaxis=nl_axis()))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
            st.caption(
                "Each bar is one open position and its length is the "
                "dollars gained or lost since we bought it. Green bars "
                "point right for gains, red bars point left for losses, "
                "with the exact amount printed at the end. The longest "
                "bars matter most — a long red bar is the position hurting "
                "the account the most right now.")
    with col_pct:
        st.markdown('<div class="nl-kicker" style="margin-top:24px;">03 · Return</div>'
                    '<div class="nl-h4">Return % ranking</div>',
                    unsafe_allow_html=True)
        if fpos.empty:
            _empty_note()
        else:
            ps = fpos.sort_values("uplpc")
            mn, mx = min(0.0, ps["uplpc"].min()), max(0.0, ps["uplpc"].max())
            sp = max(2.0, mx - mn)
            fig = go.Figure(go.Bar(
                x=ps["uplpc"], y=ps["symbol"], orientation="h",
                marker_color=[LOSS if v < 0 else GAIN for v in ps["uplpc"]],
                text=[pctf(p) for p in ps["uplpc"]],
                textposition="outside", cliponaxis=False, textfont=dict(size=12),
                hovertemplate="%{y}: %{x:.2f}%<extra></extra>"))
            fig.update_layout(**nl_layout(
                height=340, bargap=0.4, margin=dict(l=58, r=52, t=6, b=34),
                xaxis=nl_axis(ticksuffix="%",
                              range=[mn - sp * 0.6, mx + sp * 0.42]),
                yaxis=nl_axis()))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
            st.caption(
                "Same positions as the chart on the left, but bar length "
                "here is the percent change from our purchase price, so "
                "big and small positions compare fairly. Green means the "
                "price is above what we paid; red means below. A position "
                "can show a small dollar loss but a long red bar here — it "
                "fell a lot, we just own little of it.")

    a_head, a_ctl = st.columns([2, 2])
    with a_head:
        st.markdown('<div class="nl-kicker" style="margin-top:24px;">04 · Allocation</div>'
                    '<div class="nl-h4">Where the capital sits '
                    '<span class="nl-muted" style="font-weight:400;font-size:14px;">'
                    '· tile size = market value, color = unrealized P&amp;L</span></div>',
                    unsafe_allow_html=True)
    with a_ctl:
        c_av, c_cash = st.columns([3, 2])
        alloc_view = c_av.segmented_control(
            "Allocation view", ["Treemap", "Donut", "Bars"], default="Treemap",
            key="alloc_view", label_visibility="collapsed") or "Treemap"
        cash_pct = cash_val / equity * 100 if equity else 0
        show_cash = c_cash.checkbox(f"Show cash ({cash_pct:.0f}%)",
                                    value=False, key="show_cash")
    show_cash = show_cash and sector_sel == "All"

    if fpos.empty:
        _empty_note()
    else:
        max_abs = max(1.0, fpos["upl"].abs().max())
        if alloc_view == "Treemap":
            labels, parents, values, colors, texts = [], [], [], [], []
            for sec, g in fpos.groupby("sector"):
                labels.append(sec); parents.append(""); values.append(float(g["mv"].sum()))
                colors.append(NEU3); texts.append("")
            for p in fpos.itertuples():
                labels.append(p.symbol); parents.append(p.sector); values.append(p.mv)
                colors.append(div_color(p.upl, max_abs)); texts.append(smoney(p.upl))
            if show_cash:
                labels.append("Cash"); parents.append(""); values.append(cash_val)
                colors.append(CASH_FILL); texts.append("uninvested")
            fig = go.Figure(go.Treemap(
                labels=labels, parents=parents, values=values, text=texts,
                branchvalues="total",
                marker=dict(colors=colors, line=dict(width=2, color=INK)),
                textfont=dict(family=SERIF, size=14, color=INK),
                texttemplate="<b>%{label}</b><br>%{value:$,.0f}<br>%{text}",
                hovertemplate="%{label}<br>%{value:$,.0f}<extra></extra>",
                tiling=dict(pad=3), pathbar=dict(visible=False)))
            fig.update_layout(**nl_layout(height=400, margin=dict(l=2, r=2, t=2, b=2)))
        elif alloc_view == "Donut":
            labels = list(fpos["symbol"])
            values = list(fpos["mv"])
            colors = [div_color(u, max_abs) for u in fpos["upl"]]
            if show_cash:
                labels.append("Cash"); values.append(cash_val); colors.append(CASH_FILL)
            fig = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.58, sort=False,
                direction="clockwise",
                marker=dict(colors=colors, line=dict(width=2, color=INK)),
                textfont=dict(family=SERIF, size=12, color=INK),
                texttemplate="%{label}<br>%{percent}",
                hovertemplate="%{label}<br>%{value:$,.0f} · %{percent}<extra></extra>"))
            fig.update_layout(**nl_layout(height=400, showlegend=False,
                                          margin=dict(l=10, r=10, t=10, b=10)))
        else:
            agg = (fpos.groupby("sector").agg(mv=("mv", "sum"), upl=("upl", "sum"))
                   .reset_index().sort_values("mv"))
            rows_a = agg.to_dict("records")
            if show_cash:
                rows_a.insert(0, {"sector": "Cash", "mv": cash_val, "upl": 0.0,
                                  "cash": True})
            sec_max = max(1.0, max(abs(r["upl"]) for r in rows_a))
            fig = go.Figure(go.Bar(
                x=[r["mv"] for r in rows_a], y=[r["sector"] for r in rows_a],
                orientation="h",
                marker_color=[CASH_FILL if r.get("cash")
                              else div_color(r["upl"], sec_max) for r in rows_a],
                text=[money(r["mv"]) for r in rows_a],
                textposition="outside", cliponaxis=False, textfont=dict(size=12),
                hovertemplate="%{y}: %{x:$,.0f}<extra></extra>"))
            fig.update_layout(**nl_layout(
                height=400, bargap=0.4, margin=dict(l=78, r=64, t=6, b=34),
                xaxis=nl_axis(tickformat="$,.0f"), yaxis=nl_axis()))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
        st.caption(
            "This shows where the account's money sits: each tile, slice, "
            "or bar is one holding (or cash) and its size is the dollars "
            "in it. Color shows how that holding is doing — green tones "
            "for gains, red tones for losses, pale grey for uninvested "
            "cash. One very large tile means the account is concentrated "
            "in a single bet; many small tiles mean the money is spread "
            "out.")

    d_head, d_ctl = st.columns([3, 1])
    with d_head:
        st.markdown('<div class="nl-kicker" style="margin-top:24px;">05 · Detail</div>'
                    '<div class="nl-h4">Open positions</div>', unsafe_allow_html=True)
    with d_ctl:
        st.selectbox(
            "Sector", ["All"] + sorted(pos_df["sector"].unique()) if not pos_df.empty else ["All"],
            key="pnl_sector",
            help="Filters charts 02–04 and this table.")

    if fpos.empty:
        _empty_note()
    else:
        tbl = (fpos.sort_values("mv", ascending=False)
               .rename(columns={"symbol": "Symbol", "sector": "Sector",
                                "qty": "Qty", "entry": "Entry",
                                "current": "Current", "mv": "Mkt value",
                                "upl": "Unreal P&L", "uplpc": "Unreal %"})
               .reset_index(drop=True))

        def _sign_style(v: float) -> str:
            return f"color:{LOSST if v < 0 else GAINT};font-weight:600;"

        sty = (tbl.style
               .format({"Qty": "{:,.0f}", "Entry": "${:,.2f}",
                        "Current": "${:,.2f}", "Mkt value": "${:,.2f}",
                        "Unreal P&L": lambda v: smoney(v, 2),
                        "Unreal %": lambda v: pctf(v)})
               .map(_sign_style, subset=["Unreal P&L", "Unreal %"]))
        st.dataframe(sty, use_container_width=True, hide_index=True)

    st.header("Recent ledger transitions")

    ledger_df = db_query("""
        SELECT as_of_date, symbol, strategy_name, status,
               decided_at, alpaca_order_id, fill_status,
               actual_fill_price, fill_qty,
               entry_est_price, stop_price, target_price
        FROM mm_recs_ledger
        ORDER BY COALESCE(updated_at, created_at) DESC
        LIMIT 30
    """)
    if ledger_df.empty:
        st.info(
            "mm_recs_ledger is empty — fills up once the mini logs decisions "
            "and the ledger sync runs (20:30 cron or manual)."
        )
    else:
        st.dataframe(ledger_df, use_container_width=True, hide_index=True)

    st.header("Recommendation scoreboard")

    scb_df = db_query("""
        SELECT as_of_date, window, decision_bucket,
               n_recs, n_closed,
               ROUND(win_rate, 3) AS win_rate,
               ROUND(avg_return_pct * 100, 2) AS avg_return_pct,
               ROUND(median_return_pct * 100, 2) AS median_return_pct,
               ROUND(total_return_pct * 100, 2) AS total_return_pct
        FROM mm_recommendation_scoreboard
        ORDER BY as_of_date DESC, window, decision_bucket
        LIMIT 30
    """)
    if scb_df.empty:
        st.info("mm_recommendation_scoreboard empty — no decisions attributed yet.")
    else:
        latest_asof = scb_df["as_of_date"].iloc[0]
        scb_latest = scb_df[scb_df["as_of_date"] == latest_asof].copy()
        st.caption(f"latest as_of {latest_asof}")

        cL, cR = st.columns([1, 1])
        with cL:
            scb_show = scb_latest.copy()
            scb_show["win_rate"] = scb_show["win_rate"].map(
                lambda v: "—" if pd.isna(v) else f"{v:.0%}")
            st.dataframe(scb_show, use_container_width=True, hide_index=True)
        with cR:
            if not scb_latest["win_rate"].notna().any():
                st.info(
                    "No closed recommendations yet — no win-rate history. "
                    "Decisions close after their 20-trading-day bracket window; "
                    "this fills in once the first ones resolve."
                )
            else:
                scb_latest["group"] = scb_latest["window"] + " · " + scb_latest["decision_bucket"]

                def _wr_color(v):
                    if v is None or pd.isna(v):
                        return NEU3
                    return GAIN if v >= 0.5 else LOSS

                def _wr_text(v):
                    return "—" if (v is None or pd.isna(v)) else f"{v:.0%}"

                wr = scb_latest["win_rate"]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=scb_latest["group"], y=wr,
                    name="win rate",
                    marker_color=[_wr_color(v) for v in wr],
                    text=[_wr_text(v) for v in wr],
                    textposition="outside",
                ))
                fig.update_layout(**nl_layout(
                    height=280, margin=dict(l=8, r=8, t=32, b=8),
                    title="Win rate by window · bucket",
                    yaxis=nl_axis(range=[0, 1.05], tickformat=".0%"),
                ))
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
                st.caption(
                    "Each bar is a group of past recommendations, split by "
                    "look-back window and by whether they were approved "
                    "or rejected them. Bar height is the share of picks in "
                    "that group that ended profitable — the top of the "
                    "chart means every pick won. Green bars (half or "
                    "better) were right more often than wrong; red bars "
                    "mean most picks in that group lost.")

    perf_df = db_query("""
        SELECT symbol, strategy_name, decision, exit_reason, days_held,
               entry_date, exit_date,
               return_pct_counterfactual, return_pct_actual,
               slippage_pct
        FROM mm_recommendation_performance
        WHERE status = 'closed' AND exit_date IS NOT NULL
          AND (return_pct_actual IS NOT NULL
               OR return_pct_counterfactual IS NOT NULL)
        ORDER BY exit_date
    """)
    if not perf_df.empty:
        st.subheader("Every closed recommendation")

        approved = perf_df["decision"] == "approve"
        perf_df["ret"] = perf_df["return_pct_counterfactual"].where(
            ~(approved & perf_df["return_pct_actual"].notna()),
            perf_df["return_pct_actual"]) * 100
        perf_df["bucket"] = "rejected/expired"
        perf_df.loc[approved, "bucket"] = "approved"
        perf_df["exit_date"] = pd.to_datetime(perf_df["exit_date"])

        RET_BINS = [
            (-10.0, "loss ≤ −10%", LOSST),
            (-2.0, "−10% … −2%", "#e0857d"),
            (2.0, "−2% … +2% (flat)", "#8a949c"),
            (10.0, "+2% … +10%", "#7fc39d"),
            (float("inf"), "gain > +10%", GAINT),
        ]

        def _bin_color(v: float) -> str:
            for hi, _, color in RET_BINS:
                if v <= hi:
                    return color
            return RET_BINS[-1][2]

        spy_df = db_query(
            "SELECT date, close FROM mm_bars WHERE symbol='SPY' "
            "AND date >= ? ORDER BY date",
            (str(perf_df["entry_date"].min()),))
        spy = None
        if not spy_df.empty:
            spy = (spy_df.assign(date=pd.to_datetime(spy_df["date"]))
                   .set_index("date")["close"]
                   .reindex(pd.date_range(spy_df["date"].min(),
                                          spy_df["date"].max()))
                   .ffill())

            def _spy_at(d) -> float:
                return float(spy.asof(pd.Timestamp(d)))

            perf_df["spy_ret"] = [
                (_spy_at(x) / _spy_at(e) - 1) * 100
                for e, x in zip(perf_df["entry_date"], perf_df["exit_date"])]
            perf_df["excess"] = perf_df["ret"] - perf_df["spy_ret"]
        else:
            perf_df["spy_ret"] = float("nan")
            perf_df["excess"] = float("nan")

        label_mask = perf_df["ret"].abs() >= 12

        fig = go.Figure()
        if spy is not None:
            spy_cum = (spy / spy.iloc[0] - 1) * 100
            fig.add_trace(go.Scatter(
                x=spy_cum.index, y=spy_cum.values,
                mode="lines", name="SPY cumulative",
                legendgroup="bench", legendgrouptitle_text="benchmark",
                line=dict(color=ACCENT, width=2),
                text=[pctf(v, 2) for v in spy_cum.values],
                hovertemplate="SPY since %s: %%{text}<extra></extra>"
                              % spy_cum.index[0].strftime("%b %d"),
            ))
        for bucket, filled in (("approved", True), ("rejected/expired", False)):
            sub = perf_df[perf_df["bucket"] == bucket]
            if sub.empty:
                continue
            labels = sub["symbol"].where(label_mask[sub.index], "")
            colors = [_bin_color(v) for v in sub["ret"]]
            marker = (dict(symbol="circle", size=11, color=colors,
                           line=dict(color="#ffffff", width=2)) if filled
                      else dict(symbol="circle-open", size=11, color=colors,
                                line=dict(color=colors, width=2)))
            fig.add_trace(go.Scatter(
                x=sub["exit_date"], y=sub["ret"],
                mode="markers+text", name=bucket, marker=marker,
                showlegend=False,
                text=labels, textposition="top center",
                textfont=dict(family=SERIF, size=11, color=INK),
                customdata=list(zip(
                    sub["symbol"], sub["strategy_name"], sub["exit_reason"],
                    sub["days_held"], sub["entry_date"],
                    ["—" if pd.isna(v) else pctf(v, 2) for v in sub["spy_ret"]],
                    ["—" if pd.isna(v) else pctf(v, 2) for v in sub["excess"]],
                    [pctf(v, 2) for v in sub["ret"]])),
                hovertemplate=(
                    "<b>%{customdata[0]}</b> — " + bucket +
                    "<br>%{customdata[1]}"
                    "<br>return %{customdata[7]} · SPY same window "
                    "%{customdata[5]} · excess %{customdata[6]}"
                    "<br>%{customdata[4]} → %{x|%b %d} · "
                    "%{customdata[3]:.0f}d · exit: %{customdata[2]}"
                    "<extra></extra>"),
            ))
        for _, blabel, bcolor in RET_BINS:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=blabel,
                legendgroup="bins", legendgrouptitle_text="return bin",
                marker=dict(symbol="circle", size=11, color=bcolor,
                            line=dict(color="#ffffff", width=2))))
        for shape_name, shape_marker in (
                ("approved (filled)",
                 dict(symbol="circle", size=11, color="#8a949c",
                      line=dict(color="#ffffff", width=2))),
                ("rejected/expired (open)",
                 dict(symbol="circle-open", size=11, color="#8a949c",
                      line=dict(color="#8a949c", width=2)))):
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=shape_name,
                legendgroup="shape", legendgrouptitle_text="decision",
                marker=shape_marker))
        fig.update_layout(**nl_layout(
            height=430, margin=dict(l=8, r=8, t=8, b=96),
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.14,
                        xanchor="center", x=0.5,
                        groupclick="toggleitem"),
            xaxis=nl_axis(tickformat="%b %d"),
            yaxis=nl_axis(ticksuffix="%", zeroline=True),
        ))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
        st.caption(
            "One dot per closed recommendation, placed on the day it "
            "resolved — height is what that single pick returned, and the "
            "dot's color bins that return: deep red for losses beyond "
            "−10%, pale red for milder losses, gray for roughly flat, "
            "pale green for modest gains, deep green beyond +10%. Filled "
            "dots are picks that were approved (real fills, real "
            "exits); open rings are picks passed on or expired, scored as "
            "if they had been taken (counterfactual). The gap between the "
            "two populations is what the approval filter added or cost — "
            "the same split the win-rate bars above aggregate. The blue "
            "line is the S&P 500 (SPY) running total over the whole "
            "stretch — the market backdrop these picks traded in. A fair "
            "per-pick comparison needs matching time spans, so each "
            "dot's hover shows what SPY did over that pick's own holding "
            "window and the pick's excess over it. Hover any dot for the "
            "strategy, holding period, and how the trade ended.")

    st.header("Symbol drill-down")

    default_syms = picks_df["symbol"].tolist() if not picks_df.empty else []
    all_bars_syms = db_query(
        "SELECT DISTINCT symbol FROM mm_bars ORDER BY symbol"
    )["symbol"].tolist()

    default_sym = default_syms[0] if default_syms else (all_bars_syms[0] if all_bars_syms else None)

    if default_sym:
        sym = sector_symbol_picker(all_bars_syms, _sector_of, prefix="drill",
                                   default_sym=default_sym, dot=False)

        bars = db_query(
            """SELECT date, open, high, low, close, volume
               FROM mm_bars WHERE symbol = ?
               ORDER BY date DESC LIMIT 180""",
            params=(sym,),
        ).sort_values("date")

        if bars.empty:
            st.warning(f"No bars for {sym}.")
        else:
            pick_row = picks_df[picks_df["symbol"] == sym]

            dl_df = db_query(
                "SELECT date, predicted_5d_return FROM mm_dl_signals "
                "WHERE symbol = ? ORDER BY date DESC LIMIT 30",
                params=(sym,),
            )
            rl_df = db_query(
                "SELECT date, action, suggested_size_pct FROM mm_rl_signals "
                "WHERE symbol = ? ORDER BY date DESC LIMIT 30",
                params=(sym,),
            )

            fig = go.Figure(data=[go.Candlestick(
                x=bars["date"],
                open=bars["open"], high=bars["high"],
                low=bars["low"], close=bars["close"],
                name=sym,
            )])

            overlays: list[tuple[float, str, str]] = []
            hi, lo = float(bars["high"].max()), float(bars["low"].min())
            overlays.append((hi, f"high {hi:,.2f}", "#90a4ae"))
            overlays.append((lo, f"low {lo:,.2f}", "#90a4ae"))

            entry = stop = target = None
            led = db_query(
                "SELECT entry_est_price, actual_fill_price, stop_price, target_price "
                "FROM mm_recs_ledger WHERE symbol = ? "
                "ORDER BY COALESCE(updated_at, created_at) DESC LIMIT 1",
                params=(sym,),
            )
            if not led.empty and pd.notna(led["stop_price"].iloc[0]):
                r = led.iloc[0]
                entry = r["actual_fill_price"] if pd.notna(r["actual_fill_price"]) else r["entry_est_price"]
                stop, target = r["stop_price"], r["target_price"]
            elif not pick_row.empty:
                b = db_query(
                    "SELECT suggested_stop_loss_pct, suggested_target_pct "
                    "FROM mm_active_picks WHERE symbol = ?", params=(sym,))
                if not b.empty:
                    entry = float(bars["close"].iloc[-1])
                    stop = entry * (1 - float(b["suggested_stop_loss_pct"].iloc[0]))
                    target = entry * (1 + float(b["suggested_target_pct"].iloc[0]))

            if entry is not None and pd.notna(entry):
                overlays.append((float(entry), f"entry {float(entry):,.2f}", ACCENT))
            if target is not None and pd.notna(target):
                overlays.append((float(target), f"target {float(target):,.2f}", GAINT))
            if stop is not None and pd.notna(stop):
                overlays.append((float(stop), f"stop {float(stop):,.2f}", LOSST))

            for price, label, color in overlays:
                dash = "dot" if color == "#90a4ae" else "dash"
                fig.add_hline(y=price, line_dash=dash, line_width=1.4, line_color=color)
                fig.add_annotation(
                    xref="paper", x=1.004, xanchor="left",
                    yref="y", y=price,
                    text=f" {label} ", showarrow=False,
                    font=dict(size=12, color=color),
                    bgcolor="rgba(255,255,255,0.95)",
                    bordercolor=color, borderwidth=1, borderpad=2,
                )

            span = [hi, lo] + [p for p, _, _ in overlays]
            ymin, ymax = min(span), max(span)
            pad = (ymax - ymin) * 0.05 or 1.0
            fig.update_traces(increasing_line_color=GAIN, decreasing_line_color=LOSS,
                              selector=dict(type="candlestick"))
            fig.update_layout(**nl_layout(
                height=520, xaxis_rangeslider_visible=False,
                margin=dict(l=8, r=96, t=8, b=8),
                showlegend=False,
                xaxis=nl_axis(), yaxis=nl_axis(range=[ymin - pad, ymax + pad]),
            ))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
            st.caption(
                "Each candle is one trading day: green candles closed "
                "higher than they opened, red ones lower, and the thin "
                "wicks show the day's full high-low range. The dashed "
                "lines mark this symbol's trade levels — blue entry, green "
                "profit target, red stop loss — with the period's high and "
                "low in grey. While the candles sit between the red and "
                "green lines the trade is undecided; touching green takes "
                "profit, touching red sells to cap the loss.")

            st.subheader("Growth")
            gr = db_query(
                "SELECT latest_period_end, n_quarters,"
                "       rev_chg_mrq_yoy, rev_chg_ttm, rev_growth_5y_avg,"
                "       eps_chg_mrq_yoy, eps_chg_ttm, eps_growth_5y_avg"
                "  FROM mm_fundamentals_growth WHERE symbol = ?"
                "  ORDER BY as_of_date DESC LIMIT 1",
                params=(sym,),
            )
            if gr.empty:
                st.caption("no fundamentals history for this symbol")
            else:
                g = gr.iloc[0]
                gcols = ["MRQ vs 1 yr ago", "TTM over TTM", "5-yr avg"]
                gt = pd.DataFrame({
                    "": ["Revenue", "EPS"],
                    gcols[0]: [g["rev_chg_mrq_yoy"], g["eps_chg_mrq_yoy"]],
                    gcols[1]: [g["rev_chg_ttm"], g["eps_chg_ttm"]],
                    gcols[2]: [g["rev_growth_5y_avg"], g["eps_growth_5y_avg"]],
                })

                def _g_style(v) -> str:
                    if pd.isna(v):
                        return "color:#90a4ae;"
                    return f"color:{LOSST if v < 0 else GAINT};font-weight:600;"

                gsty = (gt.style
                        .format(lambda v: "—" if pd.isna(v) else pctf(v), subset=gcols)
                        .map(_g_style, subset=gcols))
                st.dataframe(gsty, use_container_width=True, hide_index=True)
                st.caption(
                    f"latest filing {str(g['latest_period_end'])[:10]} · "
                    f"{int(g['n_quarters'])} quarterly filings on record · "
                    "— = not meaningful (negative/zero base, filing gap, or split in window)"
                )

            cA, cB = st.columns(2)
            with cA:
                st.subheader("DL signal history")
                if dl_df.empty:
                    st.caption("no DL predictions in window")
                else:
                    st.dataframe(dl_df, use_container_width=True, hide_index=True, height=250)
            with cB:
                st.subheader("RL signal history")
                if rl_df.empty:
                    st.caption("no RL predictions in window")
                else:
                    rl_df = rl_df.copy()
                    rl_df["action_label"] = rl_df["action"].map(
                        {0: "HOLD", 1: "BUY", 2: "SELL"}
                    ).fillna(rl_df["action"].astype(str))
                    st.dataframe(rl_df, use_container_width=True, hide_index=True, height=250)
    else:
        st.info("No symbols with bars found — populate mm_bars first.")


with tab_cases:
    account_strip()
    st.header("Investment cases")

    cases_df, case_extras_df = load_investment_cases()
    if cases_df.empty:
        st.info(
            "No case files at data/investment_cases/ yet. They rsync from the "
            "mini's ~/agent/investment-cases/ at 11:35 and 20:30 on weekdays."
        )
    else:
        case_days = sorted(cases_df["as_of_date"].unique(), reverse=True)
        col_day, col_m1, col_m2, col_m3 = st.columns([2, 1, 1, 1])
        sel_day = col_day.selectbox("Session", case_days, index=0, key="cases_day")
        day_cases = cases_df[cases_df["as_of_date"] == sel_day].sort_values("pick_rank")
        col_m1.metric("Cases", len(day_cases))
        col_m2.metric("Approved", int((day_cases["decision"] == "approve").sum()))
        col_m3.metric("Rejected", int((day_cases["decision"] == "reject").sum()))

        with st.expander("Session context (market regime, DB state)"):
            st.markdown(day_cases["preamble"].iloc[0])

        perf_day = db_query("""
            SELECT symbol, status,
                   COALESCE(return_pct_actual, return_pct_counterfactual) AS return_pct,
                   return_pct_actual IS NULL AS is_counterfactual
            FROM mm_recommendation_performance
            WHERE as_of_date = ?
        """, (sel_day,))
        perf_by_sym = {r.symbol: r for r in perf_day.itertuples()}

        _DECISION_ICON = {"approve": ":white_check_mark:", "reject": ":x:"}
        for row in day_cases.itertuples():
            p = perf_by_sym.get(row.symbol)
            outcome = ""
            if p is not None and pd.notna(p.return_pct):
                kind = "counterfactual" if p.is_counterfactual else "actual"
                outcome = f" · {p.return_pct:+.2%} {kind} ({p.status})"
            label = (f"{_DECISION_ICON.get(row.decision, ':white_circle:')} "
                     f"#{row.pick_rank} {row.symbol} — {row.decision.upper()}{outcome}")
            with st.expander(label):
                st.markdown(row.case_md)

        day_extras = case_extras_df[case_extras_df["as_of_date"] == sel_day]
        for extra in day_extras.itertuples():
            with st.expander(f":clipboard: {extra.title}"):
                st.markdown(extra.section_md)

        st.subheader("Approved vs rejected — performance over time")
        st.caption(
            "Cumulative average return per decision day. Approvals use actual "
            "fills when available, counterfactual in shadow mode. Rejections are "
            "always counterfactual — the return the pick *would* have made, so "
            "a falling red line means the rejections were good calls."
        )
        perf_hist = db_query("""
            SELECT as_of_date, decision,
                   COALESCE(return_pct_actual, return_pct_counterfactual) AS return_pct
            FROM mm_recommendation_performance
            WHERE decision IN ('approve', 'reject')
              AND COALESCE(return_pct_actual, return_pct_counterfactual) IS NOT NULL
            ORDER BY as_of_date
        """)
        if perf_hist.empty:
            st.info("No attributed decisions yet — fills in after the nightly attribution run.")
        else:
            daily = (perf_hist.groupby(["as_of_date", "decision"])["return_pct"]
                     .mean().reset_index())
            daily["cum_pct"] = daily.groupby("decision")["return_pct"].cumsum() * 100
            fig = go.Figure()
            for dec, color, marker in (("approve", GAIN, "circle"),
                                       ("reject", LOSS, "x")):
                sub = daily[daily["decision"] == dec]
                if sub.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=sub["as_of_date"], y=sub["cum_pct"],
                    mode="lines+markers", name=dec,
                    line=dict(color=color, width=2),
                    marker=dict(symbol=marker, size=8),
                    text=[pctf(v, 2) for v in sub["cum_pct"]],
                    hovertemplate="%{x} · %{text}<extra>" + dec + "</extra>",
                ))
                fig.add_annotation(
                    x=sub["as_of_date"].iloc[-1], y=sub["cum_pct"].iloc[-1],
                    text=f"{dec} {sub['cum_pct'].iloc[-1]:+.1f}%",
                    showarrow=False, xanchor="left", xshift=8,
                    font=dict(color=color, size=12),
                )
            fig.update_layout(**nl_layout(
                height=320, margin=dict(l=8, r=90, t=32, b=8),
                title="Cumulative avg return by decision",
                yaxis=nl_axis(ticksuffix="%"), xaxis=nl_axis(),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            ))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
            st.caption(
                "The green line follows picks that were approved and "
                "the red line follows picks that were rejected, using the "
                "return each pick went on to make (for rejected picks, the "
                "return it would have made). Each point along the bottom "
                "adds one decision day's average return to a running "
                "percent total. The ideal picture is green rising while "
                "red falls — approvals made money and the rejections would "
                "have lost it, so the approval step added value.")


KRONOS_SCRIPT = PROJ_ROOT / "scripts" / "dl" / "kronos_nightly.py"
KRONOS_CMP_JSON = PROJ_ROOT / "exports" / "kronos_vs_patchtst_latest.json"


def _kronos_compute(symbol: str) -> bool:
    ""
    if str(PROJ_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(PROJ_ROOT / "src"))
    from macmini.dl.kronos_infer import (
        KronosForecaster, load_symbol_bars, persist_forecast)
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as c:
        bars = load_symbol_bars(c, symbol)
    fx = KronosForecaster.get()
    res = fx.forecast_symbols({symbol: bars})
    if symbol not in res:
        return False
    persist_forecast(DB_PATH, symbol, res[symbol])
    db_query.clear()
    return True


with tab_kronos:
    account_strip()
    st.header("Kronos — zero-shot price forecasts")
    st.caption(
        "Kronos-base (arXiv:2508.02739), run as published over our daily bars. "
        "Unproven on this universe: forecasts are annotation + out-of-sample "
        "evidence only and influence nothing. Judged by the same rank-IC gate "
        "as PatchTST (≥60 OOS days, t≥3). Evidence ledger started 2026-07-24.")

    latest_bar = db_query("SELECT MAX(date) AS d FROM mm_bars")["d"].iloc[0]

    ksyms = db_query("SELECT DISTINCT symbol FROM mm_bars ORDER BY symbol")["symbol"].tolist()
    ksym = sector_symbol_picker(ksyms, _sector_of, prefix="kronos")

    ksig = db_query(
        "SELECT * FROM mm_kronos_signals WHERE symbol = ? "
        "ORDER BY date DESC LIMIT 1", params=(ksym,))
    is_fresh = (not ksig.empty) and ksig["date"].iloc[0] == latest_bar

    if not is_fresh:
        note = ("No forecast stored for this symbol yet."
                if ksig.empty else
                f"Stored forecast is as of {ksig['date'].iloc[0]}; bars end {latest_bar}.")
        st.info(note)
        if st.button(f"Forecast {ksym} now", key="kronos_compute"):
            with st.spinner(f"Running Kronos-base on {ksym} (first call loads the model, ~20s)…"):
                ok = _kronos_compute(ksym)
            if ok:
                st.rerun()
            else:
                st.warning(f"{ksym} skipped — needs ≥64 bars of history.")

    if not ksig.empty:
        kpaths = db_query(
            "SELECT step, date, close, close_lo, close_hi FROM mm_kronos_paths "
            "WHERE symbol = ? AND as_of_date = ? ORDER BY step",
            params=(ksym, ksig["date"].iloc[0]))
        kbars = db_query(
            "SELECT date, open, high, low, close FROM mm_bars WHERE symbol = ? "
            "ORDER BY date DESC LIMIT 180", params=(ksym,)).sort_values("date")

        s = ksig.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Predicted 5d", f"{s['predicted_5d_return']*100:+.2f}%")
        m2.metric("Predicted 30d", f"{s['predicted_30d_return']*100:+.2f}%")
        m3.metric("Sample spread (5d σ)", f"{s['predicted_5d_vol']*100:.2f}%")
        m4.metric("As of", s["date"])

        fig = go.Figure(data=[go.Candlestick(
            x=kbars["date"], open=kbars["open"], high=kbars["high"],
            low=kbars["low"], close=kbars["close"], name=ksym)])
        fig.update_traces(increasing_line_color=GAIN, decreasing_line_color=LOSS,
                          selector=dict(type="candlestick"))
        if not kpaths.empty:
            last_close = float(kbars["close"].iloc[-1])
            px = [kbars["date"].iloc[-1]] + kpaths["date"].tolist()
            fig.add_trace(go.Scatter(
                x=px + px[::-1],
                y=([last_close] + kpaths["close_hi"].tolist()
                   + kpaths["close_lo"].tolist()[::-1] + [last_close]),
                fill="toself", fillcolor="rgba(27,95,168,0.10)",
                line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(
                x=px, y=[last_close] + kpaths["close"].tolist(),
                mode="lines", name="Kronos mean path",
                line=dict(color=ACCENT, width=2, dash="dash"),
                hovertemplate="%{x} · %{y:,.2f}<extra>forecast</extra>"))
        fig.update_layout(**nl_layout(
            height=520, xaxis_rangeslider_visible=False,
            margin=dict(l=8, r=24, t=8, b=8), showlegend=False,
            xaxis=nl_axis(), yaxis=nl_axis()))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
        st.caption(
            f"Candles are {ksym}'s real daily prices; the dashed blue line "
            f"past the last candle is the average of {int(s['sample_count'])} "
            f"forecast paths sampled from {s['model_id']}. The shaded band "
            "is the min-to-max spread of those sampled paths — a wider "
            "band means the model itself is less sure (it is model "
            "uncertainty, not a statistical confidence interval). This "
            "model has no proven track record on our stocks yet, so a "
            "rising line is a claim to verify, not a reason to trade.")

    st.subheader("Universe coverage")
    cov = db_query(
        "SELECT COUNT(*) AS n FROM mm_kronos_signals WHERE date = ?",
        params=(latest_bar,))["n"].iloc[0]
    st.write(f"{cov} symbols have a forecast as of {latest_bar}.")
    running = st.session_state.get("kronos_run_proc")
    if running is not None and running.poll() is None:
        st.info("Full-universe run in progress — refresh to update coverage.")
    elif st.button(f"Run full universe ({len(ksyms)} symbols, ~5 min GPU)", key="kronos_run_all"):
        import subprocess
        log_path = PROJ_ROOT / "logs" / "kronos_dashboard_run.log"
        with open(log_path, "ab") as lf:
            st.session_state["kronos_run_proc"] = subprocess.Popen(
                [str(PROJ_ROOT / "dl_venv" / "bin" / "python"), str(KRONOS_SCRIPT)],
                cwd=PROJ_ROOT, stdout=lf, stderr=lf)
        st.info(f"Started — tailing to {log_path.name}. Refresh to update coverage.")

    st.subheader("Kronos vs PatchTST — daily rank IC")
    if KRONOS_CMP_JSON.exists():
        import json as _json
        cmp_data = _json.loads(KRONOS_CMP_JSON.read_text())
        rows = []
        for label, key in (("Kronos", "kronos"), ("PatchTST", "patchtst")):
            e = cmp_data.get(key, {})
            rows.append({"model": label, "OOS days": e.get("n_days"),
                         "mean rank IC": e.get("mean_ic"),
                         "t-stat": e.get("t_stat"),
                         "gate (60d, t≥3)": "PASS" if e.get("gate_pass") else "—"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        pdiff = cmp_data.get("paired_diff_kronos_minus_patchtst", {})
        if pdiff.get("n_common_days", 0) > 1:
            st.caption(
                f"Paired diff (Kronos − PatchTST) over {pdiff['n_common_days']} common "
                f"days: mean {pdiff['mean_ic_diff']:+.4f}, t = {pdiff['t_stat']}.")
        else:
            st.caption("No overlapping scoreable days yet — a 5d-forward window "
                       "must complete before a day can be scored.")
        st.caption(f"computed {cmp_data.get('generated_at', '?')} · refresh via "
                   "`PYTHONPATH=src venv/bin/python scripts/dl/eval_kronos_vs_patchtst.py`")
    else:
        st.info("No comparison computed yet — it lands nightly with the Kronos cron, "
                "or run scripts/dl/eval_kronos_vs_patchtst.py manually.")

    st.caption("The model-pool rank-IC scoreboard now lives in the **Models** tab.")


SCOREBOARD_JSON = PROJ_ROOT / "exports" / "model_scoreboard.json"
BLEND_JSON = PROJ_ROOT / "exports" / "blend_bma_weights.json"

with tab_models:
    account_strip()
    st.header("Models — the OOS referee")
    st.caption(
        "Every model in the pool is judged on one instrument: daily "
        "cross-sectional rank IC against realized 5-day forward returns "
        "(enter next open, exit open five trading days later). Promotion "
        "needs ≥60 OOS days and a t-stat clearing the pool-size-adjusted "
        "bar. Until then everything here is annotation — it influences "
        "nothing.")

    if not SCOREBOARD_JSON.exists():
        st.info("No scoreboard yet — it lands with the nightly 21:15 run "
                "(scripts/dl/eval_model_scoreboard.py).")
    else:
        import json as _json
        sb = _json.loads(SCOREBOARD_JSON.read_text())
        mdf = pd.DataFrame(sb["models"])
        latest_bar = db_query("SELECT MAX(date) AS d FROM mm_bars")["d"].iloc[0]
        eff_bar = float(sb["effective_t_bar"])
        gate_days = int(sb["gate_min_days"])

        def _status(r) -> str:
            if r["clears_adjusted_bar"]:
                return "clears bar"
            if r["mean_ic"] is not None and r["mean_ic"] < 0:
                return "negative IC"
            return "accumulating"

        mdf["status"] = mdf.apply(_status, axis=1)
        _STATUS_COLOR = {"clears bar": GAIN, "negative IC": LOSS,
                         "accumulating": STATUS_GREY}
        scored = mdf[mdf["t_stat"].notna()].copy()
        best = scored.iloc[0] if not scored.empty else None
        deepest = mdf.loc[mdf["n_days"].idxmax()] if not mdf.empty else None

        kpi_strip([
            ("Models in pool", str(sb["n_models_in_pool"]), INK,
             f"gate ≥{gate_days} OOS days", MUTED),
            ("Effective t bar", f"{eff_bar:.2f}", INK,
             f"single {sb['single_model_t_bar']:.1f} · Bonferroni "
             f"{sb['bonferroni_adjusted_t_bar']:.2f}", MUTED),
            ("Best t-stat", f"{best['t_stat']:.2f}" if best is not None else "—",
             INK, _short_model(best["model_id"]) if best is not None else "—", MUTED),
            ("Clear the bar", str(int(mdf["clears_adjusted_bar"].sum())),
             GAINT if mdf["clears_adjusted_bar"].any() else INK,
             "days + t, jointly", MUTED),
            ("Deepest evidence", f"{int(deepest['n_days'])}d" if deepest is not None else "—",
             INK, _short_model(deepest["model_id"]) if deepest is not None else "—", MUTED),
        ])

        st.markdown('<div class="nl-kicker">01 · Evidence map</div>'
                    '<div class="nl-h4">Mean rank IC vs t-stat '
                    '<span class="nl-muted" style="font-weight:400;font-size:14px;">'
                    '· dot size = OOS days of evidence</span></div>',
                    unsafe_allow_html=True)
        if scored.empty:
            st.info("No model has a scoreable IC series yet.")
        else:
            fig = go.Figure()
            y_top, y_bot = 100.0, -100.0
            x_span = max(0.12, scored["mean_ic"].abs().max() * 1.35)
            fig.add_shape(type="rect", x0=0, x1=x_span, y0=eff_bar, y1=y_top,
                          fillcolor="rgba(46,158,91,0.07)", line_width=0)
            fig.add_hline(y=eff_bar, line_dash="dash", line_width=1.4,
                          line_color=ACCENT,
                          annotation_text=f"effective bar t = {eff_bar:.2f}",
                          annotation_font=dict(size=11, color=ACCENT),
                          annotation_position="top left")
            fig.add_vline(x=0, line_dash="dot", line_width=1,
                          line_color="rgba(20,20,20,0.30)")
            for status_name, grp in scored.groupby("status"):
                fig.add_trace(go.Scatter(
                    x=grp["mean_ic"], y=grp["t_stat"], mode="markers",
                    name=status_name,
                    marker=dict(
                        size=(grp["n_days"].clip(1, 90) / 90 * 24 + 9),
                        color=_STATUS_COLOR[status_name], opacity=0.85,
                        line=dict(width=1.5, color="#ffffff")),
                    customdata=list(zip(
                        grp["model_id"], grp["n_days"],
                        [f"{v:+.3f}" for v in grp["mean_ic"]])),
                    hovertemplate=("%{customdata[0]}<br>IC %{customdata[2]} · "
                                   "t %{y:.2f} · %{customdata[1]} OOS days"
                                   "<extra></extra>")))
            _label_ids = set()
            if best is not None:
                _label_ids.add(best["model_id"])
            if deepest is not None:
                _label_ids.add(deepest["model_id"])
            _label_ids.add("blend_bma")
            for r in scored.itertuples():
                if r.model_id in _label_ids:
                    fig.add_annotation(
                        x=r.mean_ic, y=r.t_stat, text=_short_model(r.model_id),
                        showarrow=False, yshift=14,
                        font=dict(size=11, color=MUTED))
            fig.update_layout(**nl_layout(
                height=680, margin=dict(l=56, r=20, t=10, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis=nl_axis(title=dict(text="mean daily rank IC",
                                         font=dict(size=11)),
                              tickformat="+.2f", range=[-x_span, x_span]),
                yaxis=nl_axis(title=dict(text="t-stat", font=dict(size=11)),
                              range=[y_bot, y_top])))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
            n_unscored = int(mdf["t_stat"].isna().sum())
            st.caption(
                "Each dot is one model: left-right is its average daily "
                "rank IC (how well its predictions ordered stocks — right "
                "of the dotted line beats chance) and up-down is the "
                "t-stat (how statistically solid that looks; above the "
                "dashed line clears the promotion bar). Dot size is how "
                "many days of out-of-sample evidence the model has, and "
                "color is the verdict: green clears the bar, red predicts "
                "backwards, grey is still gathering evidence"
                + (f" ({n_unscored} with under 2 scored days are left off)"
                   if n_unscored else "") + ". "
                "The only dots that matter are big AND high AND to the "
                "right — a high small dot may just be lucky, and a big dot "
                "near zero is proven mediocre.")

        st.markdown('<div class="nl-kicker" style="margin-top:24px;">02 · Leaderboard</div>'
                    '<div class="nl-h4">Rank-IC scoreboard, full pool</div>',
                    unsafe_allow_html=True)
        tbl = mdf[["model_id", "table", "n_days", "mean_ic", "ic_std",
                   "t_stat", "status"]].copy()
        tbl["days to gate"] = (gate_days - tbl["n_days"]).clip(lower=0)
        tbl["mean_ic"] = tbl["mean_ic"].map(
            lambda v: "—" if pd.isna(v) else f"{v:+.3f}")
        tbl["ic_std"] = tbl["ic_std"].map(
            lambda v: "—" if pd.isna(v) else f"{v:.3f}")
        tbl["t_stat"] = tbl["t_stat"].map(
            lambda v: "—" if pd.isna(v) else f"{v:.2f}")
        tbl = tbl.rename(columns={
            "model_id": "Model", "table": "Signals table", "n_days": "OOS days",
            "mean_ic": "Mean IC", "ic_std": "IC σ", "t_stat": "t-stat",
            "status": "Status", "days to gate": "Days to gate"})

        def _status_style(v: str) -> str:
            return (f"color:{_STATUS_COLOR.get(v, INK)};font-weight:600;"
                    if v in _STATUS_COLOR else "")

        sty = tbl.style.map(_status_style, subset=["Status"])
        st.dataframe(sty, use_container_width=True, hide_index=True,
                     height=min(560, 40 + 35 * len(tbl)))
        st.caption(
            f"{sb['n_models_in_pool']} model_ids → effective t bar "
            f"{eff_bar:.2f} (multiplicity rule: best-of-pool must clear "
            "the Bonferroni-adjusted bar, not the single-model 3.0). Days "
            "span each model's full signal history; the authoritative "
            "promotion reading is eval_dl_gate.py with the checkpoint's OOS "
            f"start date. Computed {sb.get('generated_at', '?')}.")

        st.markdown('<div class="nl-kicker" style="margin-top:24px;">03 · Daily IC</div>'
                    '<div class="nl-h4">Cumulative daily rank IC '
                    '<span class="nl-muted" style="font-weight:400;font-size:14px;">'
                    '· slope is skill, wiggle is noise</span></div>',
                    unsafe_allow_html=True)
        table_of = dict(zip(mdf["model_id"], mdf["table"]))
        opts = scored["model_id"].tolist()
        defaults = [m for m in [
            best["model_id"] if best is not None else None,
            "blend_bma", "patchtst_seq64_dm128_20260720"]
            if m in opts and m is not None][:3]
        sel = st.multiselect("Models (max 5)", opts, default=defaults,
                             max_selections=5, key="models_ic_sel")
        if sel:
            sel = [m for m in opts if m in sel]
            fig = go.Figure()
            ns = {}
            for i, mid in enumerate(sel):
                s = ic_series(mid, table_of[mid], str(latest_bar))
                if s.empty:
                    continue
                ns[mid] = len(s)
                cum = s.cumsum()
                color = MODEL_SERIES[i % len(MODEL_SERIES)]
                fig.add_trace(go.Scatter(
                    x=list(cum.index), y=cum.values, mode="lines+markers",
                    name=_short_model(mid),
                    line=dict(color=color, width=2), marker=dict(size=5),
                    text=[f"{v:+.3f}" for v in cum.values],
                    hovertemplate="%{x} · cum IC %{text}<extra>"
                                  + _short_model(mid) + "</extra>"))
                fig.add_annotation(
                    x=cum.index[-1], y=float(cum.iloc[-1]),
                    text=_short_model(mid), showarrow=False,
                    xanchor="left", xshift=8, font=dict(size=11, color=color))
            fig.add_hline(y=0, line_width=1, line_color="rgba(20,20,20,0.30)")
            fig.update_layout(**nl_layout(
                height=360, margin=dict(l=56, r=130, t=10, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis=nl_axis(type="category"),
                yaxis=nl_axis(title=dict(text="Σ daily IC", font=dict(size=11)))))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
            st.caption(
                "Each line is one model; every step to the right is one "
                "scored trading day, and the line moves up when the model "
                "ordered that day's stocks correctly and down when it got "
                "them backwards. The vertical axis is the running total of "
                "those daily scores. A steady climb means repeatable "
                "skill, while wandering around zero means noise — and "
                "short lines mean little evidence either way (days per "
                "series: "
                + " · ".join(f"{_short_model(m)} {n}" for m, n in ns.items())
                + ").")

        st.markdown('<div class="nl-kicker" style="margin-top:24px;">04 · Rank calibration</div>'
                    '<div class="nl-h4">Predicted vs realized rank, latest scoreable day</div>',
                    unsafe_allow_html=True)
        cal_default = (
            scored.loc[scored["mean_ic"].astype(float).idxmax(), "model_id"]
            if not scored.empty and scored["mean_ic"].notna().any()
            else (opts[0] if opts else None))
        if cal_default is None:
            st.info("Nothing scoreable yet.")
        else:
            if st.session_state.get("models_cal_model") not in opts:
                st.session_state["models_cal_model"] = cal_default
            _codes = _model_codes(opts)
            _cal_opts = sorted(opts, key=lambda m: _codes[m])
            _PER_MROW = 13
            for _mrs in range(0, len(_cal_opts), _PER_MROW):
                _mrow = _cal_opts[_mrs:_mrs + _PER_MROW]
                mcols = st.columns(_PER_MROW)
                for _mc, _mid in zip(mcols, _mrow):
                    _mkey = "calbtn_" + "".join(
                        ch if ch.isalnum() else "_" for ch in _mid)
                    if _mc.button(
                        _codes[_mid], key=_mkey,
                        help=f"{_short_model(_mid)} · {_model_date(_mid)}",
                        type="primary"
                        if _mid == st.session_state["models_cal_model"]
                        else "secondary",
                    ):
                        st.session_state["models_cal_model"] = _mid
                        st.rerun()
            with st.expander("Model glossary — what each button is"):
                _grows = []
                for _mid in _cal_opts:
                    _fam = _model_family(_mid)
                    _blurb = _MODEL_GLOSSARY.get(
                        _fam, (None, "No blurb yet for this family."))[1]
                    _grows.append(
                        f"| **{_codes[_mid]}** | {_short_model(_mid)} | "
                        f"{_model_date(_mid)} | {_blurb} |")
                st.markdown(
                    "| Label | Model | Date | What makes it special |\n"
                    "|---|---|---|---|\n" + "\n".join(_grows))
            cal_mid = st.session_state["models_cal_model"]
            cs = rank_cross_section(cal_mid, table_of[cal_mid], str(latest_bar))
            if cs.empty:
                st.info("No day with a completed forward window and ≥30 names "
                        "for this model yet.")
            else:
                day = cs["day"].iloc[0]
                rho = float(cs["pred_rank"].corr(cs["real_rank"]))
                st.markdown(
                    f'<div style="font-size:13px;color:{INK};margin:2px 0 0 2px;">'
                    f'Spearman ρ = {rho:+.3f} · n = {len(cs)} · {day}</div>',
                    unsafe_allow_html=True)

                import numpy as np

                sfig = go.Figure()
                sfig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                               line=dict(color="rgba(20,20,20,0.35)",
                                         width=1.2, dash="dash"))
                cs["sector"] = cs["symbol"].map(_sector_of).fillna("Other")
                for sec, color in SECTOR_COLORS.items():
                    sub = cs[cs["sector"] == sec]
                    if sub.empty:
                        continue
                    sfig.add_trace(go.Scatter(
                        x=sub["pred_rank"], y=sub["real_rank"],
                        mode="markers", name=sec,
                        marker=dict(size=6, color=color, opacity=0.85,
                                    line=dict(width=1, color="#ffffff")),
                        customdata=sub[["symbol", "sector"]],
                        hovertemplate="%{customdata[0]} · %{customdata[1]}"
                                      "<br>predicted pct %{x:.2f} · "
                                      "realized pct %{y:.2f}<extra></extra>"))
                sfig.update_layout(**nl_layout(
                    height=400, margin=dict(l=44, r=8, t=10, b=36),
                    showlegend=False,
                    xaxis=nl_axis(title=dict(text="predicted percentile",
                                             font=dict(size=10)),
                                  range=[-0.02, 1.02],
                                  tickfont=dict(size=9)),
                    yaxis=nl_axis(title=dict(text="realized 5d percentile",
                                             font=dict(size=10)),
                                  range=[-0.02, 1.02],
                                  tickfont=dict(size=9))))

                _edges = np.linspace(0.0, 1.0, 11)
                _px = np.clip(np.digitize(cs["pred_rank"], _edges) - 1, 0, 9)
                _py = np.clip(np.digitize(cs["real_rank"], _edges) - 1, 0, 9)
                _z = np.zeros((10, 10))
                for _a, _b in zip(_px, _py):
                    _z[_b, _a] += 1
                _col_tot = _z.sum(axis=0, keepdims=True).clip(min=1)
                _pct = 100.0 * _z / _col_tot
                _dec = [f"{i*10}" for i in range(10)]
                gfig = go.Figure(go.Heatmap(
                    z=_z, x=_dec, y=_dec,
                    colorscale=[[0, "#ffffff"], [1, "#1b5fa8"]],
                    text=np.where(_z > 0, _z.astype(int).astype(str), ""),
                    texttemplate="%{text}",
                    textfont=dict(size=8, color=MUTED),
                    customdata=_pct,
                    hovertemplate=("predicted decile %{x} · realized %{y}"
                                   "<br>%{z:.0f} stocks · %{customdata:.0f}%% "
                                   "of that predicted decile<extra></extra>"),
                    showscale=False, xgap=1, ygap=1))
                gfig.add_shape(type="line", x0=-0.5, y0=-0.5, x1=9.5, y1=9.5,
                               line=dict(color="rgba(20,20,20,0.35)",
                                         width=1.2, dash="dash"))
                gfig.update_layout(**nl_layout(
                    height=400, margin=dict(l=44, r=8, t=10, b=36),
                    xaxis=nl_axis(title=dict(text="predicted decile",
                                             font=dict(size=10)),
                                  tickfont=dict(size=9)),
                    yaxis=nl_axis(title=dict(text="realized decile",
                                             font=dict(size=10)),
                                  tickfont=dict(size=9))))

                _tbl = table_of[cal_mid]
                with sqlite3.connect(f"file:{DB_PATH}?mode=ro",
                                     uri=True) as _c:
                    top10 = pd.read_sql_query(
                        f"SELECT symbol, date, predicted_5d_return "
                        f"FROM {_tbl} WHERE model_id = ? AND date = "
                        f"(SELECT MAX(date) FROM {_tbl} WHERE "
                        f"model_id = ?) AND predicted_5d_return IS "
                        f"NOT NULL ORDER BY predicted_5d_return "
                        f"DESC LIMIT 10",
                        _c, params=(cal_mid, cal_mid))

                _cal_legend = " &nbsp; ".join(
                    f'<span style="display:inline-block;width:8px;height:8px;'
                    f'border-radius:50%;background:{c};margin-right:4px;"></span>'
                    f'<span class="nl-muted" style="font-size:12px;">{s}</span>'
                    for s, c in SECTOR_COLORS.items())
                st.markdown(f'<div style="margin:6px 0 4px 2px;">{_cal_legend}'
                            '</div>', unsafe_allow_html=True)

                pc1, pc2, pc3 = st.columns(3)
                pc1.markdown('<div class="nl-kicker" style="text-align:center;">'
                             'Scatter · by sector</div>',
                             unsafe_allow_html=True)
                pc1.plotly_chart(sfig, use_container_width=True,
                                 config=PLOTLY_CFG)
                pc2.markdown('<div class="nl-kicker" style="text-align:center;">'
                             'Density grid · counts</div>',
                             unsafe_allow_html=True)
                pc2.plotly_chart(gfig, use_container_width=True,
                                 config=PLOTLY_CFG)
                with pc3:
                    if top10.empty:
                        st.markdown('<div class="nl-kicker" '
                                    'style="text-align:center;">'
                                    'Top-ranked today</div>',
                                    unsafe_allow_html=True)
                        st.info("No current predictions stored for this model.")
                    else:
                        _pd_day = top10["date"].iloc[0]
                        st.markdown(
                            '<div class="nl-kicker" style="text-align:center;">'
                            f'Top-ranked today · {_pd_day}</div>',
                            unsafe_allow_html=True)
                        _t = top10.iloc[::-1]
                        _tcolors = [
                            SECTOR_COLORS.get(_sector_of.get(s, "Other"),
                                              SECTOR_COLORS["Other"])
                            for s in _t["symbol"]]
                        tfig = go.Figure(go.Bar(
                            x=_t["predicted_5d_return"] * 100,
                            y=_t["symbol"], orientation="h",
                            marker=dict(color=_tcolors, opacity=0.88),
                            customdata=[pctf(v * 100, 2)
                                        for v in _t["predicted_5d_return"]],
                            hovertemplate=("%{y} · predicted 5d "
                                           "%{customdata}<extra></extra>")))
                        tfig.update_layout(**nl_layout(
                            height=400, margin=dict(l=8, r=12, t=10, b=36),
                            xaxis=nl_axis(
                                title=dict(text="predicted 5d return (%)",
                                           font=dict(size=10)),
                                tickformat=".1f", tickfont=dict(size=9)),
                            yaxis=nl_axis(tickfont=dict(size=11))))
                        st.plotly_chart(tfig, use_container_width=True,
                                        config=PLOTLY_CFG)

                st.caption(
                    "Three views of the same question — can this model "
                    "order stocks?  \n"
                    "**Scatter · by sector:** each dot is one stock on the "
                    "latest gradeable day; left-right is where the model "
                    "ranked its expected 5-day return, bottom-top where "
                    "the return actually landed, colors by sector (a "
                    "perfect model puts every dot on the dashed diagonal; "
                    "the ρ above measures the lean and IS that day's IC).  \n"
                    "**Density grid · counts:** the same dots binned into "
                    "deciles — darker cell = more stocks; a no-skill model "
                    "spreads mass evenly, a skilled one concentrates it "
                    "along the diagonal.  \n"
                    "**Top-ranked today:** the model's 10 highest-ranked "
                    "names from its newest predictions — annotation only, "
                    "nothing trades from this. Ranks are used instead of "
                    "raw predicted sizes because ordering stocks is the "
                    "only skill any model here has demonstrated (audit "
                    "C3).  \n"
                    "To use this for the next trade: first make sure "
                    "the model clears the bar on the evidence map at the "
                    "top of the page — if it doesn't, treat everything "
                    "here as practice, not signal. If it does, and the "
                    "dark cells hug the top-right of the diagonal, the "
                    "top-ranked list is where its measured edge lives — "
                    "candidates, not orders. Before acting, cross-check a "
                    "candidate against the Picks tab and the account's "
                    "risk limits: a name both here AND in the pipeline's "
                    "picks is a confirmed idea; a name only the model "
                    "likes is a research lead to watch, because these "
                    "models are annotation-only and place no trades.")

        st.markdown('<div class="nl-kicker" style="margin-top:24px;">05 · Blend</div>'
                    '<div class="nl-h4">blend_bma member weights</div>',
                    unsafe_allow_html=True)
        if not BLEND_JSON.exists():
            st.info("No blend weights yet — blend_bma.py writes them nightly.")
        else:
            bw = _json.loads(BLEND_JSON.read_text())
            ws = pd.Series(bw.get("weights", {})).sort_values()
            nz = ws[ws > 0]
            if nz.empty:
                st.info("All member weights are zero.")
            else:
                fig = go.Figure(go.Bar(
                    x=nz.values, y=[_short_model(m) for m in nz.index],
                    orientation="h", marker_color=ACCENT,
                    text=[f"{v:.1%}" for v in nz.values],
                    textposition="outside", cliponaxis=False,
                    textfont=dict(size=12),
                    hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
                fig.update_layout(**nl_layout(
                    height=max(220, 30 * len(nz) + 60), bargap=0.4,
                    margin=dict(l=150, r=70, t=6, b=34),
                    xaxis=nl_axis(tickformat=".0%"), yaxis=nl_axis()))
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
                st.caption(
                    "Each bar is one member model of the blended signal, "
                    "and its length is the share of influence that model "
                    "currently gets when the blend combines everyone's "
                    "predictions. Weights are computed from each model's "
                    "out-of-sample track record, so members with more "
                    "evidence of skill get longer bars "
                    f"({len(ws) - len(nz)} zero-weight members are hidden; "
                    f"formula: {bw.get('note', '?')}; computed "
                    f"{str(bw.get('generated_at', '?'))[:10]}). One dominant "
                    "bar means the blend is mostly one model in disguise — "
                    "expect the bars to even out as younger models build "
                    "history.")

        st.markdown('<div class="nl-kicker" style="margin-top:24px;">'
                    '06 · Exit policies</div>'
                    '<div class="nl-h4">Phase C — learned exit annotation '
                    '<span class="nl-muted" style="font-weight:400;font-size:14px;">'
                    '· HOLD / EXIT / TIGHTEN_STOP, shadow only</span></div>',
                    unsafe_allow_html=True)
        xp = pd.DataFrame(sb.get("exit_policies", []))
        if xp.empty:
            st.info("No exit-policy entries yet — they appear after the "
                    "first Sunday training run.")
        else:
            disp = pd.DataFrame({
                "policy": xp["algo"].str.upper(),
                "model": xp.get("model_id", pd.Series(dtype=str)).fillna("not trained yet"),
                "OOS days": xp["oos_annotation_days"].astype(int),
                "uplift/trade": xp.get("mean_uplift_logret", pd.Series(dtype=float))
                    .map(lambda v: f"{v:+.4f}" if pd.notna(v) else "—"),
                "t": xp.get("t_stat", pd.Series(dtype=float))
                    .map(lambda v: f"{v:+.2f}" if pd.notna(v) else "—"),
                "gate": xp["clears_gate"].map({True: "CLEARS", False: "accumulating"}),
            })
            st.dataframe(disp, use_container_width=True, hide_index=True)
            ann = db_query(
                """SELECT policy_id, symbol, date, action, policy_logits_json
                   FROM mm_rl_signals
                   WHERE policy_id LIKE 'rl_exit_%' AND policy_id NOT LIKE '%smoke%'
                     AND date = (SELECT MAX(date) FROM mm_rl_signals
                                 WHERE policy_id LIKE 'rl_exit_%'
                                   AND policy_id NOT LIKE '%smoke%')
                   ORDER BY symbol, policy_id""")
            if ann.empty:
                st.caption("No live annotations yet — the nightly 22:15 "
                           "chain writes one row per open position per "
                           "policy once a non-smoke checkpoint exists.")
            else:
                _ACT = {0: "HOLD", 1: "EXIT", 2: "TIGHTEN_STOP"}
                ann["says"] = ann["action"].map(_ACT)
                st.dataframe(
                    ann[["date", "symbol", "policy_id", "says"]],
                    use_container_width=True, hide_index=True)
            st.caption(
                "Each row is a model that watches our open positions every "
                "evening and records what it would do — keep holding, sell "
                "tomorrow, or pull the protective stop up — without touching "
                "anything. 'Uplift/trade' is how much better (in log-return "
                "per trade) its exits would have been than the fixed-bracket "
                "rule on held-out history, and 'OOS days' counts real "
                f"evenings of live annotations (gate: ≥{gate_days} days and "
                "t ≥ 3). Backtests showed simple trailing rules don't "
                "beat the bracket, so these policies are the serious attempt "
                "— and they only earn control if the evidence table above "
                "says so.")


with tab_clusters:
    latest_bar = db_query("SELECT MAX(date) AS d FROM mm_bars")["d"].iloc[0]
    sd = structure_data(str(latest_bar))
    if sd is None:
        st.info("Not enough bar history to compute market structure.")
    else:
        import numpy as np

        syms, labels = sd["syms"], sd["labels"]
        meta = {m["id"]: m for m in sd["cluster_meta"]}
        big_ids = [m["id"] for m in sd["cluster_meta"] if m["n"] >= 15][:6]
        micro_ids = [m["id"] for m in sd["cluster_meta"]
                     if m["id"] not in big_ids]
        _hues = ["#1b5fa8", "#b0711f", "#c23a63", "#7b4b94", "#0e8a70",
                 "#a8552d"]
        cl_color = {cid: _hues[i] for i, cid in enumerate(big_ids)}
        cname = {cid: meta[cid]["name"] for cid in meta}
        RET_SCALE = [[0, LOSS], [0.5, NEU3], [1, GAIN]]
        CORR_SCALE = [[0, "#1b5fa8"], [0.5, NEU3], [1, "#c23a63"]]

        st.markdown(
            '<div class="nl-kicker">Market structure</div>'
            '<div class="nl-h4">What the prices say the groups are '
            '<span class="nl-muted" style="font-weight:400;font-size:14px;">'
            f'· {sd["window"][0]} → {sd["window"][1]} · annotation only'
            '</span></div>', unsafe_allow_html=True)
        kpi_strip([
            ("Symbols", str(len(syms)), INK,
             f"{sd['n_dropped']} dropped (<90% history)", MUTED),
            ("Clusters", str(sd["k"]), INK,
             f"{len(big_ids)} large · silhouette {sd['sil']:.2f}", MUTED),
            ("Real factors", str(sd["n_real"]), INK,
             "eigenvalues above noise", MUTED),
            ("Sector agreement", f"{sd['ari_sector']:.2f}", INK,
             "ARI vs mm_sector_map", MUTED),
            ("Stability", f"{sd['ari_halves']:.2f}", INK,
             "ARI, half-year vs half-year", MUTED),
        ])

        h_hdr, h_ctl = st.columns([3.2, 2.8])
        with h_hdr:
            st.markdown(
                '<div class="nl-kicker" style="margin-top:18px;">'
                '01 · Structure map</div>'
                '<div class="nl-h4">Every symbol placed by co-movement</div>',
                unsafe_allow_html=True)
        with h_ctl:
            color_by = st.segmented_control(
                "Color by", ["Cluster", "Sector", "Return", "Beta"],
                default="Cluster", key="clu_color",
                label_visibility="collapsed") or "Cluster"

        cd = np.column_stack([
            syms, sd["sectors"],
            [cname[c] for c in labels],
            [f"{r:+.1%}" for r in sd["win_ret"]],
            [f"{b:.2f}" for b in sd["beta"]]])
        HOVER = ("%{customdata[0]} · %{customdata[1]}<br>"
                 "%{customdata[2]}<br>12m %{customdata[3]} · "
                 "β %{customdata[4]}<extra></extra>")
        fig = go.Figure()
        if color_by == "Cluster":
            for cid in big_ids:
                m = labels == cid
                fig.add_trace(go.Scatter(
                    x=sd["mds"][m, 0], y=sd["mds"][m, 1], mode="markers",
                    name=cname[cid],
                    marker=dict(size=8, color=cl_color[cid], opacity=0.85,
                                line=dict(width=1, color="#ffffff")),
                    customdata=cd[m], hovertemplate=HOVER))
            m = np.isin(labels, micro_ids)
            if m.any():
                fig.add_trace(go.Scatter(
                    x=sd["mds"][m, 0], y=sd["mds"][m, 1], mode="markers",
                    name=f"Micro ({len(micro_ids)} clusters)",
                    marker=dict(size=7, color="#6b7a84", opacity=0.6,
                                line=dict(width=1, color="#ffffff")),
                    customdata=cd[m], hovertemplate=HOVER))
        elif color_by == "Sector":
            for sec_name, color in SECTOR_COLORS.items():
                m = np.array([s == sec_name for s in sd["sectors"]])
                if not m.any():
                    continue
                fig.add_trace(go.Scatter(
                    x=sd["mds"][m, 0], y=sd["mds"][m, 1], mode="markers",
                    name=sec_name,
                    marker=dict(size=8, color=color, opacity=0.85,
                                line=dict(width=1, color="#ffffff")),
                    customdata=cd[m], hovertemplate=HOVER))
        else:
            vals = sd["win_ret"] if color_by == "Return" else sd["beta"]
            marker = dict(size=8, color=vals, opacity=0.9,
                          line=dict(width=1, color="#ffffff"),
                          colorbar=dict(thickness=12, outlinewidth=0,
                                        tickfont=dict(size=11)))
            if color_by == "Return":
                marker.update(colorscale=RET_SCALE, cmid=0.0, cmin=-0.6,
                              cmax=0.6,
                              colorbar=dict(thickness=12, outlinewidth=0,
                                            tickformat="+.0%",
                                            tickfont=dict(size=11)))
            else:
                marker.update(colorscale=[[0, "#dce7f3"], [1, "#123f70"]],
                              cmin=0.0, cmax=2.0)
            fig.add_trace(go.Scatter(
                x=sd["mds"][:, 0], y=sd["mds"][:, 1], mode="markers",
                marker=marker, customdata=cd, hovertemplate=HOVER,
                showlegend=False))
        held = {p["symbol"] for p in (alpaca_get("/v2/positions") or [])
                if isinstance(p, dict)} & set(syms)
        if held:
            hi = [syms.index(s) for s in sorted(held)]
            fig.add_trace(go.Scatter(
                x=sd["mds"][hi, 0], y=sd["mds"][hi, 1],
                mode="markers+text", name="Open positions",
                text=sorted(held), textposition="top center",
                textfont=dict(size=11, color=INK),
                marker=dict(size=14, symbol="circle-open", color=INK,
                            line=dict(width=2)),
                hoverinfo="skip"))
        fig.update_layout(**nl_layout(
            height=520, margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        font=dict(size=12), itemsizing="constant"),
            xaxis=nl_axis(showticklabels=False, title=None),
            yaxis=nl_axis(showticklabels=False, title=None)))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
        st.caption(
            "Each dot is one stock, placed so that stocks whose daily "
            "moves track each other (after removing the market-wide move "
            "everyone shares) sit close together — the axes themselves "
            "have no meaning, only distance does. The buttons recolor the "
            "same map: by discovered cluster, by official sector label, "
            "by trailing 12-month return, or by sensitivity to the "
            "overall market (beta). Black rings mark the open positions — "
            "rings sitting in the same patch of the map are, whatever "
            "their labels say, the same bet.")

        h_hdr2, h_ctl2 = st.columns([3.2, 2.8])
        with h_hdr2:
            st.markdown(
                '<div class="nl-kicker" style="margin-top:24px;">'
                '02 · Co-movement</div>'
                '<div class="nl-h4">Residual correlation, '
                'cluster-ordered</div>', unsafe_allow_html=True)
        with h_ctl2:
            hm_view = st.segmented_control(
                "Grain", ["Cluster averages", "All symbols"],
                default="Cluster averages", key="clu_hm",
                label_visibility="collapsed") or "Cluster averages"
        fig = go.Figure()
        if hm_view == "Cluster averages":
            names = [cname[c] for c in range(1, sd["k"] + 1)]
            fig.add_trace(go.Heatmap(
                z=sd["block"], x=names, y=names, zmid=0, zmin=-0.35,
                zmax=0.35, colorscale=CORR_SCALE,
                colorbar=dict(thickness=12, outlinewidth=0,
                              tickfont=dict(size=11)),
                hovertemplate="%{y}<br>%{x}<br>mean ρ %{z:.3f}"
                              "<extra></extra>"))
            fig.update_layout(**nl_layout(
                height=560, margin=dict(l=20, r=20, t=10, b=20),
                xaxis=nl_axis(tickfont=dict(size=10), tickangle=45),
                yaxis=nl_axis(tickfont=dict(size=10), autorange="reversed")))
        else:
            o = sd["order"]
            osyms = [syms[i] for i in o]
            fig.add_trace(go.Heatmap(
                z=sd["corr"][np.ix_(o, o)], x=osyms, y=osyms, zmid=0,
                zmin=-0.6, zmax=0.6, colorscale=CORR_SCALE,
                colorbar=dict(thickness=12, outlinewidth=0,
                              tickfont=dict(size=11)),
                hovertemplate="%{y} × %{x}<br>ρ %{z:.3f}<extra></extra>"))
            olab = labels[o]
            for b in np.where(np.diff(olab) != 0)[0]:
                fig.add_hline(y=b + 0.5, line_width=0.6, line_color=INK)
                fig.add_vline(x=b + 0.5, line_width=0.6, line_color=INK)
            fig.update_layout(**nl_layout(
                height=640, margin=dict(l=20, r=20, t=10, b=20),
                xaxis=nl_axis(showticklabels=False),
                yaxis=nl_axis(showticklabels=False,
                              autorange="reversed")))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
        st.caption(
            "Each cell shows how strongly two stocks (or two whole "
            "clusters) move together once the market-wide move is "
            "removed — warm red for together, cool blue for opposite, "
            "pale for unrelated. Stocks are ordered so cluster-mates sit "
            "side by side, which is why red squares line the diagonal: "
            "those are the discovered groups. Off-diagonal warmth means "
            "two groups are cousins; blue patches (like miners vs "
            "software) are natural hedges against each other.")

        st.markdown(
            '<div class="nl-kicker" style="margin-top:24px;">'
            '03 · Composition</div>'
            '<div class="nl-h4">Who is in each cluster · color = '
            '12-month return</div>', unsafe_allow_html=True)
        ids = [f"c{c}" for c in range(1, sd["k"] + 1)] + list(syms)
        tm_labels = ([cname[c] for c in range(1, sd["k"] + 1)]
                     + list(syms))
        parents = [""] * sd["k"] + [f"c{c}" for c in labels]
        values = [meta[c]["n"] for c in range(1, sd["k"] + 1)] + [1] * len(syms)
        cl_mean = {c: float(np.mean(sd["win_ret"][labels == c]))
                   for c in range(1, sd["k"] + 1)}
        colors = ([cl_mean[c] for c in range(1, sd["k"] + 1)]
                  + list(sd["win_ret"]))
        fig = go.Figure(go.Treemap(
            ids=ids, labels=tm_labels, parents=parents, values=values,
            branchvalues="total",
            marker=dict(colors=colors, colorscale=RET_SCALE, cmid=0.0,
                        cmin=-0.6, cmax=0.6,
                        line=dict(width=1.5, color="#ffffff"),
                        colorbar=dict(thickness=12, outlinewidth=0,
                                      tickformat=".0%",
                                      tickfont=dict(size=11))),
            customdata=[pctf(v * 100, 2) for v in colors],
            hovertemplate="%{label}<br>12m %{customdata}<extra></extra>",
            tiling=dict(pad=2)))
        fig.update_layout(**nl_layout(
            height=560, margin=dict(l=10, r=10, t=10, b=10)))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
        st.caption(
            "Each big box is one discovered cluster, sized by how many "
            "stocks it holds; the small tiles inside are the member "
            "stocks. Color is each stock's trailing 12-month price "
            "return — green made money, red lost it — so a mostly-green "
            "box is a group whose whole trade worked. Click a cluster to "
            "zoom in and read its members; click the header to zoom back "
            "out.")

        st.markdown(
            '<div class="nl-kicker" style="margin-top:24px;">'
            '04 · Spectrum</div>'
            '<div class="nl-h4">How many dimensions are real</div>',
            unsafe_allow_html=True)
        ranks = list(range(1, len(sd["evals"]) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ranks, y=sd["evals"], mode="lines+markers",
            line=dict(color=ACCENT, width=2), marker=dict(size=6),
            hovertemplate="rank %{x} · eigenvalue %{y:.1f}<extra></extra>",
            showlegend=False))
        fig.add_hline(y=sd["mp_upper"], line_width=1.2, line_dash="dash",
                      line_color=LOSS,
                      annotation_text=f"noise bound {sd['mp_upper']:.2f}",
                      annotation_font=dict(size=11, color=LOSST))
        fig.update_layout(**nl_layout(
            height=360, margin=dict(l=56, r=20, t=10, b=40),
            xaxis=nl_axis(title=dict(text="factor rank",
                                     font=dict(size=11))),
            yaxis=nl_axis(type="log",
                          range=[float(np.log10(sd["evals"][-1] * 0.7)),
                                 float(np.log10(sd["evals"][0] * 1.8))],
                          title=dict(text="eigenvalue (log)",
                                     font=dict(size=11)))))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
        st.caption(
            f"Each dot measures how much of the whole universe's daily "
            "movement one shared 'dimension' explains, from the biggest "
            "(the market itself, "
            f"{sd['pc1_pct']:.0%} of everything) on down. The dashed red "
            "line is the level pure random noise would reach — a "
            "mathematical bound, not a judgment call — so only the "
            f"{sd['n_real']} dots above it represent real, shared forces. "
            "Everything below the line is statistical static, which is "
            "why the clusters above are built from a handful of strong "
            "patterns rather than hundreds of weak ones.")


with tab_shadow:
    account_strip()
    st.header("Shadow book — every recommendation, tracked")
    st.caption(
        "Approved picks carry their actual fills. Rejected and expired picks "
        "are tracked anyway: the counterfactual bracket simulation says what "
        "they would have returned, and still-open untraded picks are marked "
        "to the latest close. The approved-vs-rejected spread is the "
        "counterfactual attribution loop — the audit calls it the system's "
        "most valuable instrument.")

    perf_all = db_query("""
        SELECT rec_id, as_of_date, symbol, decision, status, exit_reason,
               days_held,
               return_pct_actual, return_pct_counterfactual,
               COALESCE(return_pct_counterfactual, return_pct_actual) AS ret_shadow,
               COALESCE(return_pct_actual, return_pct_counterfactual) AS ret_best
        FROM mm_recommendation_performance
    """)
    ledger_all = db_query("""
        SELECT rec_id, entry_date, rank, symbol, sector, strategy_name,
               status, fill_status, entry_est_price, actual_fill_price,
               exit_status, exit_price
        FROM mm_recs_ledger
        ORDER BY entry_date DESC, rank
    """)

    closed = perf_all[perf_all["status"] == "closed"]
    appr_cl = closed[(closed["decision"] == "approve")
                     & closed["ret_best"].notna()]
    rej_cl = closed[(closed["decision"] == "reject")
                    & closed["ret_shadow"].notna()]
    appr_mean = float(appr_cl["ret_best"].mean()) if not appr_cl.empty else None
    rej_mean = float(rej_cl["ret_shadow"].mean()) if not rej_cl.empty else None
    spread = (appr_mean - rej_mean) if (appr_mean is not None
                                        and rej_mean is not None) else None

    n_rec = len(ledger_all)
    n_appr = int((ledger_all["status"] == "approved").sum())
    n_fill = int((ledger_all["fill_status"] == "filled").sum())

    def _pcol(v):
        return MUTED if v is None else (GAINT if v >= 0 else LOSST)

    kpi_strip([
        ("Recommendations", str(n_rec), INK,
         f"{n_appr} approved · {n_fill} filled", MUTED),
        ("Closed outcomes", str(len(closed)), INK,
         f"{len(appr_cl)} approved · {len(rej_cl)} rejected", MUTED),
        ("Approved avg", pctf(appr_mean * 100) if appr_mean is not None else "—",
         _pcol(appr_mean), "actual fills, closed", MUTED),
        ("Rejected avg", pctf(rej_mean * 100) if rej_mean is not None else "—",
         _pcol(rej_mean), "counterfactual, closed", MUTED),
        ("Decision spread", pctf(spread * 100) if spread is not None else "—",
         _pcol(spread), "approved − rejected · >0 = good calls", MUTED),
    ])

    st.markdown('<div class="nl-kicker">01 · Funnel</div>'
                '<div class="nl-h4">From universe to the book '
                '<span class="nl-muted" style="font-weight:400;font-size:14px;">'
                '· universe/sweep are current snapshots, the rest all-time</span></div>',
                unsafe_allow_html=True)
    n_univ = int(db_query(
        "SELECT COUNT(DISTINCT symbol) AS n FROM mm_bars")["n"].iloc[0])
    n_swept = int(db_query("""
        SELECT COUNT(DISTINCT symbol) AS n FROM mm_backtest_runs
        WHERE run_id = (SELECT run_id FROM mm_backtest_runs
                        ORDER BY run_id DESC LIMIT 1)""")["n"].iloc[0])
    stages = [
        ("Universe", n_univ, "symbols with daily bars on file"),
        ("Backtested", n_swept, "symbols in the latest walk-forward sweep"),
        ("Recommended", n_rec, "ranked picks exported to the mini (all-time)"),
        ("Approved", n_appr, "approved in the morning brief"),
        ("Filled", n_fill, "bracket order filled at the broker"),
    ]
    _funnel_fill = [_mix("#ffffff", ACCENT, t) for t in
                    (0.25, 0.42, 0.60, 0.78, 0.95)]
    fig = go.Figure(go.Funnel(
        y=[s[0] for s in stages], x=[s[1] for s in stages],
        textinfo="value+percent initial",
        textfont=dict(family=SERIF, size=13),
        marker=dict(color=_funnel_fill,
                    line=dict(width=2, color="#ffffff")),
        connector=dict(line=dict(color=DIVIDER, width=1, dash="dot")),
        customdata=[s[2] for s in stages],
        hovertemplate="%{y}: %{x:,d}<br>%{customdata}<extra></extra>"))
    fig.update_layout(**nl_layout(height=340, margin=dict(l=8, r=8, t=8, b=8)))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
    st.caption(
        "Each layer is a stage of the pipeline, from every stock we track "
        "at the top down to trades actually filled at the broker; the "
        "number is how many made it that far, with the percent of the "
        "starting universe beside it. The width shrinks at each stage "
        "because the system throws out most candidates — that narrowing "
        "is by design, not a problem. A sharp collapse between two layers "
        "shows where the strictest filter sits: most symbols never become "
        "recommendations at all.")

    _EXIT_LABEL = {"stopped_out": "STOP", "target_hit": "TARGET",
                   "manual_override": "MANUAL"}
    recs_num = pd.DataFrame()
    if not ledger_all.empty:
        last_close = db_query("""
            SELECT b.symbol, b.close FROM mm_bars b
            JOIN (SELECT symbol, MAX(date) AS d FROM mm_bars GROUP BY symbol) m
              ON m.symbol = b.symbol AND m.d = b.date
        """)
        _close_of = dict(zip(last_close["symbol"], last_close["close"]))
        pmap = perf_all.set_index("rec_id")

        _rows = []
        for r in ledger_all.itertuples():
            p = pmap.loc[r.rec_id] if r.rec_id in pmap.index else None
            ret, kind, outcome, days = None, "—", "OPEN", None
            if p is not None:
                days = p["days_held"]
                if pd.notna(p["exit_reason"]):
                    outcome = str(p["exit_reason"]).upper()
                if r.status == "approved" and pd.notna(p["return_pct_actual"]):
                    ret, kind = float(p["return_pct_actual"]), "actual"
                elif pd.notna(p["return_pct_counterfactual"]):
                    ret, kind = float(p["return_pct_counterfactual"]), "counterfactual"
            entry = (r.actual_fill_price
                     if pd.notna(r.actual_fill_price) else r.entry_est_price)
            if ret is None and r.exit_status in _EXIT_LABEL:
                outcome = _EXIT_LABEL[r.exit_status]
                if entry and pd.notna(r.exit_price) and float(entry) > 0:
                    ret, kind = float(r.exit_price) / float(entry) - 1.0, "actual"
            if ret is None:
                close = _close_of.get(r.symbol)
                if entry and close and float(entry) > 0:
                    ret, kind = float(close) / float(entry) - 1.0, "mark"
            _rows.append({
                "entry": r.entry_date, "symbol": r.symbol,
                "sector": r.sector if pd.notna(r.sector) else "Other",
                "strategy": r.strategy_name, "decision": r.status,
                "outcome": outcome, "kind": kind, "ret": ret, "days": days,
                "entry_px": (float(entry) if entry is not None
                             and pd.notna(entry) and float(entry) > 0
                             else None),
            })
        recs_num = pd.DataFrame(_rows)

    def _sdol(v: float) -> str:
        return ("−" if v < 0 else "+") + f"${abs(v):,.0f}"

    _CAPITAL = 100_000.0
    _live = (recs_num[recs_num["ret"].notna()]
             if not recs_num.empty else pd.DataFrame())
    if not _live.empty:
        _psyms = sorted(_live["symbol"].unique())
        _ph = ",".join("?" * len(_psyms))
        _pxw = db_query(
            f"SELECT symbol, date, close FROM mm_bars "
            f"WHERE symbol IN ({_ph}) AND date >= ? ORDER BY date",
            tuple(_psyms) + (str(_live["entry"].min()),)
        ).pivot(index="date", columns="symbol", values="close").ffill()
        _days_idx = list(_pxw.index)

    def _sim_book(subset: pd.DataFrame) -> dict | None:
        spans = []
        for r in subset.itertuples():
            if r.symbol not in _pxw.columns:
                continue
            col = _pxw[r.symbol]
            start = next((i for i, d in enumerate(_days_idx)
                          if d >= r.entry and pd.notna(col.iloc[i])), None)
            if start is None:
                continue
            if r.kind in ("actual", "counterfactual") and pd.notna(r.days):
                stop, snap = min(start + int(r.days), len(_days_idx) - 1), True
            else:
                stop, snap = len(_days_idx) - 1, r.kind != "mark"
            spans.append((r, start, stop, snap))
        if not spans:
            return None
        open_ct = [0] * len(_days_idx)
        for _, s, e, _snap in spans:
            for i in range(s, e + 1):
                open_ct[i] += 1
        max_conc = max(open_ct)
        size = _CAPITAL / max_conc
        contrib = pd.DataFrame(
            float("nan"), index=sorted({r.symbol for r, *_ in spans}),
            columns=_days_idx, dtype=float)
        for r, s, e, snap in spans:
            closes = _pxw[r.symbol].iloc[s:e + 1].astype(float)
            basis = (float(r.entry_px) if pd.notna(r.entry_px)
                     else float(closes.iloc[0]))
            vals = (size / basis) * closes
            if snap:
                vals.iloc[-1] = size * (1.0 + r.ret)
            chg = vals.diff()
            chg.iloc[0] = vals.iloc[0] - size
            seg = contrib.loc[r.symbol, closes.index]
            contrib.loc[r.symbol, closes.index] = (
                seg.fillna(0.0).values + chg.values)
        return {"cum": contrib.fillna(0.0).sum(axis=0).cumsum(),
                "size": size, "max_conc": max_conc,
                "spans": spans, "contrib": contrib}

    st.markdown('<div class="nl-kicker" style="margin-top:24px;">02 · Shadow vs actual</div>'
                '<div class="nl-h4">Three $100,000 accounts — everything, '
                'approved only, rejected only</div>',
                unsafe_allow_html=True)
    _sims = {}
    if _live.empty:
        st.info("No measurable recommendations yet — a pick gets one once "
                "it is filled, simulated, or marked against a close.")
    else:
        _books = [
            ("Shadow book — every recommendation", None, ACCENT, "circle"),
            ("Actual book — approved only", "approved", GAIN, "square"),
            ("Rejected only", "rejected", LOSS, "x"),
        ]
        fig = go.Figure()
        for name, dec, color, marker in _books:
            sub = _live if dec is None else _live[_live["decision"] == dec]
            sim = _sim_book(sub)
            if sim is None:
                continue
            _sims[name] = sim
            cum = sim["cum"]
            v = float(cum.iloc[-1])
            fig.add_trace(go.Scatter(
                x=list(cum.index), y=list(cum.values), mode="lines+markers",
                name=f"{name} · {money(sim['size'])}/position",
                line=dict(color=color, width=2),
                marker=dict(symbol=marker, size=5),
                customdata=[_sdol(x) for x in cum.values],
                hovertemplate="%{x} · %{customdata}<extra>" + name
                              + "</extra>"))
            fig.add_annotation(
                x=cum.index[-1], y=v,
                text=f"{_sdol(v)} · {pctf(v / _CAPITAL * 100.0, 2)}",
                showarrow=False, xanchor="left", xshift=8,
                font=dict(size=12, color=color))
        fig.add_hline(y=0, line_width=1, line_color="rgba(20,20,20,0.30)")
        fig.update_layout(**nl_layout(
            height=380, margin=dict(l=60, r=150, t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=nl_axis(tickprefix="$"), xaxis=nl_axis(type="category")))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
        st.caption(
            "Three simulated \\$100,000 accounts trading the same book "
            "through different filters, marked to market from daily closes "
            "with closed trades settled at their bracket exit: blue takes "
            "every recommendation, green only what was approved, "
            "red only what was rejected. Each account gives every position "
            "an equal slice of capital, sized so its own busiest day still "
            "fits — the accounts hold different numbers of positions, so "
            "their position sizes differ (see the legend). The figure at "
            "each line's end is real money: what that account would "
            "actually be up or down on \\$100,000 today.")

    st.markdown('<div class="nl-kicker" style="margin-top:24px;">03 · Stocks</div>'
                '<div class="nl-h4">What each stock would have generated '
                '<span class="nl-muted" style="font-weight:400;font-size:14px;">'
                '· all recommendations since trading began, summed per stock</span></div>',
                unsafe_allow_html=True)
    _meas = (recs_num[recs_num["ret"].notna()]
             if not recs_num.empty else pd.DataFrame())
    if _meas.empty:
        st.info("No measurable returns yet — recommendations need a fill, a "
                "counterfactual close, or at least a bar to mark against.")
    else:
        per_stock = (_meas.groupby("symbol")
                     .agg(sector=("sector", "first"),
                          first_entry=("entry", "min"),
                          n_recs=("ret", "size"),
                          total=("ret", "sum"),
                          marked=("kind", lambda k: bool((k == "mark").any())))
                     .reset_index())
        per_stock["total"] *= 100.0

        per_stock["plot_x"] = pd.to_datetime(per_stock["first_entry"])
        for _, idx in per_stock.groupby(
                ["first_entry", per_stock["total"].round(0)]).groups.items():
            if len(idx) > 1:
                for j, i in enumerate(sorted(idx)):
                    per_stock.loc[i, "plot_x"] += pd.Timedelta(
                        hours=(j - (len(idx) - 1) / 2) * 14)

        fig = go.Figure()
        for sec in [s for s in SECTOR_COLORS if s in set(per_stock["sector"])]:
            g = per_stock[per_stock["sector"] == sec]
            color = SECTOR_COLORS[sec]
            fig.add_trace(go.Scatter(
                x=list(g["plot_x"]), y=list(g["total"]),
                mode="markers", name=sec,
                marker=dict(
                    size=11, color=color,
                    symbol=["circle-open" if m else "circle" for m in g["marked"]],
                    line=dict(width=2,
                              color=[color if m else "#ffffff"
                                     for m in g["marked"]])),
                customdata=list(zip(
                    g["symbol"], g["n_recs"],
                    ["marked to latest close" if m else "resolved"
                     for m in g["marked"]], g["first_entry"],
                    [pctf(v, 2) for v in g["total"]])),
                hovertemplate=("<b>%{customdata[0]}</b> · " + sec +
                               "<br>%{customdata[4]} over %{customdata[1]} rec(s)"
                               "<br>first recommended %{customdata[3]} · "
                               "%{customdata[2]}<extra></extra>")))
        _ext = pd.concat([per_stock.nlargest(3, "total"),
                          per_stock.nsmallest(3, "total")]).drop_duplicates("symbol")
        _placed = []
        for e in _ext.itertuples():
            clash = any(abs((e.plot_x - px).days) < 2 and abs(e.total - py) < 5
                        for px, py in _placed)
            if not clash:
                _placed.append((e.plot_x, e.total))
            fig.add_annotation(
                x=e.plot_x, y=e.total, text=e.symbol, showarrow=False,
                yanchor="top" if clash else "bottom",
                yshift=-9 if clash else 9,
                font=dict(size=11, color=SECTOR_COLORS.get(e.sector,
                                                           SECTOR_COLORS["Other"])))
        fig.add_hline(y=0, line_width=1, line_color="rgba(20,20,20,0.30)")
        fig.update_layout(**nl_layout(
            height=420, margin=dict(l=56, r=20, t=24, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=nl_axis(ticksuffix="%", range=[-50, 50], title=dict(
                text="total shadow return", font=dict(size=12))),
            xaxis=nl_axis(title=dict(text="first recommended",
                                     font=dict(size=12)))))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
        st.caption(
            "One dot per stock: how far right it sits is the day the system "
            "first recommended it, and its height is the total return all of "
            "its recommendations would have generated — actual fills where "
            "they exist, the counterfactual bracket otherwise, summed (not "
            "compounded) when a stock was picked more than once. Solid dots "
            "are fully resolved; hollow dots still carry at least one open "
            "pick marked to the latest close, so their height can move. "
            "Color is the stock's sector — a cluster of one color above or "
            "below the line says the system's edge (or its misses) is "
            "concentrated in that sector rather than spread evenly.")

    st.markdown('<div class="nl-kicker" style="margin-top:24px;">04 · One stock</div>'
                '<div class="nl-h4">P&amp;L on $10,000 invested at first '
                'recommendation</div>',
                unsafe_allow_html=True)
    if _meas.empty:
        st.info("Needs at least one recommendation with a measurable return.")
    else:
        _sec_of_stock = dict(zip(per_stock["symbol"], per_stock["sector"]))
        _default_stock = str(
            per_stock.loc[per_stock["total"].idxmax(), "symbol"])
        psym = sector_symbol_picker(
            sorted(per_stock["symbol"]), _sec_of_stock,
            prefix="shadowpnl", default_sym=_default_stock)

        _lrecs = (ledger_all[ledger_all["symbol"] == psym]
                  .sort_values("entry_date"))
        _first = _lrecs.iloc[0]
        entry_px = (_first.actual_fill_price
                    if pd.notna(_first.actual_fill_price)
                    else _first.entry_est_price)
        bars_px = db_query(
            "SELECT date, close FROM mm_bars WHERE symbol = ? AND date >= ? "
            "ORDER BY date", (psym, str(_first.entry_date)))
        if not (entry_px and pd.notna(entry_px) and float(entry_px) > 0):
            entry_px = (float(bars_px["close"].iloc[0])
                        if not bars_px.empty else None)
        if bars_px.empty or not entry_px:
            st.info(f"No bars on file for {psym} since {_first.entry_date}.")
        else:
            shares = 10000.0 / float(entry_px)
            pnl = shares * bars_px["close"].astype(float) - 10000.0
            daily = pnl.diff()
            daily.iloc[0] = pnl.iloc[0]
            last_pnl = float(pnl.iloc[-1])

            kpi_strip([
                ("First recommended", str(_first.entry_date), INK,
                 f"{len(_lrecs)} rec(s) all-time · {_first.strategy_name}",
                 MUTED),
                ("Entry price", f"${float(entry_px):,.2f}", INK,
                 f"{shares:,.1f} shares on $10,000", MUTED),
                ("Position value", money(10000.0 + last_pnl), INK,
                 f"marked at the {bars_px['date'].iloc[-1]} close", MUTED),
                ("P&L", _sdol(last_pnl),
                 GAINT if last_pnl >= 0 else LOSST,
                 pctf(last_pnl / 100.0, 2) + " on cost", MUTED),
            ])

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=bars_px["date"], y=daily, name="daily change",
                marker=dict(color=[GAIN if v >= 0 else LOSS for v in daily],
                            opacity=0.55),
                customdata=[_sdol(v) for v in daily],
                hovertemplate="%{x} · %{customdata} on the day"
                              "<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=bars_px["date"], y=pnl, mode="lines",
                name="cumulative P&L", line=dict(color=INK, width=2),
                customdata=[_sdol(v) for v in pnl],
                hovertemplate="%{x} · %{customdata} since entry"
                              "<extra></extra>"))
            for rd in _lrecs["entry_date"].tolist()[1:]:
                fig.add_vline(x=rd, line_dash="dot", line_width=1,
                              line_color="rgba(27,95,168,0.5)")
            fig.add_annotation(
                x=bars_px["date"].iloc[-1], y=last_pnl, text=_sdol(last_pnl),
                showarrow=False, xanchor="left", xshift=8,
                font=dict(size=12,
                          color=GAINT if last_pnl >= 0 else LOSST))
            fig.add_hline(y=0, line_width=1,
                          line_color="rgba(20,20,20,0.30)")
            fig.update_layout(**nl_layout(
                height=380, margin=dict(l=56, r=84, t=10, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                yaxis=nl_axis(tickprefix="$", title=dict(
                    text=f"P&L on $10,000 in {psym}", font=dict(size=12))),
                xaxis=nl_axis()))
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
            st.caption(
                f"Imagine putting $10,000 into {psym} at its estimated entry "
                "price the first day the system recommended it, and simply "
                "holding to the latest close. The bars are how many dollars "
                "the position gained or lost each day; the black line adds "
                "them up into the running profit or loss. Dotted blue lines "
                "mark days the system recommended the same stock again. "
                "Note this is a buy-and-hold view — the scatter above uses "
                "the bracket simulation instead, which exits at the stop or "
                "target, so the two can legitimately disagree.")

    st.markdown('<div class="nl-kicker" style="margin-top:24px;">05 · The whole book</div>'
                '<div class="nl-h4">A $100,000 shadow portfolio — who is in '
                'the book, when, and what each day paid</div>',
                unsafe_allow_html=True)
    _shadow_sim = _sims.get("Shadow book — every recommendation")
    if _shadow_sim is None:
        st.info("No measurable recommendations yet.")
    else:
        _size = _shadow_sim["size"]
        _max_conc = _shadow_sim["max_conc"]
        _spans = _shadow_sim["spans"]
        _cum_tot = _shadow_sim["cum"]
        _total = float(_cum_tot.iloc[-1])
        _order = [s for s in (_live.groupby("symbol")["entry"].min()
                              .sort_values().index)
                  if s in set(_shadow_sim["contrib"].index)]
        _contrib = _shadow_sim["contrib"].loc[_order]
        _open_now = sum(1 for _, s, e, snap in _spans
                        if e == len(_days_idx) - 1 and not snap)
        _sec_of = dict(zip(recs_num["symbol"], recs_num["sector"]))

        kpi_strip([
            ("Portfolio", money(_CAPITAL), INK,
             "shadow capital · every recommendation taken", MUTED),
            ("Position size", money(_size), INK,
             f"equal size · {_max_conc} max concurrent positions", MUTED),
            ("Positions", f"{len(_spans)} recs · {len(_order)} stocks", INK,
             f"{_open_now} still open, marked to the latest close", MUTED),
            ("Total P&L", _sdol(_total),
             GAINT if _total >= 0 else LOSST,
             pctf(_total / _CAPITAL * 100.0, 2) + " on the portfolio", MUTED),
        ])

        figA = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.26, 0.74], vertical_spacing=0.06)
        figA.add_trace(go.Scatter(
            x=_days_idx, y=list(_cum_tot.values), mode="lines",
            name="portfolio cumulative P&L", showlegend=False,
            line=dict(color=ACCENT, width=2),
            customdata=[_sdol(v) for v in _cum_tot.values],
            hovertemplate="%{x} · %{customdata} cumulative<extra></extra>"),
            row=1, col=1)
        figA.add_annotation(
            x=_days_idx[-1], y=_total, text=_sdol(_total), showarrow=False,
            xanchor="left", xshift=8,
            font=dict(size=12, color=GAINT if _total >= 0 else LOSST),
            row=1, col=1)
        figA.add_hline(y=0, line_width=1,
                       line_color="rgba(20,20,20,0.30)", row=1, col=1)
        _zmax = max(float(_contrib.abs().max().max()), 1.0)
        _hov = [[("" if pd.isna(v) else
                  f"<b>{sym}</b> · {d} · {_sdol(v)} on the day")
                 for d, v in zip(_days_idx, _contrib.loc[sym])]
                for sym in _contrib.index]
        figA.add_trace(go.Heatmap(
            z=_contrib.values, x=_days_idx, y=list(_contrib.index),
            colorscale=[[0, LOSS], [0.5, NEU3], [1, GAIN]],
            zmin=-_zmax, zmax=_zmax, hoverongaps=False, xgap=1, ygap=1,
            customdata=_hov, hovertemplate="%{customdata}<extra></extra>",
            colorbar=dict(thickness=12, outlinewidth=0, tickprefix="$",
                          tickfont=dict(size=11), len=0.66, y=0.30)),
            row=2, col=1)
        figA.update_layout(**nl_layout(
            height=220 + 24 * len(_order),
            margin=dict(l=60, r=80, t=10, b=40)))
        figA.update_xaxes(**nl_axis(type="category"), row=1, col=1)
        figA.update_xaxes(**nl_axis(type="category"), row=2, col=1)
        figA.update_yaxes(**nl_axis(tickprefix="$"), row=1, col=1)
        figA.update_yaxes(**nl_axis(autorange="reversed",
                                    tickfont=dict(size=10)), row=2, col=1)
        st.plotly_chart(figA, use_container_width=True, config=PLOTLY_CFG)
        st.caption(
            "Each row is a stock, each column a trading day. A row begins "
            "the day the stock entered the book and stops when its bracket "
            "exit closed the position; blank cells mean the stock wasn't "
            "held that day. Green cells are days the position added "
            "dollars, red cells days it gave them back, and the blue line "
            "on top adds every cell up into the portfolio's running P&L. "
            f"All positions are the same size (\\{money(_size)}) — "
            "\\$100,000 divided by the most positions ever open at once — "
            "so no stock dominates by construction; a deeply colored row "
            "is genuinely a big mover.")

        st.markdown('<div class="nl-kicker" style="margin-top:18px;'
                    'text-align:center;">Same data · stacked daily bars</div>',
                    unsafe_allow_html=True)
        figB = go.Figure()
        _seen_sec = set()
        for sym in _contrib.index:
            sec = _sec_of.get(sym, "Other")
            color = SECTOR_COLORS.get(sec, SECTOR_COLORS["Other"])
            vals = _contrib.loc[sym]
            mask = vals.notna()
            figB.add_trace(go.Bar(
                x=[d for d, m in zip(_days_idx, mask) if m],
                y=[v for v, m in zip(vals, mask) if m],
                name=sec, legendgroup=sec,
                showlegend=sec not in _seen_sec,
                marker=dict(color=color, opacity=0.8,
                            line=dict(width=0.5, color="#ffffff")),
                customdata=[f"<b>{sym}</b> · {_sdol(v)}"
                            for v, m in zip(vals, mask) if m],
                hovertemplate="%{x} · %{customdata} on the day<extra>"
                              + sec + "</extra>"))
            _seen_sec.add(sec)
        figB.add_trace(go.Scatter(
            x=_days_idx, y=list(_cum_tot.values), mode="lines",
            name="cumulative P&L", line=dict(color=INK, width=2),
            customdata=[_sdol(v) for v in _cum_tot.values],
            hovertemplate="%{x} · %{customdata} cumulative<extra></extra>"))
        figB.add_hline(y=0, line_width=1, line_color="rgba(20,20,20,0.30)")
        figB.update_layout(**nl_layout(
            height=420, barmode="relative",
            margin=dict(l=60, r=20, t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis=nl_axis(tickprefix="$"),
            xaxis=nl_axis(type="category")))
        st.plotly_chart(figB, use_container_width=True, config=PLOTLY_CFG)
        st.caption(
            "The same matrix, stacked. Each day's bar is assembled from "
            "every open position's dollar move that day — sector-colored "
            "slices above zero are the day's winners, below zero its "
            "losers, and hovering a slice names the stock. The black line "
            "is the identical running total as the blue line above; when "
            "the bars below zero outweigh those above, it bends down.")

    st.markdown('<div class="nl-kicker" style="margin-top:24px;">06 · Price ranges</div>'
                '<div class="nl-h4">How far each stock travelled since '
                'entry</div>',
                unsafe_allow_html=True)
    if _live.empty:
        st.info("No measurable recommendations yet.")
    else:
        _rng_syms = [s for s in _live["symbol"].unique()
                     if s in _pxw.columns]
        _ph2 = ",".join("?" * len(_rng_syms))
        _ohlc = db_query(
            f"SELECT symbol, date, low, high, close FROM mm_bars "
            f"WHERE symbol IN ({_ph2}) AND date >= ? ORDER BY date",
            tuple(_rng_syms) + (str(_live["entry"].min()),))
        _rrows = []
        for sym in _rng_syms:
            first = (_live[_live["symbol"] == sym]
                     .sort_values("entry").iloc[0])
            g = _ohlc[(_ohlc["symbol"] == sym)
                      & (_ohlc["date"] >= first["entry"])]
            if g.empty:
                continue
            basis = (float(first["entry_px"]) if pd.notna(first["entry_px"])
                     else float(g["close"].iloc[0]))
            lo, hi = float(g["low"].min()), float(g["high"].max())
            last = float(g["close"].iloc[-1])
            _rrows.append({
                "symbol": sym, "entry_date": first["entry"], "basis": basis,
                "lo": lo, "hi": hi, "last": last,
                "lo_pct": (lo / basis - 1.0) * 100.0,
                "hi_pct": (hi / basis - 1.0) * 100.0,
                "ret_pct": (last / basis - 1.0) * 100.0,
            })
        rng = pd.DataFrame(_rrows).sort_values("ret_pct")
        fig = go.Figure()
        for r in rng.itertuples():
            fig.add_trace(go.Scatter(
                x=[r.lo_pct, r.hi_pct], y=[r.symbol, r.symbol],
                mode="lines", line=dict(color="rgba(20,20,20,0.25)", width=2),
                showlegend=False, hoverinfo="skip"))
        _mk = [
            ("lowest since entry", "lo_pct", LOSS, "circle",
             lambda r: f"<b>{r.symbol}</b> · lowest ${r.lo:,.2f} · "
                       f"{pctf(r.lo_pct, 2)} from entry ${r.basis:,.2f}"),
            ("highest since entry", "hi_pct", GAIN, "circle",
             lambda r: f"<b>{r.symbol}</b> · highest ${r.hi:,.2f} · "
                       f"{pctf(r.hi_pct, 2)} from entry ${r.basis:,.2f}"),
            ("latest close", "ret_pct", INK, "diamond",
             lambda r: f"<b>{r.symbol}</b> · last close ${r.last:,.2f} · "
                       f"{pctf(r.ret_pct, 2)} from entry ${r.basis:,.2f} · "
                       f"entered {r.entry_date}"),
        ]
        for name, colname, color, symb, hovfn in _mk:
            fig.add_trace(go.Scatter(
                x=list(rng[colname]), y=list(rng["symbol"]), mode="markers",
                name=name,
                marker=dict(symbol=symb, size=9, color=color,
                            line=dict(width=1.5, color="#ffffff")),
                customdata=[hovfn(r) for r in rng.itertuples()],
                hovertemplate="%{customdata}<extra></extra>"))
        for r in rng.itertuples():
            fig.add_annotation(
                x=r.hi_pct, y=r.symbol, text=pctf(r.ret_pct, 2),
                showarrow=False, xanchor="left", xshift=10,
                font=dict(size=11,
                          color=GAINT if r.ret_pct >= 0 else LOSST))
        fig.add_vline(x=0, line_dash="dot", line_width=1.4,
                      line_color="rgba(20,20,20,0.45)",
                      annotation_text="entry price",
                      annotation_font=dict(size=11, color=MUTED),
                      annotation_position="top")
        fig.update_layout(**nl_layout(
            height=150 + 24 * len(rng),
            margin=dict(l=60, r=80, t=28, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=nl_axis(ticksuffix="%", title=dict(
                text="distance from entry price", font=dict(size=12))),
            yaxis=nl_axis(categoryorder="array",
                          categoryarray=list(rng["symbol"]),
                          tickfont=dict(size=11))))
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CFG)
        st.caption(
            "One line per stock, anchored at its entry price (the dotted "
            "zero line). The red dot is the lowest low and the green dot "
            "the highest high the stock has touched since it was first "
            "recommended, as a percent of the entry price; the black "
            "diamond is the latest close, and the figure at the end of "
            "each line is that same latest-close return — the label the "
            "row is sorted by. Hover any dot for the actual dollar "
            "prices. Stocks recommended more than once are anchored at "
            "their first entry, and the range keeps running to today even "
            "after a bracket exit, so a stopped-out stock can still show "
            "a tall green dot — the move it made after the system was "
            "out.")


with tab_briefing:
    account_strip()
    st.header("Daily briefing — deepseek")
    st.caption(
        "Written by deepseek-r1:14b (Ollama, laptop-local) from the numbers in "
        "this dashboard: positions, picks, Kronos forecasts, sentiment, "
        "divergences. Advisory prose only — it reads state, it cannot act. "
        "Every briefing stores the exact numbers it was given (audit trail).")

    briefs = db_query(
        "SELECT as_of_date, generated_at, model, commentary_md, context_json "
        "FROM mm_llm_commentary WHERE kind = 'daily_briefing' "
        "ORDER BY generated_at DESC LIMIT 30")

    bcol1, bcol2 = st.columns([3, 1])
    with bcol2:
        if st.button("Generate now", key="briefing_generate"):
            import subprocess
            with st.spinner("deepseek is writing (1–3 min on first model load)…"):
                proc = subprocess.run(
                    [str(PROJ_ROOT / "dl_venv" / "bin" / "python"),
                     "-m", "macmini.llm.commentary", "--generate"],
                    cwd=PROJ_ROOT, capture_output=True, text=True, timeout=600,
                    env={**os.environ, "PYTHONPATH": str(PROJ_ROOT / "src")})
            if proc.returncode == 0:
                db_query.clear()
                st.rerun()
            else:
                st.error(f"generation failed: {proc.stderr[-500:]}")

    if briefs.empty:
        st.info("No briefings yet — press Generate now, or wait for the nightly run.")
    else:
        with bcol1:
            options = [f"{r.as_of_date} · {r.generated_at[11:16]}Z"
                       for r in briefs.itertuples()]
            pick_i = st.selectbox("Briefing", range(len(options)),
                                  format_func=lambda i: options[i],
                                  key="briefing_pick")
        b = briefs.iloc[pick_i]
        st.markdown(f"*{b['model']} · generated {b['generated_at']}*")
        st.markdown("---")
        st.markdown(b["commentary_md"])
        if b["context_json"]:
            with st.expander("Numbers the model was given (audit trail)"):
                st.json(b["context_json"])


with tab_policy:
    account_strip()
    st.header("Policy baseline")
    _policy_path = PROJ_ROOT / "POLICY.md"
    if _policy_path.exists():
        st.caption(
            "Trading-policy document.")
        st.markdown(_policy_path.read_text())
    else:
        st.warning("POLICY.md not found at repo root.")


st.markdown(
    f"""
<div class="nl-muted" style="margin-top:56px;padding-top:16px;border-top:1px solid {DIVIDER};font-size:11px;">
Vicuna monitor v2 · built {date.today().isoformat()} · broker: Alpaca paper · read-only · long-only swing system · $100K paper account
</div>
""",
    unsafe_allow_html=True,
)
