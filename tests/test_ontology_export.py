""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ / "src"))

from macmini.data import schema
from macmini.ontology import export as ox

N_PAST = 300
STOP = 0.08


class _Strat:
    ""

    id = 14
    name = "Internal Bar Strength (IBS < 0.2 in uptrend)"
    category = "mean_reversion"
    timeframe = "swing"
    data_required = ("mm_bars",)


class _Strat2(_Strat):
    ""

    id = 55
    name = "Calendar / Seasonality Composite"
    category = "composite"


REGISTRY = {14: _Strat, 55: _Strat2}


def _weekdays(start: date, n: int) -> list[str]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


DAYS = _weekdays(date(2025, 1, 1), N_PAST + 60)
AS_OF = DAYS[N_PAST - 1]
PAST = [100.0 * (1.001 ** i) for i in range(N_PAST)]


def _future(kind: str, n: int) -> list[float]:
    last = PAST[-1]
    if kind == "up":
        return [last * (1.01 ** (i + 1)) for i in range(n)]
    if kind == "down":
        return [last * (0.99 ** (i + 1)) for i in range(n)]
    if kind == "dip":
        return [last * 0.98 for _ in range(n)]
    raise ValueError(kind)


def _insert_bars(c, sym: str, closes: list[float], time_suffix: bool = False) -> None:
    for d, px in zip(DAYS, closes):
        stamp = f"{d}T00:00:00-04:00" if time_suffix else d
        c.execute(
            "INSERT INTO mm_bars (symbol, date, open, high, low, close, adj_close, "
            "volume, source) VALUES (?, ?, ?, ?, ?, ?, ?, 1000, 'test')",
            (sym, stamp, px, px * 1.005, px * 0.995, px, px))


def _insert_rec(c, rec_id: str, sym: str, as_of: str = AS_OF, sid: int = 14,
                rank: int = 1, stop: float | None = STOP, **kw) -> None:
    row = {
        "rec_id": rec_id, "as_of_date": as_of, "entry_date": as_of, "rank": rank,
        "symbol": sym, "strategy_id": sid, "strategy_name": "IBS",
        "sector": "Other", "suggested_size_pct": 0.05,
        "suggested_stop_loss_pct": stop, "suggested_target_pct": 0.16,
        "rationale": 'IBS 0.12 | "quoted": text', "status": "approved",
        "created_at": "x", "updated_at": "x",
    }
    row.update(kw)
    cols = ", ".join(row)
    c.execute(f"INSERT INTO mm_recs_ledger ({cols}) VALUES ({', '.join('?' for _ in row)})",
              tuple(row.values()))


def _make_db(path: Path, up_future: str = "up") -> Path:
    with sqlite3.connect(path) as c:
        for name in ("MM_BARS", "MM_SECTOR_MAP", "MM_LEADERBOARD",
                     "MM_RECS_LEDGER", "MM_TITAN_NOTES"):
            c.execute(getattr(schema, name))
        _insert_bars(c, "UPP", PAST + _future(up_future, 60))
        _insert_bars(c, "DWN", PAST + _future("down", 60))
        _insert_bars(c, "DIP", PAST + _future("dip", 60))
        _insert_bars(c, "NEW", PAST + _future("up", 5), time_suffix=True)
        _insert_bars(c, "MOG.A", PAST + _future("up", 60))
        c.execute("INSERT INTO mm_sector_map (symbol, sector, finnhub_sector, "
                  "market_cap_usd, archetype) VALUES ('UPP', 'Software', 'Technology', 1e9, 'x')")
        c.execute("INSERT INTO mm_leaderboard (strategy_id, symbol, sector, last_run_id, "
                  "qualifies, updated_at) VALUES (14, 'UPP', 'Software', 'bt_test', 1, 'x')")
        c.execute("INSERT INTO mm_leaderboard (strategy_id, symbol, sector, last_run_id, "
                  "qualifies, updated_at) VALUES (55, 'DWN', 'Other', 'bt_test', 0, 'x')")
        _insert_rec(c, "aaaaaaaa11111111", "UPP", rank=1,
                    fill_status="filled", actual_fill_price=100.0,
                    actual_fill_at="2026-03-02T14:30:00+00:00", fill_qty=10,
                    stop_price=92.0, target_price=116.0, exit_status="target_hit",
                    exit_price=116.0, exit_at="2026-03-12T15:00:00Z", exit_qty=10)
        _insert_rec(c, "bbbbbbbb22222222", "DWN", rank=2)
        _insert_rec(c, "cccccccc33333333", "NEW", rank=3)
        _insert_rec(c, "dddddddd44444444", "DIP", rank=4, sid=55)
        _insert_rec(c, "eeeeeeee55555555", "MOG.A", rank=5)
        _insert_rec(c, "ffffffff66666666", "OUT", rank=6, sid=0)
        _insert_rec(c, "fixture-0001", "UPP", rank=7)
        for metric, text in (("summary", "Short."), ("detail", "Longer | piped."),
                             ("wrong_if", "If it falls.")):
            c.execute("INSERT INTO mm_titan_notes (as_of_date, symbol, section, metric, "
                      "source_column, narrative, created_at) VALUES (?, 'UPP', 'narrative', "
                      "?, 'model=template', ?, 'x')", (AS_OF, metric, text))
        c.execute("INSERT INTO mm_titan_notes (as_of_date, symbol, section, metric, value, "
                  "unit, source_table, source_column, source_asof, created_at) VALUES "
                  "(?, 'UPP', 'price', 'close', 12.5, 'USD/share', 'mm_bars', 'close', ?, 'x')",
                  (AS_OF, AS_OF))
        c.commit()
    return path


POLICY = """# Policy

## Rules

- **not a change** just a bullet

## Change history

- **2026-09-02 — v1.** Baseline adopted. All prior rules void.
  Second line of the same entry.
- **2026-09-29 — v1.3.** Adopted: "quoted" text
  with a colon: and more.

## Appendix

- **2026-01-01 — v9.** not part of the history
"""


@pytest.fixture()
def env(tmp_path) -> dict:
    (tmp_path / "symbols.csv").write_text("symbol\nUPP\nDWN\nDIP\nNEW\nMOG.A\nGONE\n")
    (tmp_path / "POLICY.md").write_text(POLICY)
    return {
        "db_path": _make_db(tmp_path / "scratch.db"),
        "vault": tmp_path / "vault" / "Vicuna",
        "data_dir": tmp_path / "data",
        "symbols_csv": tmp_path / "symbols.csv",
        "policy_path": tmp_path / "POLICY.md",
        "registry": REGISTRY,
        "project_root": tmp_path / "repo",
        "memory_as_of": date(2026, 12, 31),
    }


def _run(env: dict, **kw) -> dict:
    return ox.export(**{**env, "generated_at": "2026-01-01T00:00:00Z", **kw})


def _jsonl(env: dict, name: str) -> list[dict]:
    text = (env["data_dir"] / name).read_text()
    return [json.loads(x) for x in text.splitlines() if x]


def _recs(env: dict) -> dict[str, dict]:
    return {r["ticker"]: r for r in _jsonl(env, "recommendations.jsonl")}


def _snapshot(root: Path, skip: tuple[str, ...] = ()) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file() and p.name not in skip}


def _frontmatter(path: Path) -> dict:
    yaml = pytest.importorskip("yaml")
    text = path.read_text()
    assert text.startswith("---\n")
    return yaml.safe_load(text.split("\n---\n", 1)[0][4:])


def test_features_ignore_bars_after_as_of(env, tmp_path):
    _run(env)
    up = _recs(env)["UPP"]

    other = dict(env, db_path=_make_db(tmp_path / "other.db", up_future="down"),
                 vault=tmp_path / "vault2" / "Vicuna", data_dir=tmp_path / "data2")
    _run(other)
    down = _recs(other)["UPP"]

    for key in ("feature_bar_date", *ox.FEATURE_KEYS):
        assert up[key] == down[key], key
        assert up[key] is not None, key
    assert up["feature_bar_date"] == AS_OF
    assert up["r20"] > 0 > down["r20"]


def test_feature_values_match_hand_math(env):
    _run(env)
    r = _recs(env)["UPP"]
    assert r["close"] == pytest.approx(PAST[-1], abs=1e-5)
    assert r["mom21"] == pytest.approx(1.001 ** 21 - 1, abs=1e-6)
    assert r["mom63"] == pytest.approx(1.001 ** 63 - 1, abs=1e-6)
    assert r["mom252"] == pytest.approx(1.001 ** 252 - 1, abs=1e-6)
    assert r["off_252d_high"] == pytest.approx(0.0, abs=1e-9)
    assert r["vol60"] == pytest.approx(0.0, abs=1e-6)
    sma50 = sum(PAST[-50:]) / 50
    assert r["close_vs_sma50"] == pytest.approx(PAST[-1] / sma50 - 1, abs=1e-6)


def test_features_need_enough_history():
    bars = ox.Bars(tuple(DAYS[:30]), *(ox.np.array(PAST[:30]),) * 4)
    f = ox.features_at(bars, DAYS[29])
    assert f["close"] == pytest.approx(PAST[29])
    assert f["mom21"] is not None
    assert f["vol60"] is None and f["close_vs_sma50"] is None
    assert f["mom63"] is None and f["off_252d_high"] is None
    assert ox.features_at(bars, "2000-01-01")["close"] is None


def test_outcomes_start_at_first_bar_after_as_of(env):
    _run(env)
    r = _recs(env)["UPP"]
    fut = _future("up", 60)
    entry = fut[0]
    assert r["outcome_entry_date"] == DAYS[N_PAST]
    assert r["outcome_entry_price"] == pytest.approx(entry, abs=1e-5)
    assert r["bars_forward"] == 60
    assert r["r5"] == pytest.approx(fut[5] / entry - 1, abs=1e-6)
    assert r["r20"] == pytest.approx(fut[20] / entry - 1, abs=1e-6)
    assert r["r40"] == pytest.approx(fut[40] / entry - 1, abs=1e-6)
    assert r["r_now"] == pytest.approx(fut[59] / entry - 1, abs=1e-6)
    assert r["r_now_date"] == DAYS[N_PAST + 59]
    assert r["mae20"] == pytest.approx(min(fut[:20]) * 0.995 / entry - 1, abs=1e-6)
    assert r["mfe20"] == pytest.approx(max(fut[:20]) * 1.005 / entry - 1, abs=1e-6)


def test_outcome_labels_follow_the_definition(env):
    _run(env)
    recs = _recs(env)
    assert recs["UPP"]["outcome_label"] == "good"
    assert recs["DWN"]["outcome_label"] == "bad_stays_down"
    assert recs["DWN"]["mae20"] <= -STOP and recs["DWN"]["r40"] < 0
    assert recs["DIP"]["outcome_label"] == "bad"
    assert recs["DIP"]["mae20"] > -STOP
    assert recs["NEW"]["outcome_label"] == "pending"
    assert recs["NEW"]["r20"] is None and recs["NEW"]["r5"] is None
    assert recs["NEW"]["bars_forward"] == 5 and recs["NEW"]["r_now"] is not None
    assert recs["OUT"]["outcome_label"] == "pending"
    assert recs["OUT"]["close"] is None


@pytest.mark.parametrize("r20,r40,r_now,mae20,stop,want", [
    (None, None, 0.05, None, 0.08, "pending"),
    (0.01, -0.5, -0.5, -0.30, 0.08, "good"),
    (0.0, -0.1, -0.1, -0.08, 0.08, "bad_stays_down"),
    (-0.05, None, -0.1, -0.09, 0.08, "bad_stays_down"),
    (-0.05, 0.02, -0.1, -0.09, 0.08, "bad"),
    (-0.05, -0.1, -0.1, -0.07, 0.08, "bad"),
    (-0.05, -0.1, -0.1, -0.30, None, "bad"),
])
def test_outcome_label_truth_table(r20, r40, r_now, mae20, stop, want):
    assert ox.outcome_label(r20, r40, r_now, mae20, stop) == want


def test_fixture_recs_are_excluded(env):
    summary = _run(env)
    ids = [r["rec_id"] for r in _jsonl(env, "recommendations.jsonl")]
    assert len(ids) == 6 == summary["counts"]["Recommendation"]
    assert not any(i.startswith("fixture") for i in ids)
    assert not any("fixture" in p.name
                   for p in (env["vault"] / "Recommendations").iterdir())
    sym = {s["ticker"]: s for s in _jsonl(env, "symbols.jsonl")}
    assert sym["UPP"]["recommendations"] == ["rec:aaaaaaaa11111111"]


def test_trade_properties(env):
    _run(env)
    (t,) = _jsonl(env, "trades.jsonl")
    assert t["id"] == "trade:aaaaaaaa11111111"
    assert t["recommendation"] == "rec:aaaaaaaa11111111"
    assert t["symbol"] == "sym:UPP" and t["strategy"] == "strat:14"
    assert t["realized_return_pct"] == pytest.approx(0.16)
    assert t["realized_pnl_usd"] == pytest.approx(160.0)
    assert t["days_held"] == 10
    assert _recs(env)["UPP"]["trade"] == "trade:aaaaaaaa11111111"
    assert _recs(env)["DWN"]["trade"] is None


def test_rerun_is_byte_identical(env):
    _run(env)
    skip = (f"{ox.INDEX_STEM}.md", "ontology.json")
    notes, data = _snapshot(env["vault"], skip), _snapshot(env["data_dir"], skip)
    first = _run(env, generated_at="2027-02-02T00:00:00Z")
    assert first["notes_written"] == 1
    assert _snapshot(env["vault"], skip) == notes
    assert _snapshot(env["data_dir"], skip) == data
    assert len(notes) > 10 and len(data) == 6

    for blob in (*notes.values(), *data.values()):
        assert b"2026-01-01T00:00:00Z" not in blob
    assert "2027-02-02T00:00:00Z" in (env["vault"] / f"{ox.INDEX_STEM}.md").read_text()
    assert json.loads((env["data_dir"] / "ontology.json").read_text())[
        "generated_at"] == "2027-02-02T00:00:00Z"


def test_stale_generated_notes_removed_handwritten_survive(env):
    _run(env)
    folder = env["vault"] / "Symbols"
    assert (folder / "GONE.md").exists()

    mine = folder / "My thoughts.md"
    mine.write_text("# mine\n\ngenerated_by: macmini.ontology.export (just text)\n")
    fm_other = folder / "Other tool.md"
    fm_other.write_text("---\ntype: Symbol\ngenerated_by: someone.else\n---\nbody\n")
    squat = folder / "DWN.md"
    squat.write_text("# my own DWN note\n")
    outside = env["vault"].parent / "Personal.md"
    outside.write_text(f"---\n{ox.MARKER_LINE}\n---\nnot in our folder\n")

    env["symbols_csv"].write_text("symbol\nUPP\nDWN\nDIP\nNEW\nMOG.A\n")
    summary = _run(env)

    assert not (folder / "GONE.md").exists()
    assert summary["notes_removed"] == 1
    assert mine.read_text().startswith("# mine")
    assert fm_other.exists()
    assert squat.read_text() == "# my own DWN note\n"
    assert summary["skipped_unmarked"] == ["Symbols/DWN.md"]
    assert outside.exists()
    assert not any(s["ticker"] == "GONE" for s in _jsonl(env, "symbols.jsonl"))


def test_refuses_a_vault_root(env, tmp_path):
    root = tmp_path / "RealVault"
    (root / ".obsidian").mkdir(parents=True)
    with pytest.raises(ValueError):
        _run(env, vault=root)
    assert sorted(p.name for p in root.iterdir()) == [".obsidian"]


def _assert_links_resolve(vault: Path, prefix: str = "") -> int:
    n = 0
    for note in sorted(vault.rglob("*.md")):
        if not ox.has_marker(note):
            continue
        for target in ox._WIKILINK.findall(note.read_text()):
            assert target.startswith(prefix), (note, target)
            assert (vault / f"{target[len(prefix):]}.md").is_file(), (note.name, target)
            n += 1
    return n


def test_wikilinks_resolve(env):
    _run(env)
    rec = env["vault"] / "Recommendations" / f"{AS_OF} UPP 14 (aaaaaaaa).md"
    targets = ox._WIKILINK.findall(rec.read_text())
    assert "Symbols/UPP" in targets
    assert "Strategies/14 - Internal Bar Strength" in targets
    assert any(t.startswith("Trades/") for t in targets)
    for t in targets:
        assert (env["vault"] / f"{t}.md").is_file(), t
    assert _assert_links_resolve(env["vault"]) > 30


def test_links_are_vault_absolute_inside_an_obsidian_vault(env, tmp_path):
    root = tmp_path / "Vault"
    (root / ".obsidian").mkdir(parents=True)
    vault = root / "10_Projects" / "Vicuna"
    summary = _run(env, vault=vault)
    assert summary["link_prefix"] == "10_Projects/Vicuna/"
    assert _assert_links_resolve(vault, "10_Projects/Vicuna/") > 30
    assert sorted(p.name for p in root.iterdir()) == [".obsidian", "10_Projects"]


def test_frontmatter_is_valid_typed_yaml(env):
    _run(env)
    fm = _frontmatter(env["vault"] / "Recommendations" / f"{AS_OF} UPP 14 (aaaaaaaa).md")
    assert fm["type"] == "Recommendation" and fm["id"] == "rec:aaaaaaaa11111111"
    assert fm["generated_by"] == ox.GENERATOR
    assert fm["rationale"] == 'IBS 0.12 | "quoted": text'
    assert fm["as_of_date"] == date.fromisoformat(AS_OF)
    assert isinstance(fm["r20"], float) and isinstance(fm["rank"], int)
    assert fm["symbol"] == "[[Symbols/UPP|UPP]]"
    assert fm["outcome_label"] == "good"

    for ticker, stem in (("UPP", f"{AS_OF} UPP 14 (aaaaaaaa)"),
                         ("DWN", f"{AS_OF} DWN 14 (bbbbbbbb)")):
        fm = _frontmatter(env["vault"] / "Recommendations" / f"{stem}.md")
        json_rec = _recs(env)[ticker]
        for prop in ox.RECOMMENDATION.props:
            got, want = fm[prop.name], json_rec[prop.name]
            if prop.type == "date" and want is not None:
                assert got == date.fromisoformat(want), prop.name
            else:
                assert got == want, prop.name

    sym = _frontmatter(env["vault"] / "Symbols" / "UPP.md")
    assert sym["qualifying_strategies"] == [
        "[[Strategies/14 - Internal Bar Strength|14 - Internal Bar Strength]]"]
    assert sym["analyst_notes"] and sym["trades"]
    assert sym["market_cap_usd"] == 1e9 and sym["in_universe"] is True
    for note in env["vault"].rglob("*.md"):
        if ox.has_marker(note):
            assert _frontmatter(note)["generated_by"] == ox.GENERATOR


def test_dotted_symbol_and_unsafe_names(env):
    _run(env)
    assert (env["vault"] / "Symbols" / "MOG-A.md").is_file()
    assert not list((env["vault"] / "Symbols").glob("MOG.A*"))
    sym = {s["ticker"]: s for s in _jsonl(env, "symbols.jsonl")}
    assert sym["MOG.A"]["id"] == "sym:MOG.A" and sym["MOG.A"]["n_bars"] == N_PAST + 60
    assert sym["NEW"]["last_close_date"] == DAYS[N_PAST + 4]
    assert (env["vault"] / "Recommendations" / f"{AS_OF} MOG-A 14 (eeeeeeee).md").is_file()
    assert (env["vault"] / "Strategies" / "55 - Calendar - Seasonality Composite.md").is_file()
    assert sym["OUT"]["in_universe"] is False
    strat = {s["strategy_id"]: s for s in _jsonl(env, "strategies.jsonl")}
    assert strat[0]["in_registry"] is False and strat[14]["in_registry"] is True
    assert strat[55]["description"] == "Second strategy / with unsafe: characters."
    for p in env["vault"].rglob("*.md"):
        assert "." not in p.stem and not set(p.stem) & set('\\/:*?"<>|#^[]')


def test_jsonl_counts_equal_note_counts(env):
    summary = _run(env)
    assert summary["counts"] == {"Symbol": 7, "Strategy": 3, "Recommendation": 6,
                                 "Trade": 1, "PolicyChange": 2, "AnalystNote": 1}
    seen_ids: set[str] = set()
    for c in ox.CLASSES:
        rows = _jsonl(env, c.jsonl)
        notes = [p for p in (env["vault"] / c.folder).glob("*.md") if ox.has_marker(p)]
        assert len(rows) == len(notes) == summary["counts"][c.name], c.name
        for r in rows:
            assert r["type"] == c.name
            assert list(r)[2:] == ([p.name for p in c.props] + [x.name for x in c.rels]
                                   + [p.name for p in c.extras])
            assert not any(isinstance(v, float) and not math.isfinite(v)
                           for v in r.values())
            assert "[[" not in json.dumps(r)
            seen_ids.add(r["id"])
    assert len(seen_ids) == sum(summary["counts"].values())

    for c in ox.CLASSES:
        for r in _jsonl(env, c.jsonl):
            for rel in c.rels:
                v = r[rel.name]
                for target in (v if isinstance(v, list) else [v] if v else []):
                    assert target in seen_ids, (r["id"], rel.name, target)


def test_ontology_json_describes_the_schema(env):
    summary = _run(env)
    o = json.loads((env["data_dir"] / "ontology.json").read_text())
    assert o["source_run_id"] == "bt_test" == summary["source_run_id"]
    assert [c["name"] for c in o["classes"]] == [c.name for c in ox.CLASSES]
    rec = next(c for c in o["classes"] if c["name"] == "Recommendation")
    roles = {p["name"]: p["role"] for p in rec["properties"]}
    assert roles["vol60"] == "feature" and roles["r20"] == "outcome"
    assert roles["outcome_label"] == "label"
    assert all(set(p) == {"name", "type", "unit", "role", "description"}
               for c in o["classes"] for p in c["properties"])
    assert {"from": "Recommendation", "to": "Symbol", "name": "symbol",
            "cardinality": "one"}.items() <= next(
        r for r in o["relations"]
        if r["from"] == "Recommendation" and r["name"] == "symbol").items()
    assert set(o["label_definitions"]["outcome_label"]) == {
        "good", "bad_stays_down", "bad", "pending"}
    assert o["outcome_label_counts"] == {"good": 2, "bad_stays_down": 1,
                                         "bad": 1, "pending": 2}
    index = (env["vault"] / f"{ox.INDEX_STEM}.md").read_text()
    assert "bad_stays_down" in index and "2026-01-01T00:00:00Z" in index


def test_policy_changes_parsed_from_change_history_only(env):
    _run(env)
    rows = _jsonl(env, "policy_changes.jsonl")
    assert [(r["date"], r["version"]) for r in rows] == [
        ("2026-09-02", "1"), ("2026-09-29", "1.3")]
    assert rows[0]["text"] == ("Baseline adopted. All prior rules void. "
                               "Second line of the same entry.")
    assert rows[1]["text"].endswith('"quoted" text with a colon: and more.')
    assert (env["vault"] / "Policy" / "2026-09-29 v1-3.md").is_file()


def test_analyst_note(env):
    _run(env)
    (n,) = _jsonl(env, "analyst_notes.jsonl")
    assert n["id"] == f"note:{AS_OF}:UPP" and n["symbol"] == "sym:UPP"
    assert n["summary"] == "Short." and n["wrong_if"] == "If it falls."
    assert n["narrative_model"] == "model=template" and n["n_metrics"] == 1
    assert n["metrics"][0]["source_column"] == "close"
    body = (env["vault"] / "AnalystNotes" / f"{AS_OF} UPP analyst note.md").read_text()
    assert "## Wrong if" in body and "Longer | piped." in body
    assert "| price | close | 12.5 | USD/share | mm_bars | close |" in body


def test_missing_optional_tables(env, tmp_path):
    bare = tmp_path / "bare.db"
    with sqlite3.connect(bare) as c:
        c.execute(schema.MM_BARS)
        c.execute(schema.MM_RECS_LEDGER)
        c.commit()
    summary = _run(env, db_path=bare)
    assert summary["counts"]["AnalystNote"] == 0
    assert summary["counts"]["Recommendation"] == 0
    assert summary["counts"]["Symbol"] == 6 and summary["source_run_id"] is None
    assert (env["data_dir"] / "analyst_notes.jsonl").read_text() == ""


def test_db_is_opened_read_only(env, tmp_path):
    conn = ox.connect_ro(env["db_path"])
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM mm_recs_ledger")
    conn.close()
    with pytest.raises(sqlite3.OperationalError):
        ox.connect_ro(tmp_path / "does_not_exist.db")
    assert not (tmp_path / "does_not_exist.db").exists()


def _symbols(env: dict) -> dict[str, dict]:
    return {r["ticker"]: r for r in _jsonl(env, "symbols.jsonl")}


def _ontology(env: dict) -> dict:
    return json.loads((env["data_dir"] / "ontology.json").read_text())


def _insert_memory(c, rec_id: str, sym: str, state: str, **kw) -> None:
    row = {"rec_id": rec_id, "as_of_date": AS_OF, "symbol": sym, "filled": 0,
           "state": state, "computed_at": "x"}
    row.update(kw)
    c.execute(f"INSERT INTO mm_outcome_memory ({', '.join(row)}) "
              f"VALUES ({', '.join('?' for _ in row)})", tuple(row.values()))


def test_memory_is_computed_when_the_table_is_missing(env):
    summary = _run(env)
    assert summary["memory_source"] == "computed:macmini.recs.outcome_memory.evaluate"
    recs = _recs(env)
    assert recs["UPP"]["memory_state"] == "good"
    assert recs["UPP"]["memory_entry_price"] == pytest.approx(100.0)
    assert recs["UPP"]["outcome_entry_price"] != pytest.approx(100.0)
    assert recs["UPP"]["memory_stop_hit_date"] is None
    assert recs["UPP"]["memory_return_to_date"] == pytest.approx(
        _future("up", 60)[-1] / 100.0 - 1, abs=1e-6)
    fut = _future("down", 60)
    hit = next(i for i, px in enumerate(fut) if px * 0.995 <= fut[0] * (1 - STOP))
    assert recs["DWN"]["memory_state"] == "bad_down"
    assert recs["DWN"]["memory_stop_hit_date"] == DAYS[N_PAST + hit]
    assert recs["DWN"]["memory_return_to_date"] == pytest.approx(
        fut[-1] / fut[0] - 1, abs=1e-6)
    assert recs["OUT"]["memory_state"] == "pending"
    assert summary["memory_states"] == {"good": 4, "open": 0, "recovered": 0,
                                        "bad_down": 1, "pending": 1}
    assert summary["memory_not_scored"] == 0

    sym = _symbols(env)
    assert summary["memory_blocked"] == ["DWN"]
    assert sym["DWN"]["memory_blocked"] is True
    assert sym["DWN"]["memory_blocked_reason"].startswith(f"Recommended {AS_OF} at ")
    assert DAYS[N_PAST + hit] in sym["DWN"]["memory_blocked_reason"]
    assert sym["UPP"]["memory_blocked"] is False
    assert sym["UPP"]["memory_blocked_reason"] is None
    assert sym["GONE"]["memory_blocked"] is False
    fm = _frontmatter(env["vault"] / "Symbols" / "DWN.md")
    assert fm["memory_blocked"] is True
    assert fm["memory_blocked_reason"] == sym["DWN"]["memory_blocked_reason"]


def test_the_two_label_systems_are_independent(env):
    ""
    _run(env)
    dip = _recs(env)["DIP"]
    assert dip["outcome_label"] == "bad" and dip["memory_state"] == "good"


def test_memory_table_wins_when_present(env):
    with sqlite3.connect(env["db_path"]) as c:
        c.execute(schema.MM_OUTCOME_MEMORY)
        _insert_memory(c, "aaaaaaaa11111111", "UPP", "bad_down", entry_price=100.0,
                       stop_hit_date="2026-03-05T00:00:00", last_close=90.0,
                       last_date="2026-04-01", return_to_date=-0.1, filled=1)
        _insert_memory(c, "bbbbbbbb22222222", "DWN", "recovered", entry_price=50.0,
                       stop_hit_date="2026-03-06", last_close=55.0,
                       last_date="2026-04-01", return_to_date=0.1)
        _insert_memory(c, "dddddddd44444444", "DIP", "open", entry_price=10.0,
                       last_close=9.5, last_date="2026-04-01", return_to_date=-0.05)
        _insert_memory(c, "fixture-0001", "MOG.A", "bad_down", entry_price=1.0)
        _insert_memory(c, "deleted-rec", "NEW", "bad_down", entry_price=1.0)
        c.commit()
    summary = _run(env)
    assert summary["memory_source"] == "table:mm_outcome_memory"
    recs = _recs(env)
    assert recs["UPP"]["memory_state"] == "bad_down"
    assert recs["UPP"]["memory_stop_hit_date"] == "2026-03-05"
    assert recs["UPP"]["memory_return_to_date"] == pytest.approx(-0.1)
    assert recs["DWN"]["memory_state"] == "recovered"
    assert recs["DIP"]["memory_state"] == "open"
    assert recs["NEW"]["memory_state"] is None
    assert recs["NEW"]["memory_return_to_date"] is None
    assert summary["memory_states"] == {"good": 0, "open": 1, "recovered": 1,
                                        "bad_down": 1, "pending": 0}
    assert summary["memory_not_scored"] == 3
    assert summary["memory_blocked"] == ["UPP"]
    sym = _symbols(env)
    assert sym["UPP"]["memory_blocked_reason"] == (
        "Recommended " + AS_OF + " at 100.00; stop level breached 2026-03-05; "
        "last close 90.00 on 2026-04-01 is still -10.0% vs entry. "
        "1 rec on this symbol in bad_down.")
    assert sym["DWN"]["memory_blocked"] is False
    assert sym["NEW"]["memory_blocked"] is False and sym["MOG.A"]["memory_blocked"] is False
    assert recs["UPP"]["outcome_label"] == "good"
    assert recs["DWN"]["outcome_label"] == "bad_stays_down"


def test_empty_memory_table_falls_back_to_computing(env):
    with sqlite3.connect(env["db_path"]) as c:
        c.execute(schema.MM_OUTCOME_MEMORY)
        c.commit()
    summary = _run(env)
    assert summary["memory_source"].startswith("computed:")
    assert summary["memory_blocked"] == ["DWN"]


def test_memory_unavailable_is_null_not_false(env, tmp_path):
    bars_only = tmp_path / "bars_only.db"
    with sqlite3.connect(bars_only) as c:
        c.execute(schema.MM_BARS)
        _insert_bars(c, "UPP", PAST)
        c.commit()
    summary = _run(env, db_path=bars_only)
    assert summary["memory_source"] is None and summary["memory_blocked"] == []
    sym = _symbols(env)
    assert sym["UPP"]["memory_blocked"] is None
    assert sym["UPP"]["n_bars"] == N_PAST
    assert "not available" in (env["vault"] / f"{ox.INDEX_STEM}.md").read_text()


def test_both_label_systems_are_documented(env):
    _run(env)
    o = _ontology(env)
    assert set(o["label_definitions"]["memory_state"]) == {
        "good", "open", "recovered", "bad_down", "pending"}
    assert set(o["label_definitions"]["outcome_label"]) == {
        "good", "bad_stays_down", "bad", "pending"}
    assert "fixed-horizon" in o["label_systems"]["summary"]
    assert "clears itself" in o["label_systems"]["summary"]
    assert [c["aspect"] for c in o["label_systems"]["comparison"]][:3] == [
        "Purpose", "Horizon", "Changes over time"]
    assert o["outcome_memory"]["state_counts"]["bad_down"] == 1
    assert [b["ticker"] for b in o["outcome_memory"]["blocked_symbols"]] == ["DWN"]
    assert {"outcome_label": "bad", "memory_state": "good", "recs": 1} in \
        o["outcome_memory"]["outcome_label_by_memory_state"]
    rec = next(c for c in o["classes"] if c["name"] == "Recommendation")
    roles = {p["name"]: p["role"] for p in rec["properties"]}
    assert roles["memory_state"] == "label"
    assert roles["memory_stop_hit_date"] == roles["memory_return_to_date"] == "outcome"

    index = (env["vault"] / f"{ox.INDEX_STEM}.md").read_text()
    assert "## The two label systems, side by side" in index
    assert "| `outcome_label` | `memory_state` |" in index
    assert "Blocked symbols (1): [[Symbols/DWN|DWN]]." in index
    assert o["label_systems"]["summary"] in index
    rec_note = (env["vault"] / "Recommendations" / f"{AS_OF} DWN 14 (bbbbbbbb).md").read_text()
    assert "Memory state (live, changes with price): **bad_down**" in rec_note


def test_selector_fields_are_null_without_the_table(env):
    summary = _run(env)
    assert summary["selector_symbols"] == 0 and summary["selector_as_of"] is None
    for s in _symbols(env).values():
        assert s["selector_score"] is None and s["selector_rank_pct"] is None
        assert s["selector_as_of"] is None and s["selector_model_id"] is None
        assert s["selector_eligible"] is None


def test_selector_latest_row_per_symbol(env):
    with sqlite3.connect(env["db_path"]) as c:
        c.execute(schema.MM_SELECTOR_SCORES)
        rows = [
            ("2026-03-01", "UPP", 0.10, 0.20, 1, "m1", "2026-03-01T22:00:00"),
            ("2026-03-02T00:00:00", "UPP", 0.70, 0.90, 1, "m1", "2026-03-02T22:00:00"),
            ("2026-03-02", "UPP", 0.80, 0.95, 1, "m2", "2026-03-02T23:00:00"),
            ("2026-03-01", "DWN", -0.30, None, 0, "m1", "2026-03-01T22:00:00"),
            ("2026-03-02", "ZZZ", 0.50, 0.50, 1, "m1", "2026-03-02T22:00:00"),
        ]
        c.executemany("INSERT INTO mm_selector_scores (as_of_date, symbol, score, "
                      "rank_pct, eligible, model_id, created_at) VALUES (?,?,?,?,?,?,?)",
                      rows)
        c.commit()
    summary = _run(env)
    assert summary["selector_symbols"] == 2 and summary["selector_as_of"] == "2026-03-02"
    sym = _symbols(env)
    assert sym["UPP"]["selector_score"] == pytest.approx(0.80)
    assert sym["UPP"]["selector_rank_pct"] == pytest.approx(0.95)
    assert sym["UPP"]["selector_as_of"] == "2026-03-02"
    assert sym["UPP"]["selector_model_id"] == "m2" and sym["UPP"]["selector_eligible"] is True
    assert sym["DWN"]["selector_score"] == pytest.approx(-0.30)
    assert sym["DWN"]["selector_rank_pct"] is None
    assert sym["DWN"]["selector_eligible"] is False
    assert sym["DWN"]["selector_as_of"] == "2026-03-01"
    assert sym["NEW"]["selector_score"] is None
    assert "ZZZ" not in sym
    fm = _frontmatter(env["vault"] / "Symbols" / "UPP.md")
    assert fm["selector_score"] == pytest.approx(0.80)
    assert fm["selector_as_of"] == date(2026, 3, 2)
    assert _ontology(env)["selector"] == {"symbols": 2, "as_of": "2026-03-02"}


def test_empty_selector_table(env):
    with sqlite3.connect(env["db_path"]) as c:
        c.execute(schema.MM_SELECTOR_SCORES)
        c.commit()
    assert _run(env)["selector_symbols"] == 0


def test_datasets_absent(env):
    summary = _run(env)
    assert summary["datasets"] == [] and _ontology(env)["datasets"] == []
    index = (env["vault"] / f"{ox.INDEX_STEM}.md").read_text()
    assert "## Datasets" in index and "_None found on disk._" in index


def test_datasets_are_referenced_not_copied(env):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    repo = env["project_root"]
    for d in ("data/selector", "exports", "docs",
              "scripts/analysis/bad_picks"):
        (repo / d).mkdir(parents=True)
    pq.write_table(pa.table({"x": list(range(7))}), repo / "data/selector/panel.parquet")
    (repo / "data/selector/scored_other.parquet").write_bytes(b"not listed")
    (repo / "exports/selector_eval_20260929_v2.json").write_text("{}")
    (repo / "exports/selector_eval_20260929_price.json").write_text("{}")
    (repo / "exports/unrelated.json").write_text("{}")
    (repo / "docs/bad_picks_root_cause_2026-09-29.md").write_text("# doc")
    (repo / "scripts/analysis/bad_picks/a.py").write_text("")
    (repo / "scripts/analysis/bad_picks/b.py").write_text("")

    before = _snapshot(repo)
    summary = _run(env)
    assert _snapshot(repo) == before
    assert summary["datasets"] == [
        "data/selector/panel.parquet",
        "exports/selector_eval_20260929_price.json",
        "exports/selector_eval_20260929_v2.json",
        "scripts/analysis/bad_picks",
        "docs/bad_picks_root_cause_2026-09-29.md",
    ]
    ds = {d["path"]: d for d in _ontology(env)["datasets"]}
    assert ds["data/selector/panel.parquet"]["rows"] == 7
    assert ds["data/selector/panel.parquet"]["kind"] == "parquet"
    assert ds["scripts/analysis/bad_picks"]["files"] == 2
    assert ds["exports/selector_eval_20260929_v2.json"]["rows"] is None
    assert all(set(d) == {"path", "kind", "rows", "files", "description"} and d["description"]
               for d in ds.values())
    index = (env["vault"] / f"{ox.INDEX_STEM}.md").read_text()
    assert "| `data/selector/panel.parquet` | parquet | 7 |" in index
    assert sorted(p.name for p in env["data_dir"].iterdir()) == sorted(
        [c.jsonl for c in ox.CLASSES] + ["ontology.json"])
    assert not list(env["vault"].rglob("*.parquet"))


def test_unreadable_parquet_still_listed(env):
    repo = env["project_root"]
    (repo / "data/selector").mkdir(parents=True)
    (repo / "data/selector/panel_daily.parquet").write_bytes(b"garbage")
    _run(env)
    (d,) = _ontology(env)["datasets"]
    assert d["path"] == "data/selector/panel_daily.parquet" and d["rows"] is None
