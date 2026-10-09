""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "hermes" / "trading_prep.py"


def _build_fixture_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE mm_export_metadata (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO mm_export_metadata VALUES
          ('as_of_date', '2026-07-01'),
          ('generated_at', '2026-07-01T12:54:59+00:00'),
          ('backtest_run_id', 'bt_test'),
          ('schema_version', '3.0'),
          ('macmini_min_version', '0.1.0');

        CREATE TABLE mm_regime (as_of_date TEXT, vix_close REAL, vix_regime TEXT);
        INSERT INTO mm_regime VALUES ('2026-07-01', 17.65, 'normal');

        CREATE TABLE mm_active_picks (
          rank INTEGER, strategy_id INTEGER, strategy_name TEXT,
          symbol TEXT, sector TEXT, archetype TEXT,
          deflated_sharpe REAL, profit_factor REAL, max_dd REAL,
          suggested_size_pct REAL, suggested_stop_loss_pct REAL,
          suggested_target_pct REAL, rationale TEXT
        );
        INSERT INTO mm_active_picks VALUES
          (1, 5, 'Ichimoku Cloud Breakout', 'QQQI', 'Other', NULL,
           0.388, 999.0, -0.048, 0.05, 0.08, 0.20, 'cloud breakout'),
          (2, 7, 'Heikin-Ashi Streak', 'CVLT', 'Software', NULL,
           0.242, 252.4, -0.093, 0.05, 0.08, 0.20, '3 green HA bars'),
          (3, 3, 'Donchian 20', 'GE', 'Defense', 'defense_prime',
           0.220, 916.1, -0.090, 0.05, 0.08, 0.20, 'close > 20d high'),
          (4, 8, 'PSAR + EMA200', 'NSIT', 'Other', NULL,
           0.256, 666.9, -0.071, 0.05, 0.08, 0.20, 'SAR bullish'),
          (5, 17, 'Vol Breakout', 'NSIT', 'Other', NULL,
           0.216, 666.0, -0.072, 0.05, 0.08, 0.20, 'close > prior high + ATR');
        """
    )
    conn.commit()
    conn.close()


def _run_script(db_path: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["AGENT_SHIPPED_DB"] = str(db_path)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )


def test_script_runs_and_emits_expected_sections(tmp_path):
    db = tmp_path / "macmini.db"
    _build_fixture_db(db)

    result = _run_script(db)

    assert result.returncode == 0, f"stderr: {result.stderr}"
    out = result.stdout

    for section in (
        "DATE_LOCAL:",
        "DATA_AS_OF:",
        "VIX:",
        "PICK_COUNT: 5",
        "PICKS:",
    ):
        assert section in out, f"missing section {section!r}\n---\n{out}"


def test_sector_concentration_flag_fires_at_three(tmp_path):
    db = tmp_path / "macmini.db"
    _build_fixture_db(db)

    out = _run_script(db).stdout
    assert "SECTOR_CONCENTRATION" in out
    assert "Other=3" in out


def test_missing_db_exits_nonzero(tmp_path):
    db = tmp_path / "does_not_exist.db"
    result = _run_script(db)
    assert result.returncode == 1
    assert "shipped DB missing" in result.stderr


def test_freshness_label_flags_stale(tmp_path):
    db = tmp_path / "macmini.db"
    _build_fixture_db(db)
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE mm_export_metadata SET value=? WHERE key='as_of_date'",
        ("2026-06-25",),
    )
    conn.commit()
    conn.close()

    out = _run_script(db).stdout
    assert "very stale" in out or "HALT" in out
