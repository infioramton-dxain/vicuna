""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from macmini.data._db import connect


def daily_sentiment(
    symbol: str,
    start: str,
    end: str,
    db_path: Path | str | None = None,
) -> pd.DataFrame:
    ""
    sql = """
        SELECT
            substr(window_start, 1, 10) AS date,
            AVG(sentiment)              AS sentiment_mean,
            AVG(sentiment_confidence)   AS sentiment_confidence_mean,
            MAX(materiality)            AS materiality_max,
            SUM(article_count)          AS article_count_sum,
            MAX(contract_award)         AS contract_award_any,
            MAX(guidance)               AS guidance_any,
            MAX(mna)                    AS mna_any,
            MAX(analyst_action)         AS analyst_action_any,
            MAX(commodity_move)         AS commodity_move_any
        FROM sentiment
        WHERE symbol = ?
          AND substr(window_start, 1, 10) BETWEEN ? AND ?
          AND status = 'ok'
        GROUP BY date
        ORDER BY date
    """
    with connect(db_path) as c:
        return pd.read_sql_query(sql, c, params=[symbol, start, end])
