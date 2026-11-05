"""
Build the model's features: one row per player per target season, with
inputs from earlier seasons and the target season's actual fantasy points.

Run from the project root to print a preview:
    python -m src.features
"""

import sqlite3

import pandas as pd

import config
from src.scoring import (
    add_fantasy_points,
    add_missed_shots,
    add_per_game,
    add_positions,
    load_player_seasons,
    load_players,
)

# Stats carried forward from earlier seasons as model inputs
FEATURE_STATS = ["FANTASY_PTS", "FANTASY_PPG", "GP_SHARE", "MPG", "MIN", "WIN_PCT", "AGE"]


def load_team_seasons():
    """Load each team's games played and win % for every season."""
    conn = sqlite3.connect(config.DB_PATH)
    teams = pd.read_sql(
        "SELECT SEASON, TEAM_ID, WINS + LOSSES AS TEAM_GAMES, WIN_PCT FROM team_seasons",
        conn,
    )
    conn.close()
    return teams


def add_rate_stats(df, teams):
    """Add team context plus the per-game and share stats that stay comparable in shortened seasons."""
    df = df.merge(teams, on=["SEASON", "TEAM_ID"], how="left", validate="many_to_one")
    df["GP_SHARE"] = (df["GP"] / df["TEAM_GAMES"]).clip(upper=1)
    df["MPG"] = df["MIN"] / df["GP"]
    return df


def build_base_table():
    """One row per player season with fantasy points, positions and rate stats."""
    df = load_player_seasons()
    df = add_missed_shots(df)
    df = add_fantasy_points(df)
    df = add_per_game(df)
    df = add_positions(df, load_players())
    df = add_rate_stats(df, load_team_seasons())
    return df


def add_season_number(df):
    """Number the seasons in order (2018-19 = 0) so earlier seasons can be found by subtraction."""
    df = df.copy()
    df["SEASON_NUM"] = df["SEASON"].map({season: i for i, season in enumerate(config.SEASONS)})
    return df


def lagged_stats(base, years_back):
    """Each player's stats from `years_back` seasons earlier, labeled with the season they lead into."""
    lagged = base[["PLAYER_ID", "SEASON_NUM"] + FEATURE_STATS].copy()
    lagged["SEASON_NUM"] += years_back
    return lagged.rename(columns={stat: f"{stat}_LAG{years_back}" for stat in FEATURE_STATS})


if __name__ == "__main__":
    base = add_season_number(build_base_table())
    last_season = lagged_stats(base, 1)
    check = base.merge(last_season, on=["PLAYER_ID", "SEASON_NUM"], how="inner")
    print(check[["PLAYER_NAME", "SEASON", "FANTASY_PPG", "FANTASY_PPG_LAG1", "AGE", "AGE_LAG1"]].head(8).to_string(index=False))
    print("\nAge minus last season's age:")
    print((check["AGE"] - check["AGE_LAG1"]).value_counts())