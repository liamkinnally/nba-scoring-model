
from datetime import datetime, timedelta
from itertools import combinations
from typing import Dict

import numpy as np

from .database import DatabaseManager
from .models import Game, Player, PlayerGameStats, Team


TEAM_NAMES = {
    "ATL": "Atlanta",
    "BOS": "Boston",
    "CHI": "Chicago",
    "DAL": "Dallas",
}


def seed_demo_database(db: DatabaseManager, *, seed: int = 42, game_days: int = 72) -> None:
    """Create deterministic synthetic NBA-like data for an offline end-to-end demo.

    The demo proves the pipeline runs. Its metrics are not real NBA model results.
    """
    rng = np.random.default_rng(seed)
    db.drop_tables()
    db.create_tables()

    team_ids = list(TEAM_NAMES)
    player_profiles: Dict[str, Dict[str, float | str]] = {}

    with db.get_session() as session:
        for team_id, team_name in TEAM_NAMES.items():
            session.add(Team(team_id=team_id, team_name=team_name, abbreviation=team_id))
            for idx in range(6):
                player_id = f"{team_id}_{idx + 1}"
                scoring = float(rng.uniform(9, 28))
                assists = float(rng.uniform(1.5, 8.5))
                rebounds = float(rng.uniform(2.5, 10.5))
                player_profiles[player_id] = {
                    "team_id": team_id,
                    "scoring": scoring,
                    "assists": assists,
                    "rebounds": rebounds,
                }
                session.add(
                    Player(
                        player_id=player_id,
                        name=f"Demo Player {player_id}",
                        team_id=team_id,
                        position="G/F",
                    )
                )

    start = datetime(2025, 10, 1, 19, 0)
    matchups = list(combinations(team_ids, 2))

    with db.get_session() as session:
        for day in range(game_days):
            game_date = start + timedelta(days=day * 2)
            # Rotate through two non-overlapping matchups each game day.
            if day % 3 == 0:
                day_matchups = [("ATL", "BOS"), ("CHI", "DAL")]
            elif day % 3 == 1:
                day_matchups = [("ATL", "CHI"), ("BOS", "DAL")]
            else:
                day_matchups = [("ATL", "DAL"), ("BOS", "CHI")]

            for slot, (home, away) in enumerate(day_matchups):
                game_id = f"DEMO_{day:03d}_{slot}"
                vegas_total = float(rng.normal(224, 8))
                vegas_spread = float(rng.normal(0, 5))
                home_total = 0
                away_total = 0
                stat_rows = []

                for team_id, is_home in [(home, True), (away, False)]:
                    opponent = away if is_home else home
                    for player_id, profile in player_profiles.items():
                        if profile["team_id"] != team_id:
                            continue

                        fatigue = -1.0 if day % 7 == 0 else 0.0
                        home_bonus = 0.8 if is_home else 0.0
                        matchup_effect = ((hash(player_id + opponent) % 9) - 4) * 0.15
                        minutes = float(np.clip(rng.normal(31, 4), 18, 40))
                        points = max(0, int(round(rng.normal(float(profile["scoring"]) + home_bonus + fatigue + matchup_effect, 4.5))))
                        assists = max(0, int(round(rng.normal(float(profile["assists"]), 2.0))))
                        rebounds = max(0, int(round(rng.normal(float(profile["rebounds"]), 2.3))))
                        fga = max(1, int(round(points / 1.35 + rng.normal(2, 2))))
                        fta = max(0, int(round(rng.normal(4, 2))))
                        tov = max(0, int(round(rng.normal(2.2, 1.2))))
                        usage = float(np.clip(100 * (fga + 0.44 * fta + tov) / max(minutes * 1.7, 1), 8, 42))

                        if is_home:
                            home_total += points
                        else:
                            away_total += points

                        stat_rows.append(
                            PlayerGameStats(
                                game_id=game_id,
                                player_id=player_id,
                                team_id=team_id,
                                minutes_played=minutes,
                                points=points,
                                assists=assists,
                                rebounds=rebounds,
                                usage_rate=usage,
                                field_goal_attempts=fga,
                                free_throw_attempts=fta,
                                turnovers=tov,
                            )
                        )

                session.add(
                    Game(
                        game_id=game_id,
                        date=game_date,
                        home_team_id=home,
                        away_team_id=away,
                        game_status="final",
                        vegas_total=vegas_total,
                        vegas_spread=vegas_spread,
                        home_score=home_total,
                        away_score=away_total,
                    )
                )
                session.add_all(stat_rows)
