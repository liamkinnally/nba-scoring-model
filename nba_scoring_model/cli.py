import argparse
from dataclasses import asdict
from datetime import datetime
import json

from nba_scoring_model.api import JSONAPIClient, NBADataCollector
from nba_scoring_model import config
from nba_scoring_model.data.database import DatabaseManager
from nba_scoring_model.data.demo import seed_demo_database
from nba_scoring_model.modeling.trainer import ModelTrainer
from nba_scoring_model.modeling.evaluation import evaluate_recent_average_baselines


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value)


def nba_date(value: str | None) -> str | None:
    if not value:
        return None
    return datetime.fromisoformat(value).strftime("%m/%d/%Y")


def make_collector(database_url: str):
    db = DatabaseManager(database_url)
    db.create_tables()
    client = JSONAPIClient(
        calls_per_minute=config.API_CALLS_PER_MINUTE,
        timeout_seconds=config.REQUEST_TIMEOUT_SECONDS,
    )
    collector = NBADataCollector(
        client,
        db,
        config.ESPN_SCOREBOARD_URL,
        config.ESPN_SUMMARY_URL,
    )
    return db, collector


def run_demo(database_url: str, tune: bool) -> dict:
    db = DatabaseManager(database_url)
    seed_demo_database(db)
    trainer = ModelTrainer(db, config.MODEL_DIR)
    results = trainer.train_all(datetime(2025, 10, 1), datetime(2026, 3, 31), tune=tune)
    return {target: asdict(result) for target, result in results.items()}


def main():
    parser = argparse.ArgumentParser(description="NBA player stat prediction project")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo")
    demo.add_argument("--database-url", default="sqlite:///data/demo_nba.db")
    demo.add_argument("--tune", action="store_true")

    ingest_today = subparsers.add_parser("ingest-today")
    ingest_today.add_argument("--database-url", default=config.DATABASE_URL)
    ingest_today.add_argument("--include-boxscores", action="store_true")

    ingest_season = subparsers.add_parser("ingest-season")
    ingest_season.add_argument("--database-url", default=config.DATABASE_URL)
    ingest_season.add_argument("--season", required=True, help="Example: 2025-26")
    ingest_season.add_argument("--season-type", default="Regular Season")
    ingest_season.add_argument("--start")
    ingest_season.add_argument("--end")
    ingest_season.add_argument("--limit", type=int)

    train = subparsers.add_parser("train")
    train.add_argument("--database-url", default=config.DATABASE_URL)
    train.add_argument("--start", required=True, type=parse_date)
    train.add_argument("--end", required=True, type=parse_date)
    train.add_argument("--no-tune", action="store_true")

    evaluate_baselines = subparsers.add_parser("evaluate-baselines")
    evaluate_baselines.add_argument("--database-url", default=config.DATABASE_URL)
    evaluate_baselines.add_argument("--start", required=True, type=parse_date)
    evaluate_baselines.add_argument("--end", required=True, type=parse_date)

    args = parser.parse_args()

    if args.command == "demo":
        output = run_demo(args.database_url, args.tune)

    elif args.command == "ingest-today":
        _, collector = make_collector(args.database_url)
        game_ids = collector.ingest_scoreboard()
        player_rows = 0
        if args.include_boxscores:
            for game_id in game_ids:
                player_rows += collector.ingest_boxscore(game_id)
        output = {
            "games_ingested": len(game_ids),
            "player_stat_rows_ingested": player_rows,
        }

    elif args.command == "ingest-season":
        _, collector = make_collector(args.database_url)
        output = collector.ingest_season(
            season=args.season,
            season_type=args.season_type,
            date_from=nba_date(args.start),
            date_to=nba_date(args.end),
            limit=args.limit,
        )

    elif args.command == "evaluate-baselines":
        db = DatabaseManager(args.database_url)
        output = evaluate_recent_average_baselines(
            db,
            args.start,
            args.end,
        )

    else:
        db = DatabaseManager(args.database_url)
        trainer = ModelTrainer(db, config.MODEL_DIR)
        results = trainer.train_all(args.start, args.end, tune=not args.no_tune)
        output = {target: asdict(result) for target, result in results.items()}

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
