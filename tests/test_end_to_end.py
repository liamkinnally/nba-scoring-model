from datetime import datetime

from nba_scoring_model.data.database import DatabaseManager
from nba_scoring_model.data.demo import seed_demo_database
from nba_scoring_model.modeling.trainer import ModelTrainer


def test_demo_training_runs(tmp_path):
    db_path = tmp_path / "demo.db"
    model_dir = tmp_path / "models"
    db = DatabaseManager(f"sqlite:///{db_path}")
    seed_demo_database(db, game_days=18)

    trainer = ModelTrainer(db, model_dir)
    results = trainer.train_all(
        datetime(2025, 10, 1),
        datetime(2026, 1, 31),
        tune=False,
    )

    assert set(results) == {"points", "assists", "rebounds"}
    for result in results.values():
        assert result.train_rows > 0
        assert result.test_rows > 0
        assert result.mae >= 0
