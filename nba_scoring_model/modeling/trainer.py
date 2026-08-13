
from datetime import datetime
from pathlib import Path
from typing import Dict

from nba_scoring_model.data.database import DatabaseManager
from nba_scoring_model.features.engineering import FeatureEngineer

from .base import PlayerStatModel, TrainingResult


class ModelTrainer:
    def __init__(self, db_manager: DatabaseManager, model_dir: str | Path = "artifacts/models") -> None:
        self.db_manager = db_manager
        self.feature_engineer = FeatureEngineer(db_manager)
        self.models = {
            target: PlayerStatModel(target, db_manager, self.feature_engineer, model_dir)
            for target in ("points", "assists", "rebounds")
        }

    def train_all(
        self,
        start_date: datetime,
        end_date: datetime,
        *,
        tune: bool = True,
    ) -> Dict[str, TrainingResult]:
        results: Dict[str, TrainingResult] = {}
        for target, model in self.models.items():
            frame = model.prepare_training_frame(start_date, end_date)
            results[target] = model.train(frame, tune=tune)
        return results
