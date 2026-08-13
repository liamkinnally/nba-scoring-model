
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base


class DatabaseManager:
    def __init__(self, connection_string: str = "sqlite:///data/nba_predictions.db") -> None:
        if connection_string.startswith("sqlite:///"):
            sqlite_path = Path(connection_string.removeprefix("sqlite:///"))
            if sqlite_path.parent != Path("."):
                sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(connection_string)
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    def drop_tables(self) -> None:
        Base.metadata.drop_all(self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
