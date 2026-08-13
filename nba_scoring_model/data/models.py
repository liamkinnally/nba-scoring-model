
from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    team_id: Mapped[str] = mapped_column(String, primary_key=True)
    team_name: Mapped[str] = mapped_column(String, nullable=False)
    abbreviation: Mapped[Optional[str]] = mapped_column(String(3))

    players: Mapped[List["Player"]] = relationship(back_populates="team")


class Player(Base):
    __tablename__ = "players"

    player_id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    team_id: Mapped[Optional[str]] = mapped_column(ForeignKey("teams.team_id"))
    position: Mapped[Optional[str]] = mapped_column(String)

    team: Mapped[Optional[Team]] = relationship(back_populates="players")
    game_stats: Mapped[List["PlayerGameStats"]] = relationship(back_populates="player")


class Game(Base):
    __tablename__ = "games"

    game_id: Mapped[str] = mapped_column(String, primary_key=True)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    home_team_id: Mapped[str] = mapped_column(ForeignKey("teams.team_id"), nullable=False)
    away_team_id: Mapped[str] = mapped_column(ForeignKey("teams.team_id"), nullable=False)
    game_status: Mapped[str] = mapped_column(String, default="final")
    vegas_total: Mapped[Optional[float]] = mapped_column(Float)
    vegas_spread: Mapped[Optional[float]] = mapped_column(Float)
    home_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_score: Mapped[Optional[int]] = mapped_column(Integer)

    player_stats: Mapped[List["PlayerGameStats"]] = relationship(back_populates="game")


class PlayerGameStats(Base):
    __tablename__ = "player_game_stats"
    __table_args__ = (UniqueConstraint("game_id", "player_id", name="uq_game_player"),)

    stat_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"), nullable=False, index=True)
    player_id: Mapped[str] = mapped_column(ForeignKey("players.player_id"), nullable=False, index=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.team_id"), nullable=False, index=True)

    minutes_played: Mapped[float] = mapped_column(Float, default=0.0)
    points: Mapped[int] = mapped_column(Integer, default=0)
    assists: Mapped[int] = mapped_column(Integer, default=0)
    rebounds: Mapped[int] = mapped_column(Integer, default=0)
    usage_rate: Mapped[Optional[float]] = mapped_column(Float)
    field_goal_attempts: Mapped[Optional[int]] = mapped_column(Integer)
    free_throw_attempts: Mapped[Optional[int]] = mapped_column(Integer)
    turnovers: Mapped[Optional[int]] = mapped_column(Integer)

    game: Mapped[Game] = relationship(back_populates="player_stats")
    player: Mapped[Player] = relationship(back_populates="game_stats")
