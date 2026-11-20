"""持久化层：用户画像、体检报告、食材、菜谱、Agent 运行记录。

默认 SQLite（零依赖启动），生产把 DATABASE_URL 换成 MySQL 即可，模型层不用改。
"""

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    email: Mapped[str | None] = mapped_column(String(120), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    reports: Mapped[list["HealthReport"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    recipes: Mapped[list["Recipe"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class HealthReport(Base):
    __tablename__ = "health_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    filename: Mapped[str] = mapped_column(String(255))
    filepath: Mapped[str] = mapped_column(String(500))
    content_text: Mapped[str] = mapped_column(Text, default="")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="reports")


class FridgeImage(Base):
    __tablename__ = "fridge_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    filename: Mapped[str] = mapped_column(String(255))
    filepath: Mapped[str] = mapped_column(String(500))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    ingredients: Mapped[list["Ingredient"]] = relationship(
        back_populates="fridge_image", cascade="all, delete-orphan"
    )


class Ingredient(Base):
    __tablename__ = "ingredients"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    fridge_image_id: Mapped[int] = mapped_column(ForeignKey("fridge_images.id"))
    name: Mapped[str] = mapped_column(String(100))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    fridge_image: Mapped[FridgeImage] = relationship(back_populates="ingredients")


class Recipe(Base):
    __tablename__ = "recipes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    title: Mapped[str] = mapped_column(String(255))
    ingredients: Mapped[str] = mapped_column(Text, default="")
    instructions: Mapped[str] = mapped_column(Text, default="")
    nutrition_info: Mapped[str] = mapped_column(Text, default="")
    quality_score: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship(back_populates="recipes")


class Favorite(Base):
    __tablename__ = "favorites"
    __table_args__ = (UniqueConstraint("user_id", "recipe_id", name="uq_user_recipe"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    recipe_id: Mapped[int] = mapped_column(ForeignKey("recipes.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AgentRun(Base):
    """一次 Agent 运行的快照，用于复盘/评估（对应简历里的可观测性）。"""

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    query: Mapped[str] = mapped_column(Text)
    intent: Mapped[str] = mapped_column(String(32), default="")
    quality_score: Mapped[int] = mapped_column(Integer, default=0)
    revision_count: Mapped[int] = mapped_column(Integer, default=0)
    final_answer: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, future=True, pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)


def ensure_user(user_id: int) -> None:
    with SessionLocal() as db:
        if db.get(User, user_id) is None:
            db.add(User(id=user_id, username=f"user_{user_id}", email=f"user_{user_id}@demo.local"))
            db.commit()


def save_recipe(user_id: int, result: dict[str, Any]) -> int | None:
    """把一次成功运行落库（MySQL）+ 写入用户历史记忆（Milvus / 内存降级），返回菜谱 id。

    向量库写入失败不影响 MySQL 主落库，独立降级。
    """
    recipe = result.get("draft_recipe") or {}
    if not recipe:
        return None
    ensure_user(user_id)

    final_answer = result.get("final_answer", "")
    session_id = result.get("session_id", "")

    with SessionLocal() as db:
        row = Recipe(
            user_id=user_id,
            session_id=session_id,
            title=recipe.get("title", ""),
            ingredients="、".join(recipe.get("ingredients", [])),
            instructions="\n".join(recipe.get("steps", [])),
            nutrition_info=recipe.get("nutrition_note", ""),
            quality_score=(result.get("reflection") or {}).get("score", 0),
        )
        db.add(row)
        db.add(
            AgentRun(
                session_id=session_id,
                user_id=user_id,
                query=result.get("user_query", ""),
                intent=result.get("intent", ""),
                quality_score=row.quality_score,
                revision_count=result.get("revision_count", 0),
                final_answer=final_answer,
            )
        )
        db.commit()

    # 写入用户历史交互向量记忆 —— 容错，失败不影响主流程
    if final_answer:
        try:
            from app.adapters.vector import store_user_interaction

            store_user_interaction(
                user_id=user_id,
                text=final_answer,
                metadata={
                    "query": result.get("user_query", ""),
                    "intent": result.get("intent", ""),
                    "session_id": session_id,
                },
            )
        except Exception:
            pass  # Milvus 连不上 / 内存降级都不会走到这里，这里是兜底

    return row.id


def list_recipes(user_id: int, limit: int = 20) -> list[dict[str, Any]]:
    from sqlalchemy import select

    with SessionLocal() as db:
        rows = db.scalars(
            select(Recipe).where(Recipe.user_id == user_id).order_by(Recipe.created_at.desc()).limit(limit)
        ).all()
        return [
            {
                "id": r.id,
                "title": r.title,
                "ingredients": r.ingredients,
                "instructions": r.instructions,
                "nutrition_info": r.nutrition_info,
                "quality_score": r.quality_score,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]
