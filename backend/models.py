from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime

from database import Base


class Drawing(Base):
    __tablename__ = "drawings"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(
        String(100),
        nullable=False,
        default="Untitled Drawing"
    )

    stroke_data = Column(
        Text,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )