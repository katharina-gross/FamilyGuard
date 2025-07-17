from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from database import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    firstName = Column(String, nullable=False)
    lastName = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    timezone = Column(String, nullable=False)
    phone = Column(String)
    createdAt = Column(DateTime, default=datetime.utcnow)

    devices = relationship("Device", back_populates="owner")
    tethering_codes = relationship("TetheringCode", back_populates="user")


class Device(Base):
    __tablename__ = "devices"

    device_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    name = Column(String, nullable=False)
    model = Column(String)
    osVersion = Column(String)
    isBlocked = Column(Boolean, default=False)
    tetheredAt = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="devices")


class TetheringCode(Base):
    __tablename__ = "tethering_codes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    code = Column(String, unique=True, nullable=False)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    expiredAt = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)

    user = relationship("User", back_populates="tethering_codes")
