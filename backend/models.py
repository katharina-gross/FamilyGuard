import uuid
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Integer
from sqlalchemy.orm import relationship
from datetime import datetime, timedelta
from database import Base

class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    firstName = Column(String, nullable=False)
    lastName = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    phone = Column(String)
    timezone = Column(String, nullable=False)
    createdAt = Column(DateTime, default=datetime.utcnow)

    devices = relationship("Device", back_populates="owner", cascade="all, delete-orphan")
    tethering_codes = relationship("TetheringCode", back_populates="user", cascade="all, delete-orphan")
    screenshots = relationship("Screenshot", back_populates="user", cascade="all, delete-orphan")
    screentimes = relationship("Screentime", back_populates="user", cascade="all, delete-orphan")
    log_screentimes = relationship("LogScreentime", back_populates="user", cascade="all, delete-orphan")

class Device(Base):
    __tablename__ = "devices"

    device_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    model = Column(String)
    osVersion = Column(String)
    isBlocked = Column(Boolean, default=False)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    tetheredAt = Column(DateTime, default=datetime.utcnow)
    heartbeat = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="devices")
    screenshots = relationship("Screenshot", back_populates="device", cascade="all, delete-orphan")
    screentimes = relationship("Screentime", back_populates="device", cascade="all, delete-orphan")
    log_screentimes = relationship("LogScreentime", back_populates="device", cascade="all, delete-orphan")

class TetheringCode(Base):
    __tablename__ = "tethering_codes"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    code = Column(String, unique=True, nullable=False, default=lambda: str(uuid.uuid4().hex)[:8].upper())
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    createdAt = Column(DateTime, default=datetime.utcnow)
    expiredAt = Column(DateTime, default=lambda: datetime.utcnow() + timedelta(minutes=10))
    used = Column(Boolean, default=False)

    user = relationship("User", back_populates="tethering_codes")

class Category(Base):
    __tablename__ = 'categories'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, unique=True, nullable=False)
    label = Column(String, nullable=False)
    description = Column(String)
    restricted = Column(Boolean, default=False)

class Screenshot(Base):
    __tablename__ = 'screenshots'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    device_id = Column(String, ForeignKey("devices.device_id"), nullable=False)
    image = Column(String, nullable=False)
    category = Column(String)
    transaction_id = Column(String)
    createdAt = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="screenshots")
    device = relationship("Device", back_populates="screenshots")

class Screentime(Base):
    __tablename__ = 'screentime'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    deviceId = Column(String, ForeignKey("devices.device_id"), nullable=False)
    limit = Column(Integer, nullable=False)
    scheduleStart = Column(String, nullable=False)
    scheduleEnd = Column(String, nullable=False)
    appName = Column(String, nullable=False)

    user = relationship("User", back_populates="screentimes")
    device = relationship("Device", back_populates="screentimes")
    logs = relationship("LogScreentime", back_populates="screentime", cascade="all, delete-orphan")

class LogScreentime(Base):
    __tablename__ = 'log_screentime'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)
    device_id = Column(String, ForeignKey("devices.device_id"), nullable=False)
    screentime_id = Column(String, ForeignKey("screentime.id"), nullable=False)
    screenTime = Column(Integer, nullable=False)
    timestamp = Column(String, nullable=False)
    activityType = Column(String, nullable=False)

    user = relationship("User", back_populates="log_screentimes")
    device = relationship("Device", back_populates="log_screentimes")
    screentime = relationship("Screentime", back_populates="logs")
