import os
import uuid
import string
import random
import logging
import base64
from datetime import datetime, timedelta, date, timezone
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import zipfile
from collections import defaultdict

from ai_worker import moderate_screenshot

import jwt
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, Request, Query, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.openapi.utils import get_openapi
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Integer, Date, JSON, func, desc
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
current_dir = Path(__file__).resolve().parent
static_dir = current_dir / "static"
uploads_dir = current_dir / "uploads"
os.makedirs(uploads_dir, exist_ok=True)  # Create upload directory

# Create base class for models
Base = declarative_base()


# Database models
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
    devices = relationship("Device", back_populates="user")
    notifications = relationship("Notification", back_populates="user")
    statistics = relationship("Statistic", back_populates="user")
    categories = relationship("Category", back_populates="user")
    screenshots = relationship("Screenshot", back_populates="user")
    screen_times = relationship("ScreenTime", back_populates="user")
    screen_time_logs = relationship("ScreenTimeLog", back_populates="user")


class Device(Base):
    __tablename__ = "devices"
    device_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    name = Column(String, nullable=False)
    model = Column(String)
    osVersion = Column(String)
    isBlocked = Column(Boolean, default=False)
    tetheredAt = Column(DateTime, default=datetime.utcnow)
    heartbeat = Column(DateTime, default=datetime.utcnow)  # Added heartbeat field
    user = relationship("User", back_populates="devices")
    statistics = relationship("Statistic", back_populates="device")
    screenshots = relationship("Screenshot", back_populates="device")
    screen_times = relationship("ScreenTime", back_populates="device")
    screen_time_logs = relationship("ScreenTimeLog", back_populates="device")


class TetheringCode(Base):
    __tablename__ = "tethering_codes"
    code = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    expiredAt = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)


class Notification(Base):
    __tablename__ = "notifications"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    title = Column(String, nullable=False)
    message = Column(String, nullable=False)
    type = Column(String, default="info")  # "info", "alert", "success"
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="notifications")


class Statistic(Base):
    __tablename__ = "statistics"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    device_id = Column(String, ForeignKey('devices.device_id'))
    date = Column(Date, default=date.today)
    total_usage = Column(Integer, default=0)  # in minutes
    app_usage = Column(JSON)  # {"app1": 30, "app2": 45}
    blocked_time = Column(Integer, default=0)  # in minutes
    user = relationship("User", back_populates="statistics")
    device = relationship("Device", back_populates="statistics")


class Category(Base):
    __tablename__ = "categories"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    name = Column(String, nullable=False)
    label = Column(String)
    description = Column(String)
    restricted = Column(Boolean, default=False)
    icon = Column(String, default="📱")
    time_limit = Column(Integer, default=0)  # in minutes
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="categories")


class Screenshot(Base):
    __tablename__ = "screenshots"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    device_id = Column(String, ForeignKey('devices.device_id'), nullable=False)
    image = Column(String, nullable=False)  # File path
    category = Column(String)
    transaction_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="screenshots")
    device = relationship("Device", back_populates="screenshots")


class ScreenTime(Base):
    __tablename__ = "screen_times"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    device_id = Column(String, ForeignKey('devices.device_id'), nullable=False)
    limit = Column(Integer, nullable=False)  # in minutes
    schedule_start = Column(String)  # "08:00"
    schedule_end = Column(String)  # "22:00"
    app_name = Column(String)  # Added app_name field
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="screen_times")
    device = relationship("Device", back_populates="screen_times")


class ScreenTimeLog(Base):
    __tablename__ = "screen_time_logs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    device_id = Column(String, ForeignKey('devices.device_id'), nullable=False)
    screen_time_id = Column(String, ForeignKey('screen_times.id'), nullable=False)
    screen_time = Column(Integer, nullable=False)  # in minutes
    timestamp = Column(DateTime, nullable=False)
    activity_type = Column(String)  # "app", "web", "total"
    user = relationship("User", back_populates="screen_time_logs")
    device = relationship("Device", back_populates="screen_time_logs")
    screen_time_obj = relationship("ScreenTime")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle context"""
    await init_db()
    yield


# Create FastAPI instance
app = FastAPI(lifespan=lifespan)


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    return response


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "fallback_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 10

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer()


# Helper functions
def get_default_category(db: Session, user_id: str) -> Optional[Category]:
    """Gets the default category for the user"""
    return db.query(Category).filter(
        Category.user_id == user_id,
        func.lower(Category.name) == "uncategorized"
    ).first()


def analyze_image_and_get_category(image_data: bytes, user_id: str, db: Session) -> Tuple[Optional[Category], float]:
    """
    Analyzes image and returns category with confidence level
    with improved error handling and category creation
    """
    try:
        # Get user's existing categories
        categories = db.query(Category).filter(Category.user_id == user_id).all()
        categories_names = [c.name for c in categories]

        # Analyze image with AI
        report = moderate_screenshot(image_data, categories_names)

        # Log AI response for debugging
        logger.info(f"AI report: {report}")

        # Handle AI errors
        if "error" in report:
            logger.error(f"AI error: {report['error']}")
            # Return default category
            return get_default_category(db, user_id), 0.0

        if "category" in report:
            category_name = report["category"]
            confidence = report.get("confidence", 0.0)

            # Find existing category by name
            category = next((c for c in categories if c.name == category_name), None)

            if not category:
                # Check if category was created in parallel
                category = db.query(Category).filter(
                    Category.user_id == user_id,
                    Category.name == category_name
                ).first()

            if not category:
                # Create new category if not found
                try:
                    category = Category(
                        user_id=user_id,
                        name=category_name,
                        label=category_name.capitalize(),
                        description="Automatically created category",
                        restricted=False,
                        icon="📱",
                        time_limit=0
                    )
                    db.add(category)
                    db.commit()
                    db.refresh(category)
                    logger.info(f"Created new category: {category_name}")
                except IntegrityError:
                    db.rollback()
                    # Category might have been created in parallel
                    category = db.query(Category).filter(
                        Category.user_id == user_id,
                        Category.name == category_name
                    ).first()
                    if category:
                        logger.info(f"Found existing category after conflict: {category_name}")
                    else:
                        logger.error(f"Error creating category: {category_name}")

            return category, confidence

    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}", exc_info=True)

    # Return default category for any errors
    default_category = get_default_category(db, user_id)
    return default_category, 0.0


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    """Verifies JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or missing token: {str(e)}"
        )


def generate_code() -> str:
    """Generates random tethering code"""
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(5))


def minutes_to_hh_mm(minutes: int) -> str:
    """Converts minutes to 'Xh Ym' format"""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m" if hours > 0 else f"{mins}m"


# Initialize database
async def init_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./familyguard.db")
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    app.state.db = SessionLocal()
    logger.info("Database initialized")


def get_db():
    return app.state.db


# Pydantic models
class UserCreate(BaseModel):
    firstName: str
    lastName: str
    email: str
    password: str
    timezone: str
    phone: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class DeviceTetherRequest(BaseModel):
    code: str
    deviceName: str
    deviceModel: Optional[str] = None
    osVersion: Optional[str] = None


class DeviceResponse(BaseModel):
    deviceId: str
    name: str
    model: Optional[str] = None
    osVersion: Optional[str] = None
    tetheringAt: str
    isBlocked: bool
    status: str
    lastActive: str


class TetheringCodeResponse(BaseModel):
    code: str
    expiresAt: str
    validitySeconds: int


class ErrorItem(BaseModel):
    field: str
    message: str


class ErrorResponse(BaseModel):
    status: str
    code: int
    message: str
    errors: Optional[List[ErrorItem]] = None


class UserResponse(BaseModel):
    userId: str
    email: str
    firstName: str
    lastName: str
    createdAt: str


class LoginData(BaseModel):
    token: str
    userId: str
    email: str
    firstName: str
    lastName: str
    expiresAt: str


class LoginViaTokenRequest(BaseModel):
    tetheringCode: str


class LoginViaTokenData(BaseModel):
    token: str
    userId: str
    email: str
    firstName: str
    lastName: str
    expiresAt: str


class SuccessResponse(BaseModel):
    status: str = "success"
    data: dict


class NotificationResponse(BaseModel):
    id: str
    title: str
    message: str
    type: str
    is_read: bool
    created_at: str
    icon: str


class StatisticResponse(BaseModel):
    date: str
    total_usage: int
    app_usage: dict
    blocked_time: int
    device_name: str


class ProfileUpdate(BaseModel):
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    phone: Optional[str] = None
    timezone: Optional[str] = None
    theme: Optional[str] = None


class PasswordChange(BaseModel):
    currentPassword: str
    newPassword: str


class StatisticReport(BaseModel):
    device_id: str
    date: date
    total_usage: int
    app_usage: Dict[str, int]
    blocked_time: int


class UsageData(BaseModel):
    labels: List[str]
    data: List[int]


class AppUsageItem(BaseModel):
    name: str
    minutes: int
    category: Optional[str] = None


class CategoryUsageItem(BaseModel):
    name: str
    minutes: int


class ComparisonData(BaseModel):
    labels: List[str]
    currentWeek: List[int]
    previousWeek: List[int]
    currentWeekTotal: int
    previousWeekTotal: int
    trendPercentage: float


class StatsData(BaseModel):
    totalUsage: str
    mostUsedDevice: str
    avgDailyUsage: str
    blockedTime: str
    usageData: UsageData
    appUsage: List[AppUsageItem]
    categoryData: List[CategoryUsageItem]
    comparisonData: ComparisonData


class StatsResponse(BaseModel):
    status: str = "success"
    data: StatsData


class WebDeviceCreate(BaseModel):
    name: str
    model: Optional[str] = None
    osVersion: Optional[str] = None
    tetheringCode: str


class CategoryCreate(BaseModel):
    name: str
    label: Optional[str] = None
    description: Optional[str] = None
    restricted: bool
    icon: str
    time_limit: int


class CategoryResponse(BaseModel):
    id: str
    name: str
    label: str
    description: str
    restricted: bool
    created_at: str
    icon: str
    time_limit: int
    screenshot_count: int


class ScreenshotCreate(BaseModel):
    transaction_id: str
    device_id: str


class ScreenshotResponse(BaseModel):
    id: str
    image: str
    category: str
    transaction_id: str
    device_id: str
    created_at: str
    device_name: str
    category_name: Optional[str] = None
    confidence: Optional[float] = None


class ScreenTimeCreate(BaseModel):
    device_id: str
    limit: int
    schedule_start: str
    schedule_end: str
    app_name: Optional[str] = None  # Added app_name


class ScreenTimeResponse(BaseModel):
    id: str
    device_id: str
    limit: int
    schedule_start: str
    schedule_end: str
    app_name: Optional[str] = None  # Added app_name
    created_at: str


class ScreenTimeLogCreate(BaseModel):
    device_id: str
    screen_time_id: str
    screen_time: int
    timestamp: str
    activity_type: str


class ScreenTimeLogResponse(BaseModel):
    used_time: int
    remaining: int
    limit: int
    schedule_start: str
    schedule_end: str
    last_update: str


class ScreenshotFilter(BaseModel):
    category_id: Optional[str] = None
    device_id: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    page: int = 1
    per_page: int = 12


# AI endpoint models
class AI(BaseModel):
    content: str  # Base64-encoded image


class CategoryAi(BaseModel):
    categoryId: str
    categoryName: str
    confidence: float


class SuccessResponseStatusMsg(BaseModel):
    data: CategoryAi
    msg: str


class StatusError(BaseModel):
    error: str


# Error handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        loc = err.get("loc", [])
        field = loc[-1] if loc else "body"
        errors.append({"field": field, "message": err.get("msg")})
    return JSONResponse(
        status_code=400,
        content={
            "status": "error",
            "code": 400,
            "message": "Invalid input",
            "errors": errors
        }
    )


@app.exception_handler(404)
async def spa_handler(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api"):
        return JSONResponse(
            status_code=404,
            content={
                "status": "error",
                "code": 404,
                "message": "Not Found"
            }
        )
    return FileResponse(static_dir / "index.html")


# В models.py добавим новую модель
class AppRestriction(Base):
    __tablename__ = "app_restrictions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.user_id'), nullable=False)
    device_id = Column(String, ForeignKey('devices.device_id'), nullable=True)  # null означает для всех устройств
    app_name = Column(String, nullable=False)
    is_blocked = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Pydantic модель для запросов
class AppRestrictionCreate(BaseModel):
    device_id: Optional[str] = None
    app_name: str
    is_blocked: bool = True


class AppRestrictionResponse(BaseModel):
    id: str
    device_id: Optional[str] = None
    device_name: Optional[str] = None
    app_name: str
    is_blocked: bool
    created_at: str
    updated_at: str


# API эндпоинты
@app.post("/api/v1/app-restrictions", status_code=status.HTTP_201_CREATED)
async def create_app_restriction(
        payload: AppRestrictionCreate,
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        # Проверка устройства, если указано
        device_name = None
        if payload.device_id:
            device = db.query(Device).filter(
                Device.device_id == payload.device_id,
                Device.user_id == user_id
            ).first()
            if not device:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Device not found"
                )
            device_name = device.name

        # Проверяем существующее правило
        existing = db.query(AppRestriction).filter(
            AppRestriction.user_id == user_id,
            AppRestriction.device_id == payload.device_id,
            AppRestriction.app_name == payload.app_name
        ).first()

        if existing:
            existing.is_blocked = payload.is_blocked
            existing.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(existing)
            action = "updated"
        else:
            restriction = AppRestriction(
                user_id=user_id,
                device_id=payload.device_id,
                app_name=payload.app_name,
                is_blocked=payload.is_blocked
            )
            db.add(restriction)
            db.commit()
            db.refresh(restriction)
            action = "created"

        # Уведомление
        action_text = "blocked" if payload.is_blocked else "unblocked"
        device_text = f" on {device_name}" if device_name else " on all devices"
        message = f"App '{payload.app_name}' {action_text}{device_text}"

        notification = Notification(
            user_id=user_id,
            title="App Restriction Updated",
            message=message,
            type="info"
        )
        db.add(notification)
        db.commit()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": f"App restriction {action}",
                "data": {
                    "id": restriction.id if not existing else existing.id,
                    "app_name": payload.app_name,
                    "is_blocked": payload.is_blocked,
                    "device_id": payload.device_id,
                    "device_name": device_name
                }
            }
        )
    except Exception as e:
        logger.error(f"Create app restriction error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


@app.get("/api/v1/app-restrictions", response_model=List[AppRestrictionResponse])
async def get_app_restrictions(
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        restrictions = db.query(AppRestriction).filter(
            AppRestriction.user_id == user_id
        ).all()

        response = []
        for r in restrictions:
            device_name = None
            if r.device_id:
                device = db.query(Device).filter(Device.device_id == r.device_id).first()
                device_name = device.name if device else "Unknown"

            response.append({
                "id": r.id,
                "device_id": r.device_id,
                "device_name": device_name,
                "app_name": r.app_name,
                "is_blocked": r.is_blocked,
                "created_at": r.created_at.isoformat(),
                "updated_at": r.updated_at.isoformat()
            })

        return response
    except Exception as e:
        logger.error(f"Get app restrictions error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@app.delete("/api/v1/app-restrictions/{restriction_id}")
async def delete_app_restriction(
        restriction_id: str,
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        restriction = db.query(AppRestriction).filter(
            AppRestriction.id == restriction_id,
            AppRestriction.user_id == user_id
        ).first()

        if not restriction:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "status": "error",
                    "message": "Restriction not found"
                }
            )

        app_name = restriction.app_name
        device_id = restriction.device_id
        device_name = None

        if device_id:
            device = db.query(Device).filter(Device.device_id == device_id).first()
            device_name = device.name if device else "Unknown"

        db.delete(restriction)

        notification = Notification(
            user_id=user_id,
            title="App Restriction Removed",
            message=f"App '{app_name}' unblocked{' on ' + device_name if device_name else ''}",
            type="info"
        )
        db.add(notification)
        db.commit()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "App restriction removed"
            }
        )
    except Exception as e:
        logger.error(f"Delete app restriction error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )

# User registration
@app.post("/api/v1/register")
async def register(payload: UserCreate, db: Session = Depends(get_db)):
    try:
        existing = db.query(User).filter(User.email == payload.email).first()
        if existing:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "code": 400,
                    "message": "Invalid input",
                    "errors": [{"field": "email", "message": "Email already exists"}]
                }
            )

        pwd = pwd_context.hash(payload.password)
        user = User(
            firstName=payload.firstName,
            lastName=payload.lastName,
            email=payload.email,
            password=pwd,
            timezone=payload.timezone,
            phone=payload.phone,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        # Create default category for new user
        default_category = Category(
            user_id=user.user_id,
            name="Uncategorized",
            label="Other",
            description="Default category for uncategorized content",
            restricted=False,
            icon="📁"
        )
        db.add(default_category)
        db.commit()

        resp = UserResponse(
            userId=user.user_id,
            email=user.email,
            firstName=user.firstName,
            lastName=user.lastName,
            createdAt=user.createdAt.isoformat()
        )
        return JSONResponse(
            status_code=201,
            content={
                "status": "success",
                "data": resp.dict()
            }
        )
    except Exception as e:
        logger.error(f"Registration error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# User login
@app.post("/api/v1/login")
async def login(payload: LoginRequest, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.email == payload.email).first()
        if not user or not pwd_context.verify(payload.password, user.password):
            return JSONResponse(
                status_code=401,
                content={
                    "status": "error",
                    "code": 401,
                    "message": "Invalid credentials"
                }
            )

        now = datetime.now()
        expire = now + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)

        token = jwt.encode({
            "sub": user.user_id,
            "email": user.email,
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp())
        }, SECRET_KEY, algorithm=ALGORITHM)

        data = LoginData(
            token=token,
            userId=user.user_id,
            email=user.email,
            firstName=user.firstName,
            lastName=user.lastName,
            expiresAt=expire.isoformat()
        )
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": data.dict()
            }
        )
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# User logout
@app.post("/api/v1/logout")
async def logout():
    return {
        "status": "success",
        "message": "Logout successful"
    }


# Token-based authentication
@app.post("/api/v1/login-via-token", response_model=SuccessResponse)
async def login_via_token(
        payload: LoginViaTokenRequest,
        db: Session = Depends(get_db)
):
    try:
        now = datetime.utcnow()
        code_entry = db.query(TetheringCode).filter(
            TetheringCode.code == payload.tetheringCode,
            TetheringCode.used == False,
            TetheringCode.expiredAt > now
        ).first()

        if not code_entry:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "error",
                    "code": 400,
                    "message": "Invalid or expired tethering code",
                    "errors": [{
                        "field": "tetheringCode",
                        "message": "The provided code is invalid or has expired"
                    }]
                }
            )

        user = db.query(User).filter(User.user_id == code_entry.user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        notification = Notification(
            user_id=user.user_id,
            title="New Device Login",
            message=f"Logged in via tethering code",
            type="info"
        )
        db.add(notification)
        db.commit()

        expire = now + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
        token = jwt.encode({
            "sub": user.user_id,
            "email": user.email,
            "iat": int(now.timestamp()),
            "exp": int(expire.timestamp())
        }, SECRET_KEY, algorithm=ALGORITHM)

        response_data = LoginViaTokenData(
            token=token,
            userId=user.user_id,
            email=user.email,
            firstName=user.firstName,
            lastName=user.lastName,
            expiresAt=expire.isoformat()
        )

        return {
            "status": "success",
            "data": response_data.dict()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login via token failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Tethering code generation
@app.post("/api/v1/tethering-code", status_code=201)
async def create_tethering_code(
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        now = datetime.utcnow()
        code = None
        expires_at = None

        for _ in range(5):
            c = generate_code()
            exp = now + timedelta(seconds=600)
            entry = TetheringCode(code=c, user_id=user_id, expiredAt=exp, used=False)
            db.add(entry)
            try:
                db.commit()
                code = c
                expires_at = exp
                break
            except IntegrityError:
                db.rollback()

        if not code:
            raise HTTPException(
                status_code=500,
                detail="Could not generate unique tethering code"
            )

        return {
            "status": "success",
            "data": {
                "code": code,
                "expiresAt": expires_at.isoformat(),
                "validitySeconds": 600
            }
        }
    except Exception as e:
        logger.error(f"Tethering code error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Device tethering
@app.post("/api/v1/devices/tether", status_code=201)
async def tether_device(
        payload: DeviceTetherRequest,
        db: Session = Depends(get_db)
):
    try:
        now = datetime.utcnow()
        code_entry = db.query(TetheringCode).filter(
            TetheringCode.code == payload.code,
            TetheringCode.used == False,
            TetheringCode.expiredAt > now
        ).first()

        if not code_entry:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "code": 400,
                    "message": "Invalid or expired tethering code"
                }
            )

        device = Device(
            user_id=code_entry.user_id,
            name=payload.deviceName,
            model=payload.deviceModel,
            osVersion=payload.osVersion,
            isBlocked=False,
            heartbeat=now  # Added heartbeat initialization
        )
        db.add(device)

        code_entry.used = True
        db.add(code_entry)

        notification = Notification(
            user_id=code_entry.user_id,
            title="New Device Added",
            message=f"Device '{payload.deviceName}' has been tethered to your account",
            type="success"
        )
        db.add(notification)

        db.commit()
        db.refresh(device)

        return JSONResponse(
            status_code=201,
            content={
                "status": "success",
                "message": "Device tethered successfully",
                "data": {
                    "deviceId": device.device_id
                }
            }
        )
    except Exception as e:
        logger.error(f"Tether device error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Web-based device addition
@app.post("/api/v1/devices", status_code=201)
async def add_device_via_web(
        payload: WebDeviceCreate,
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        now = datetime.utcnow()
        code_entry = db.query(TetheringCode).filter(
            TetheringCode.code == payload.tetheringCode,
            TetheringCode.used == False,
            TetheringCode.expiredAt > now,
            TetheringCode.user_id == user_id
        ).first()

        if not code_entry:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "code": 400,
                    "message": "Invalid or expired tethering code"
                }
            )

        device = Device(
            user_id=user_id,
            name=payload.name,
            model=payload.model,
            osVersion=payload.osVersion,
            isBlocked=False,
            heartbeat=now  # Added heartbeat initialization
        )
        db.add(device)

        code_entry.used = True
        db.add(code_entry)

        notification = Notification(
            user_id=user_id,
            title="New Device Added",
            message=f"Device '{payload.name}' has been tethered to your account",
            type="success"
        )
        db.add(notification)

        db.commit()
        db.refresh(device)

        return JSONResponse(
            status_code=201,
            content={
                "status": "success",
                "message": "Device added successfully",
                "data": {
                    "deviceId": device.device_id
                }
            }
        )
    except Exception as e:
        logger.error(f"Add device via web error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Get devices
@app.get("/api/v1/devices")
async def get_devices(
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        devices = db.query(Device).filter(Device.user_id == user_id).all()

        device_list = []
        for device in devices:
            status = "online" if not device.isBlocked else "blocked"
            last_active = datetime.utcnow() - timedelta(minutes=5)

            device_list.append(DeviceResponse(
                deviceId=device.device_id,
                name=device.name,
                model=device.model,
                osVersion=device.osVersion,
                tetheringAt=device.tetheredAt.isoformat(),
                isBlocked=device.isBlocked,
                status=status,
                lastActive=last_active.isoformat()
            ).dict())

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "data": device_list
            }
        )
    except Exception as e:
        logger.error(f"Get devices error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Get devices for filtering
@app.get("/api/v1/devices/filter")
async def get_devices_for_filter(
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        devices = db.query(Device).filter(Device.user_id == user_id).all()
        device_list = [{"id": d.device_id, "name": d.name} for d in devices]

        return {
            "status": "success",
            "data": device_list
        }
    except Exception as e:
        logger.error(f"Get devices for filter error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Block device
@app.post("/api/v1/devices/{device_id}/block")
async def block_device(
        device_id: str,
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        device = db.query(Device).filter(
            Device.device_id == device_id,
            Device.user_id == user_id
        ).first()

        if not device:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "code": 404,
                    "message": "Device not found"
                }
            )

        device.isBlocked = True
        db.add(device)

        notification = Notification(
            user_id=user_id,
            title="Device Blocked",
            message=f"Device '{device.name}' has been blocked",
            type="alert"
        )
        db.add(notification)

        db.commit()
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Device blocked"
            }
        )
    except Exception as e:
        logger.error(f"Block device error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Unblock device
@app.post("/api/v1/devices/{device_id}/unblock")
async def unblock_device(
        device_id: str,
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        device = db.query(Device).filter(
            Device.device_id == device_id,
            Device.user_id == user_id
        ).first()

        if not device:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "code": 404,
                    "message": "Device not found"
                }
            )

        device.isBlocked = False
        db.add(device)

        notification = Notification(
            user_id=user_id,
            title="Device Unblocked",
            message=f"Device '{device.name}' has been unblocked",
            type="success"
        )
        db.add(notification)

        db.commit()
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Device unblocked"
            }
        )
    except Exception as e:
        logger.error(f"Unblock device error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Emergency lock all devices
@app.post("/api/v1/devices/emergency-lock")
async def emergency_lock_all_devices(
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        devices = db.query(Device).filter(Device.user_id == user_id).all()

        for device in devices:
            device.isBlocked = True
            db.add(device)

        notification = Notification(
            user_id=user_id,
            title="Emergency Lock Activated",
            message="All devices have been locked immediately",
            type="alert"
        )
        db.add(notification)

        db.commit()

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "All devices locked successfully"
            }
        )
    except Exception as e:
        logger.error(f"Emergency lock error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Get notifications
@app.get("/api/v1/notifications", response_model=List[NotificationResponse])
async def get_notifications(
        user_id: str = Depends(verify_token),
        limit: int = Query(10, gt=0),
        db: Session = Depends(get_db)
):
    try:
        notifications = db.query(Notification).filter(
            Notification.user_id == user_id
        ).order_by(Notification.created_at.desc()).limit(limit).all()

        icon_map = {
            "info": "ℹ️",
            "alert": "⚠️",
            "success": "✅"
        }

        return [
            NotificationResponse(
                id=n.id,
                title=n.title,
                message=n.message,
                type=n.type,
                is_read=n.is_read,
                created_at=n.created_at.isoformat(),
                icon=icon_map.get(n.type, "ℹ️")
            ) for n in notifications
        ]
    except Exception as e:
        logger.error(f"Get notifications error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


# Get recent notifications
@app.get("/api/v1/notifications/recent")
async def get_recent_notifications(
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        notifications = db.query(Notification).filter(
            Notification.user_id == user_id
        ).order_by(Notification.created_at.desc()).limit(5).all()

        icon_map = {
            "info": "ℹ️",
            "alert": "⚠️",
            "success": "✅"
        }

        return [
            {
                "id": n.id,
                "title": n.title,
                "message": n.message,
                "type": n.type,
                "icon": icon_map.get(n.type, "ℹ️"),
                "time": n.created_at.strftime("%H:%M")
            } for n in notifications
        ]
    except Exception as e:
        logger.error(f"Get recent notifications error: {str(e)}")
        return []


# Clear notifications
@app.delete("/api/v1/notifications")
async def clear_notifications(
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        db.query(Notification).filter(Notification.user_id == user_id).delete()
        db.commit()
        return {"status": "success", "message": "All notifications cleared"}
    except Exception as e:
        logger.error(f"Clear notifications error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


# Report statistics
@app.post("/api/v1/stats/report")
async def report_statistics(
        payload: StatisticReport,
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        device = db.query(Device).filter(
            Device.device_id == payload.device_id,
            Device.user_id == user_id
        ).first()

        if not device:
            return JSONResponse(
                status_code=404,
                content={
                    "status": "error",
                    "code": 404,
                    "message": "Device not found"
                }
            )

        statistic = db.query(Statistic).filter(
            Statistic.user_id == user_id,
            Statistic.device_id == payload.device_id,
            Statistic.date == payload.date
        ).first()

        if statistic:
            statistic.total_usage = payload.total_usage
            statistic.app_usage = payload.app_usage
            statistic.blocked_time = payload.blocked_time
        else:
            statistic = Statistic(
                user_id=user_id,
                device_id=payload.device_id,
                date=payload.date,
                total_usage=payload.total_usage,
                app_usage=payload.app_usage,
                blocked_time=payload.blocked_time
            )
            db.add(statistic)

        db.commit()
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Statistics saved"
            }
        )
    except Exception as e:
        logger.error(f"Save statistics error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Get statistics
@app.get("/api/v1/stats")
async def get_statistics(
        user_id: str = Depends(verify_token),
        device_id: str = Query("all"),
        period: str = Query("week"),
        db: Session = Depends(get_db)
):
    try:
        today = date.today()

        if period == "week":
            start_date = today - timedelta(days=today.weekday())
            end_date = start_date + timedelta(days=6)
        elif period == "month":
            start_date = date(today.year, today.month, 1)
            if today.month == 12:
                end_date = date(today.year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = date(today.year, today.month + 1, 1) - timedelta(days=1)
        else:  # day
            start_date = today
            end_date = today

        query = db.query(Statistic).filter(
            Statistic.user_id == user_id,
            Statistic.date.between(start_date, end_date)
        )

        if device_id != "all":
            query = query.filter(Statistic.device_id == device_id)

        stats = query.all()

        if not stats:
            return JSONResponse(
                status_code=200,
                content=StatsResponse(data=StatsData(
                    totalUsage="0m",
                    mostUsedDevice="No data",
                    avgDailyUsage="0m",
                    blockedTime="0m",
                    usageData=UsageData(labels=[], data=[]).dict(),
                    appUsage=[],
                    categoryData=[],
                    comparisonData=ComparisonData(
                        labels=[],
                        currentWeek=[],
                        previousWeek=[],
                        currentWeekTotal=0,
                        previousWeekTotal=0,
                        trendPercentage=0.0
                    ).dict()
                )).dict()
            )

        total_usage = sum(s.total_usage for s in stats)
        blocked_time = sum(s.blocked_time for s in stats)

        device_usage = defaultdict(int)
        for s in stats:
            device_usage[s.device_id] += s.total_usage

        if device_usage:
            most_used_device_id = max(device_usage, key=device_usage.get)
            device = db.query(Device).filter(Device.device_id == most_used_device_id).first()
            most_used_device = device.name if device else "Unknown device"
        else:
            most_used_device = "No devices"

        days_count = (end_date - start_date).days + 1
        avg_daily = total_usage // days_count

        usage_by_day = defaultdict(int)
        current_date = start_date
        while current_date <= end_date:
            usage_by_day[current_date] = 0
            current_date += timedelta(days=1)

        for s in stats:
            usage_by_day[s.date] += s.total_usage

        day_labels = [d.strftime("%a") for d in sorted(usage_by_day.keys())]
        day_data = [usage_by_day[d] for d in sorted(usage_by_day.keys())]

        app_usage = defaultdict(int)
        for s in stats:
            if s.app_usage:
                for app, minutes in s.app_usage.items():
                    app_usage[app] += minutes

        top_apps = sorted(app_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        app_usage_data = [
            AppUsageItem(name=app, minutes=minutes, category="")
            for app, minutes in top_apps
        ]

        prev_start = start_date - timedelta(days=7)
        prev_end = end_date - timedelta(days=7)

        prev_stats = db.query(Statistic).filter(
            Statistic.user_id == user_id,
            Statistic.date.between(prev_start, prev_end)
        ).all()

        prev_total = sum(s.total_usage for s in prev_stats) if prev_stats else 0

        trend_percentage = 0.0
        if prev_total > 0:
            trend_percentage = ((total_usage - prev_total) / prev_total) * 100

        stats_data = StatsData(
            totalUsage=minutes_to_hh_mm(total_usage),
            mostUsedDevice=most_used_device,
            avgDailyUsage=minutes_to_hh_mm(avg_daily),
            blockedTime=minutes_to_hh_mm(blocked_time),
            usageData=UsageData(
                labels=day_labels,
                data=day_data
            ).dict(),
            appUsage=[item.dict() for item in app_usage_data],
            categoryData=[],
            comparisonData=ComparisonData(
                labels=["Total Usage"],
                currentWeek=[total_usage],
                previousWeek=[prev_total],
                currentWeekTotal=total_usage,
                previousWeekTotal=prev_total,
                trendPercentage=trend_percentage
            ).dict()
        )

        return JSONResponse(
            status_code=200,
            content=StatsResponse(data=stats_data).dict())
    except Exception as e:
        logger.error(f"Get statistics error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Dashboard statistics
@app.get("/api/v1/stats/dashboard")
async def get_dashboard_stats(
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        total_usage = db.query(func.sum(Statistic.total_usage)).filter(
            Statistic.user_id == user_id
        ).scalar() or 0

        most_used_device = db.query(
            Device.name,
            func.sum(Statistic.total_usage).label('total')
        ).join(Statistic).filter(
            Statistic.user_id == user_id
        ).group_by(Device.name).order_by(desc('total')).first()

        avg_daily = db.query(func.avg(Statistic.total_usage)).filter(
            Statistic.user_id == user_id
        ).scalar() or 0

        blocked_time = db.query(func.sum(Statistic.blocked_time)).filter(
            Statistic.user_id == user_id
        ).scalar() or 0

        return {
            "status": "success",
            "data": {
                "totalUsage": minutes_to_hh_mm(total_usage),
                "mostUsedDevice": most_used_device[0] if most_used_device else "No data",
                "avgDailyUsage": minutes_to_hh_mm(avg_daily),
                "blockedTime": minutes_to_hh_mm(blocked_time)
            }
        }
    except Exception as e:
        logger.error(f"Get dashboard stats error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# App usage statistics
@app.get("/api/v1/stats/app-usage")
async def get_app_usage(
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        app_usage = defaultdict(int)
        stats = db.query(Statistic).filter(Statistic.user_id == user_id).all()

        for stat in stats:
            if stat.app_usage:
                for app, minutes in stat.app_usage.items():
                    app_usage[app] += minutes

        sorted_usage = sorted(app_usage.items(), key=lambda x: x[1], reverse=True)
        top_apps = sorted_usage[:10]

        return {
            "status": "success",
            "data": {
                "labels": [app[0] for app in top_apps],
                "data": [app[1] for app in top_apps]
            }
        }
    except Exception as e:
        logger.error(f"Get app usage error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Get profile
@app.get("/api/v1/profile")
async def get_profile(
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return {
            "status": "success",
            "data": {
                "firstName": user.firstName,
                "lastName": user.lastName,
                "email": user.email,
                "phone": user.phone,
                "timezone": user.timezone,
                "avatar": user.firstName[0] + user.lastName[0]
            }
        }
    except Exception as e:
        logger.error(f"Get profile error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


# Update profile
@app.put("/api/v1/profile")
async def update_profile(
        profile: ProfileUpdate,
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if profile.firstName is not None:
            user.firstName = profile.firstName
        if profile.lastName is not None:
            user.lastName = profile.lastName
        if profile.phone is not None:
            user.phone = profile.phone
        if profile.timezone is not None:
            user.timezone = profile.timezone

        db.commit()
        return {"status": "success", "message": "Profile updated"}
    except Exception as e:
        logger.error(f"Update profile error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


# Change password
@app.post("/api/v1/change-password")
async def change_password(
        passwords: PasswordChange,
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        if not pwd_context.verify(passwords.currentPassword, user.password):
            raise HTTPException(status_code=400, detail="Current password is incorrect")

        new_password_hash = pwd_context.hash(passwords.newPassword)
        user.password = new_password_hash
        db.commit()

        notification = Notification(
            user_id=user_id,
            title="Password Changed",
            message="Your password was successfully changed",
            type="info"
        )
        db.add(notification)
        db.commit()

        return {"status": "success", "message": "Password updated"}
    except Exception as e:
        logger.error(f"Change password error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


# Create category
@app.post("/api/v1/categories", status_code=status.HTTP_201_CREATED)
async def create_category(
        payload: CategoryCreate,
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        # If label is not provided, use name as label
        label = payload.label if payload.label else payload.name
        # If description is not provided, use empty string
        description = payload.description if payload.description else ""

        category = Category(
            user_id=user_id,
            name=payload.name,
            label=label,
            description=description,
            restricted=payload.restricted,
            icon=payload.icon,
            time_limit=payload.time_limit
        )
        db.add(category)

        notification = Notification(
            user_id=user_id,
            title="New Category Created",
            message=f"Category '{payload.name}' was created",
            type="info"
        )
        db.add(notification)
        db.commit()
        db.refresh(category)

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "status": "success",
                "message": "Category created",
                "data": {
                    "id": category.id,
                    "name": category.name,
                    "label": category.label,
                    "description": category.description,
                    "restricted": category.restricted,
                    "icon": category.icon,
                    "time_limit": category.time_limit,
                    "created_at": category.created_at.isoformat()
                }
            }
        )
    except Exception as e:
        logger.error(f"Create category error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Get categories
@app.get("/api/v1/categories", response_model=List[CategoryResponse])
async def get_categories(
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        categories = db.query(Category).filter(Category.user_id == user_id).all()

        response = []
        for cat in categories:
            screenshot_count = db.query(Screenshot).filter(
                Screenshot.user_id == user_id,
                Screenshot.category == cat.id
            ).count()

            response.append(CategoryResponse(
                id=cat.id,
                name=cat.name,
                label=cat.label,
                description=cat.description,
                restricted=cat.restricted,
                created_at=cat.created_at.isoformat(),
                icon=cat.icon,
                time_limit=cat.time_limit,
                screenshot_count=screenshot_count
            ))

        return response
    except Exception as e:
        logger.error(f"Get categories error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Delete category
@app.delete("/api/v1/categories/{category_id}")
async def delete_category(
        category_id: str,
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        category = db.query(Category).filter(
            Category.id == category_id,
            Category.user_id == user_id
        ).first()
        if not category:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "status": "error",
                    "code": 404,
                    "message": "Category not found"
                }
            )

        db.delete(category)

        notification = Notification(
            user_id=user_id,
            title="Category Deleted",
            message=f"Category '{category.name}' was deleted",
            type="warning"
        )
        db.add(notification)
        db.commit()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "Category deleted"
            }
        )
    except Exception as e:
        logger.error(f"Delete category error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Upload screenshot
@app.post("/api/v1/screenshots", status_code=status.HTTP_201_CREATED)
async def upload_screenshot(
        transaction_id: str = Form(...),
        device_id: str = Form(...),
        file: UploadFile = File(...),
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        device = db.query(Device).filter(
            Device.device_id == device_id,
            Device.user_id == user_id
        ).first()
        if not device:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "status": "error",
                    "code": 404,
                    "message": "Device not found"
                }
            )

        # Check file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        file.file.seek(0, 2)  # Move to end of file
        file_size = file.file.tell()
        file.file.seek(0)  # Return to beginning

        if file_size > max_size:
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "status": "error",
                    "message": "File size exceeds 10MB limit"
                }
            )

        # Read file content
        contents = await file.read()

        # Save file
        file_ext = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
        filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = os.path.join(uploads_dir, filename)

        with open(file_path, "wb") as buffer:
            buffer.write(contents)

        # Analyze image with AI
        category, confidence = analyze_image_and_get_category(contents, user_id, db)
        category_id = category.id if category else None

        # Use default category if needed
        if not category:
            default_category = get_default_category(db, user_id)
            category_id = default_category.id if default_category else None
            logger.warning(f"Using default category for screenshot: {filename}")

        # Create database record
        screenshot = Screenshot(
            user_id=user_id,
            device_id=device_id,
            image=filename,
            category=category_id,
            transaction_id=transaction_id
        )
        db.add(screenshot)

        notification = Notification(
            user_id=user_id,
            title="New Screenshot",
            message=f"Screenshot uploaded from {device.name}",
            type="info"
        )
        db.add(notification)

        db.commit()
        db.refresh(screenshot)

        # Get category name for response
        category_name = None
        if category_id:
            cat = db.query(Category).get(category_id)
            category_name = cat.name if cat else None

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "status": "success",
                "message": "Screenshot uploaded",
                "data": {
                    "id": screenshot.id,
                    "image": f"/uploads/{filename}",
                    "category": category_id,
                    "category_name": category_name,
                    "confidence": confidence,
                    "transaction_id": screenshot.transaction_id,
                    "device_id": screenshot.device_id,
                    "created_at": screenshot.created_at.isoformat(),
                    "device_name": device.name
                }
            })
    except Exception as e:
        logger.error(f"Upload screenshot error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Get screenshots
@app.get("/api/v1/screenshots")
async def get_screenshots(
        category_id: Optional[str] = Query(None),
        device_id: Optional[str] = Query(None),
        start_date: Optional[date] = Query(None),
        end_date: Optional[date] = Query(None),
        page: int = Query(1, ge=1),
        per_page: int = Query(12, ge=1),
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        query = db.query(Screenshot).filter(
            Screenshot.user_id == user_id
        )

        if category_id:
            query = query.filter(Screenshot.category == category_id)

        if device_id:
            query = query.filter(Screenshot.device_id == device_id)

        if start_date:
            if isinstance(start_date, str):
                try:
                    start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
                except ValueError:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "status": "error",
                            "message": "Invalid start_date format. Use YYYY-MM-DD"
                        }
                    )
            query = query.filter(Screenshot.created_at >= start_date)

        if end_date:
            if isinstance(end_date, str):
                try:
                    end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
                except ValueError:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "status": "error",
                            "message": "Invalid end_date format. Use YYYY-MM-DD"
                        }
                    )
            query = query.filter(Screenshot.created_at <= end_date)

        total = query.count()
        screenshots = query.offset((page - 1) * per_page).limit(per_page).all()

        screenshot_list = []
        for screenshot in screenshots:
            # Get category name
            category_name = None
            if screenshot.category:
                category = db.query(Category).filter(Category.id == screenshot.category).first()
                category_name = category.name if category else None

            screenshot_list.append({
                "id": screenshot.id,
                "image_url": f"/uploads/{screenshot.image}",
                "created_at": screenshot.created_at.isoformat(),
                "category": screenshot.category,
                "category_name": category_name,
                "device_id": screenshot.device_id,
                "device_name": screenshot.device.name
            })

        return {
            "status": "success",
            "data": screenshot_list,
            "pagination": {
                "total": total,
                "page": page,
                "per_page": per_page,
                "total_pages": (total + per_page - 1) // per_page
            }
        }
    except Exception as e:
        logger.error(f"Get screenshots error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Export screenshots
@app.get("/api/v1/screenshots/export")
async def export_screenshots(
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        screenshots = db.query(Screenshot).filter(
            Screenshot.user_id == user_id
        ).all()

        zip_filename = f"export_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        zip_path = uploads_dir / zip_filename

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for screenshot in screenshots:
                file_path = uploads_dir / screenshot.image
                if file_path.exists():
                    zipf.write(file_path, screenshot.image)

        return FileResponse(
            zip_path,
            media_type='application/zip',
            filename=zip_filename
        )
    except Exception as e:
        logger.error(f"Export screenshots error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Create screen time
@app.post("/api/v1/screen-time", status_code=status.HTTP_201_CREATED)
async def create_screen_time(
        payload: ScreenTimeCreate,
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        device = db.query(Device).filter(
            Device.device_id == payload.device_id,
            Device.user_id == user_id
        ).first()
        if not device:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "status": "error",
                    "code": 404,
                    "message": "Device not found"
                }
            )

        screen_time = ScreenTime(
            user_id=user_id,
            device_id=payload.device_id,
            limit=payload.limit,
            schedule_start=payload.schedule_start,
            schedule_end=payload.schedule_end,
            app_name=payload.app_name  # Added app_name
        )
        db.add(screen_time)

        notification = Notification(
            user_id=user_id,
            title="Screen Time Limit Set",
            message=f"Screen time limit set for {device.name}",
            type="info"
        )
        db.add(notification)
        db.commit()
        db.refresh(screen_time)

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "status": "success",
                "message": "Screen time limit created",
                "data": {
                    "id": screen_time.id,
                    "device_id": screen_time.device_id,
                    "limit": screen_time.limit,
                    "schedule_start": screen_time.schedule_start,
                    "schedule_end": screen_time.schedule_end,
                    "app_name": screen_time.app_name,  # Added app_name
                    "created_at": screen_time.created_at.isoformat()
                }
            }
        )
    except Exception as e:
        logger.error(f"Create screen time error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Get screen times
@app.get("/api/v1/screen-time", response_model=List[ScreenTimeResponse])
async def get_screen_times(
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        screen_times = db.query(ScreenTime).filter(
            ScreenTime.user_id == user_id
        ).all()

        return [
            ScreenTimeResponse(
                id=st.id,
                device_id=st.device_id,
                limit=st.limit,
                schedule_start=st.schedule_start,
                schedule_end=st.schedule_end,
                app_name=st.app_name,  # Added app_name
                created_at=st.created_at.isoformat()
            ) for st in screen_times
        ]
    except Exception as e:
        logger.error(f"Get screen times error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


# Update screen time
@app.put("/api/v1/screen-time/{screen_time_id}")
async def update_screen_time(
        screen_time_id: str,
        payload: ScreenTimeCreate,
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        screen_time = db.query(ScreenTime).filter(
            ScreenTime.id == screen_time_id,
            ScreenTime.user_id == user_id
        ).first()
        if not screen_time:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "status": "error",
                    "code": 404,
                    "message": "Screen time limit not found"
                }
            )

        device = db.query(Device).filter(
            Device.device_id == payload.device_id,
            Device.user_id == user_id
        ).first()
        if not device:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "status": "error",
                    "code": 404,
                    "message": "Device not found"
                }
            )

        screen_time.device_id = payload.device_id
        screen_time.limit = payload.limit
        screen_time.schedule_start = payload.schedule_start
        screen_time.schedule_end = payload.schedule_end
        screen_time.app_name = payload.app_name  # Added app_name

        notification = Notification(
            user_id=user_id,
            title="Screen Time Limit Updated",
            message=f"Screen time limit updated for {device.name}",
            type="info"
        )
        db.add(notification)
        db.commit()
        db.refresh(screen_time)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "Screen time limit updated",
                "data": {
                    "id": screen_time.id,
                    "device_id": screen_time.device_id,
                    "limit": screen_time.limit,
                    "app_name": screen_time.app_name,  # Added app_name
                    "created_at": screen_time.created_at.isoformat()
                }
            }
        )
    except Exception as e:
        logger.error(f"Update screen time error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Delete screen time
@app.delete("/api/v1/screen-time/{screen_time_id}")
async def delete_screen_time(
        screen_time_id: str,
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        screen_time = db.query(ScreenTime).filter(
            ScreenTime.id == screen_time_id,
            ScreenTime.user_id == user_id
        ).first()
        if not screen_time:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "status": "error",
                    "code": 404,
                    "message": "Screen time limit not found"
                }
            )

        device_name = screen_time.device.name if screen_time.device else "Unknown device"

        db.delete(screen_time)

        notification = Notification(
            user_id=user_id,
            title="Screen Time Limit Removed",
            message=f"Screen time limit removed for {device_name}",
            type="warning"
        )
        db.add(notification)
        db.commit()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "Screen time limit deleted"
            }
        )
    except Exception as e:
        logger.error(f"Delete screen time error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Log screen time
@app.post("/api/v1/screen-time/log", status_code=status.HTTP_201_CREATED)
async def log_screen_time(
        payload: ScreenTimeLogCreate,
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        device = db.query(Device).filter(
            Device.device_id == payload.device_id,
            Device.user_id == user_id
        ).first()
        if not device:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "status": "error",
                    "code": 404,
                    "message": "Device not found"
                }
            )

        screen_time = db.query(ScreenTime).filter(
            ScreenTime.id == payload.screen_time_id,
            ScreenTime.user_id == user_id
        ).first()
        if not screen_time:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "status": "error",
                    "code": 404,
                    "message": "Screen time limit not found"
                }
            )

        log = ScreenTimeLog(
            user_id=user_id,
            device_id=payload.device_id,
            screen_time_id=payload.screen_time_id,
            screen_time=payload.screen_time,
            timestamp=datetime.fromisoformat(payload.timestamp),
            activity_type=payload.activity_type
        )
        db.add(log)
        db.commit()

        logs = db.query(ScreenTimeLog).filter(
            ScreenTimeLog.screen_time_id == payload.screen_time_id
        ).all()

        used_time = sum(log.screen_time for log in logs)
        remaining = max(0, screen_time.limit - used_time)

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "status": "success",
                "message": "Screen time logged",
                "data": {
                    "used_time": used_time,
                    "remaining": remaining,
                    "limit": screen_time.limit,
                    "schedule_start": screen_time.schedule_start,
                    "schedule_end": screen_time.schedule_end,
                    "last_update": datetime.utcnow().isoformat()
                }
            }
        )
    except Exception as e:
        logger.error(f"Log screen time error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


@app.post("/api/v1/heartbeat")
async def heartbeat(device_id: str, db: Session = Depends(get_db)):
    try:
        device = db.query(Device).filter(Device.device_id == device_id).first()
        if not device:
            raise HTTPException(status_code=404, detail="Device not found")

        device.heartbeat = datetime.utcnow()
        db.add(device)
        db.commit()

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "Heartbeat updated"
            }
        )
    except Exception as e:
        logger.error(f"Heartbeat error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": "Internal server error"
            }
        )


# Health check
@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "message": "Server is running"}


# Ping endpoint
@app.get("/api/v1/ping")
async def ping():
    return {"status": "success", "message": "pong"}


# AI endpoint
@app.post("/api/v1/ai")
async def moderate(
        payload: AI,
        user_id: str = Depends(verify_token),
        db: Session = Depends(get_db)
):
    try:
        # Decode base64 image
        try:
            image_data = base64.b64decode(payload.content)
        except Exception as e:
            logger.error(f"Base64 decoding error: {str(e)}")
            return JSONResponse(
                status_code=400,
                content=StatusError(error="Invalid base64 encoding").model_dump()
            )

        # Get all categories
        categories = db.query(Category).filter(Category.user_id == user_id).all()
        categories_names = [c.name for c in categories]

        # Analyze image
        report = moderate_screenshot(image_data, categories_names)

        if "category" in report:
            # Find or create category
            category = next((c for c in categories if c.name == report["category"]), None)

            if not category:
                category = Category(
                    user_id=user_id,
                    name=report["category"],
                    label=report["category"],
                    description="Auto-generated category",
                    restricted=False,
                    icon="📱"
                )
                db.add(category)
                db.commit()
                db.refresh(category)

            # Prepare response
            response_data = CategoryAi(
                categoryId=category.id,
                categoryName=report["category"],
                confidence=report.get("confidence", 0.0)
            )

            return JSONResponse(
                status_code=200,
                content=SuccessResponseStatusMsg(
                    data=response_data.dict(),
                    msg="Content analyzed"
                ).model_dump()
            )

        return JSONResponse(
            status_code=400,
            content=StatusError(error="Invalid content type").model_dump()
        )
    except Exception as e:
        logger.error(f"AI moderation error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=StatusError(error="Internal server error").model_dump()
        )


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    # Generate base schema
    schema = get_openapi(
        title="FamilyGuard API",
        version="1.0.0",
        description="Parental control and device management system",
        routes=app.routes,
    )

    # Add security scheme
    schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }

    # Apply security to protected routes
    secured_paths = [
        "/api/v1/devices",
        "/api/v1/tethering-code",
        "/api/v1/schedules",
        "/api/v1/notifications",
        "/api/v1/stats",
        "/api/v1/profile",
        "/api/v1/categories",
        "/api/v1/screenshots",
        "/api/v1/screen-time",
        "/api/v1/ai",
        "/api/v1/heartbeat"  # Added heartbeat
    ]

    for path, methods in schema["paths"].items():
        if any(p in path for p in secured_paths):
            for method in methods.values():
                method.setdefault("security", []).append({"BearerAuth": []})

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi

# Mount static files
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

# Run server
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
