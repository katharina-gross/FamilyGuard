from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr, root_validator, model_validator, Field, ConfigDict


class UserBase(BaseModel):
    email: EmailStr
    firstName: str
    lastName: str
    timezone: str
    phone: str


class UserCreate(UserBase):
    password: str
    confirmPassword: str

    @model_validator(mode='after')
    def check_password(cls, values):
        if values.password != values.confirmPassword:
            raise ValueError('Passwords do not match')
        return values

class UserResponse(BaseModel):
    userId: str
    email: str
    firstName: str
    lastName: str
    createdAt: str

    model_config = ConfigDict(from_attributes=True)


class DeviceInfo(BaseModel):
    deviceId: str
    platform: str
    appVersion: str

    model_config = ConfigDict(from_attributes=True)

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    deviceInfo: DeviceInfo

class LoginData(BaseModel):
    token: str
    userId: str
    email: EmailStr
    firstName: str
    lastName: str
    expiresAt: str

    model_config = ConfigDict(from_attributes=True)

class LoginViaTokenRequest(BaseModel):
    tetheringCode: str

class LoginViaTokenData(BaseModel):
    token: str
    userId: str
    email: EmailStr
    firstName: str
    lastName: str
    expiresAt: str

    model_config = ConfigDict(from_attributes=True)



class ScreenshotBase(BaseModel):
    category: str
    transaction_id: str
    device_id: str

class ScreenshotCreate(ScreenshotBase):
    pass

class ScreenshotResponse(ScreenshotBase):
    id: str
    image: str
    model_config = ConfigDict(from_attributes=True)

class Context(BaseModel):
    appName: str
    packageName: str

class AI(BaseModel):
    deviceId: str
    contentType: str
    content: str
    context: Context

class DeviceBase(BaseModel):
    name: str
    model: str
    osVersion: str


class DeviceCreate(DeviceBase):
    tetheringCode: str


class DeviceResponse(BaseModel):
    deviceId: str
    name: str
    model: str
    osVersion: str
    tetheringAt: str

    model_config = ConfigDict(from_attributes=True)

class TetherResponse(BaseModel):
    code: str
    expiredAt: str
    validitySeconds: int

    model_config = ConfigDict(from_attributes=True)

class CategoryBase(BaseModel):
    name: str
    label: str
    description: str
    restricted: bool


class CategoryCreate(CategoryBase):
    pass

class CategoryResponse(BaseModel):
    categoryId: str
    name: str
    label: str
    description: str
    restricted: bool

    model_config = ConfigDict(from_attributes=True)

class ScheduleItem(BaseModel):
    start: str
    end: str

class ScreentimeBase(BaseModel):
    limit: int
    schedule: ScheduleItem

class ScreentimeCreate(ScreentimeBase):
    deviceId: str

class ScreentimeResponse(ScreentimeBase):
    screentimeId: str
    model_config = ConfigDict(from_attributes=True)

class ScreentimeUpdate(BaseModel):
    limit: Optional[int]=None
    schedule: Optional[ScheduleItem]=None
    deviceId: Optional[str]=None

class LogScreentimeBase(BaseModel):
    screenTime: int
    timestamp: str
    screentime_id: str
    activityType: str
    device_id: str

class LogScreentimeCreate(LogScreentimeBase):
    pass

class LogScreentimeResponse(BaseModel):
    usedTime: int
    remaining: int
    limit: int
    schedule: ScheduleItem
    lastUpdate: str
    model_config = ConfigDict(from_attributes=True)

# class LogScreentime

class ErrorItem(BaseModel):
    field: str
    message: str

    model_config = ConfigDict(from_attributes=True)


class ErrorResponse(BaseModel):

    status: str = Field("error")
    code: int
    message: str
    errors: List[ErrorItem]

    model_config = ConfigDict(from_attributes=True)

class SuccessResponse(BaseModel):
    status: str = Field("success")
    data: object

    model_config = ConfigDict(from_attributes=True)

class SuccessResponseMsg(BaseModel):
    status: str = Field("success")
    msg: str

    model_config = ConfigDict(from_attributes=True)

class SuccessResponseStatusMsg(BaseModel):
    status: str = Field("success")
    data: object
    msg: str
    model_config = ConfigDict(from_attributes=True)

class CategoryAi(BaseModel):
    categoryId: str
    categoryName: str
    confidence: float

class StatusError(BaseModel):
    status: str = Field("error")
    error: str

    model_config = ConfigDict(from_attributes=True)
