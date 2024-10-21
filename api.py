# settings.py
from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    # API Keys
    OPENWEATHER_API_KEY: str = os.getenv('OPENWEATHER_API_KEY', '')
    API_KEY: Optional[str] = os.getenv('API_KEY')
    GEODB_API_KEY: Optional[str] = os.getenv('GEODB_API_KEY')
    
    # MongoDB settings
    MONGODB_URL: str = os.getenv('MONGODB_URL', "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv('MONGODB_DB_NAME', "weather")
    
    # Email settings
    EMAIL_HOST: str = os.getenv('SMTP_HOST', "smtp.gmail.com")
    EMAIL_PORT: int = int(os.getenv('SMTP_PORT', "587"))
    EMAIL_USER: Optional[str] = os.getenv('SMTP_USERNAME')
    EMAIL_PASSWORD: Optional[str] = os.getenv('SMTP_PASSWORD')
    
    # Application settings
    UPDATE_INTERVAL: int = int(os.getenv('UPDATE_INTERVAL', "5"))
    ALERT_CHECK_INTERVAL: int = int(os.getenv('ALERT_CHECK_INTERVAL', "1"))
    TEMPERATURE_UNIT: str = os.getenv('TEMPERATURE_UNIT', "celsius")
    ANOMALY_DETECTION_THRESHOLD: float = float(os.getenv('ANOMALY_DETECTION_THRESHOLD', "-0.5"))
    MAX_RETRIES: int = int(os.getenv('MAX_RETRIES', "3"))
    RETRY_DELAY: int = int(os.getenv('RETRY_DELAY', "1"))
    
    # Cache settings
    GEOCODING_CACHE_TTL: int = int(os.getenv('GEOCODING_CACHE_TTL', "86400"))
    GEOCODING_CACHE_SIZE: int = int(os.getenv('GEOCODING_CACHE_SIZE', "1000"))

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
        "extra": "allow"
    }

@lru_cache()
def get_settings():
    return Settings()

# main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security, status, Request, Response,Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Annotated
from enum import Enum
from pydantic import BaseModel, EmailStr, validator, Field
import httpx
import asyncio
import pytz
from collections import Counter
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from cachetools import TTLCache
import smtplib

# Get settings
settings = get_settings()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Weather API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key verification
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    if settings.API_KEY and api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key

# Setup MongoDB
client = AsyncIOMotorClient(settings.MONGODB_URL)
db = client[settings.MONGODB_DB_NAME]

# Initialize scheduler
scheduler = AsyncIOScheduler()

# Cache for geocoding results
geocoding_cache = TTLCache(
    maxsize=settings.GEOCODING_CACHE_SIZE,
    ttl=settings.GEOCODING_CACHE_TTL
)

class WeatherCondition(str, Enum):
    THUNDERSTORM = "Thunderstorm"
    RAIN = "Rain"
    SNOW = "Snow"
    CLEAR = "Clear"
    CLOUDS = "Clouds"
    MIST = "Mist"
    HAZE = "Haze"
    DRIZZLE = "Drizzle"
    SMOKE = "Smoke"
    DUST = "Dust"
    FOG = "Fog"
    SAND = "Sand"
    ASH = "Ash"
    SQUALL = "Squall"
    TORNADO = "Tornado"

class AlertType(str, Enum):
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    WIND = "wind"
    SEVERE_WEATHER = "severe_weather"

class AlertThreshold(BaseModel):
    alert_type: AlertType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    consecutive_readings: int = Field(ge=1, le=10, default=2)
    email_notifications: bool = True
    notification_email: Optional[EmailStr] = None
    sms_notifications: bool = False
    phone_number: Optional[str] = None

    @validator('min_value', 'max_value')
    def validate_thresholds(cls, v, values):
        if v is None and 'min_value' not in values and 'max_value' not in values:
            raise ValueError("At least one threshold (min or max) must be set")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "alert_type": "temperature",
                "min_value": 0,
                "max_value": 30,
                "consecutive_readings": 2,
                "email_notifications": True,
                "notification_email": "user@example.com"
            }]
        }
    }

class WeatherForecast(BaseModel):
    city: str
    timestamp: datetime
    temperature: float
    feels_like: float
    humidity: float
    wind_speed: float
    probability_of_precipitation: float
    condition: WeatherCondition
    pressure: float
    visibility: Optional[float] = None
    uv_index: Optional[float] = None
    description: str

    model_config = {
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }

class GeocodingService:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="weather_api")
        
    async def get_coordinates(self, city: str) -> tuple:
        if city in geocoding_cache:
            return geocoding_cache[city]
            
        try:
            location = await asyncio.to_thread(self.geolocator.geocode, city)
            if location:
                coords = (location.latitude, location.longitude)
                geocoding_cache[city] = coords
                return coords
            raise HTTPException(status_code=404, detail=f"City '{city}' not found")
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            logger.error(f"Geocoding error for city {city}: {str(e)}")
            raise HTTPException(status_code=503, detail="Geocoding service unavailable")

class WeatherService:
    def __init__(self):
        self.api_key = settings.OPENWEATHER_API_KEY
        self.geocoding_service = GeocodingService()
        
    async def get_weather_data(self, city: str, forecast_days: int = 5) -> List[WeatherForecast]:
        if not self.api_key:
            raise HTTPException(status_code=503, detail="OpenWeather API key not configured")
            
        coords = await self.geocoding_service.get_coordinates(city)
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "https://api.openweathermap.org/data/2.5/forecast",
                    params={
                        "lat": coords[0],
                        "lon": coords[1],
                        "appid": self.api_key,
                        "units": "metric"
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                return self._process_weather_data(data, city)
            except httpx.HTTPError as e:
                logger.error(f"OpenWeather API error: {str(e)}")
                raise HTTPException(status_code=503, detail="Weather service unavailable")

    def _process_weather_data(self, data: Dict, city: str) -> List[WeatherForecast]:
        forecasts = []
        for item in data['list']:
            try:
                forecast = WeatherForecast(
                    city=city,
                    timestamp=datetime.fromtimestamp(item['dt'], tz=pytz.UTC),
                    temperature=item['main']['temp'],
                    feels_like=item['main']['feels_like'],
                    humidity=item['main']['humidity'],
                    wind_speed=item['wind']['speed'],
                    probability_of_precipitation=item.get('pop', 0) * 100,
                    condition=WeatherCondition(item['weather'][0]['main']),
                    pressure=item['main']['pressure'],
                    visibility=item.get('visibility'),
                    uv_index=item.get('uvi'),
                    description=item['weather'][0]['description']
                )
                forecasts.append(forecast)
            except (KeyError, ValueError) as e:
                logger.warning(f"Error processing forecast item: {str(e)}")
                continue
        return forecasts

class AlertManager:
    def __init__(self, db):
        self.db = db
        
    async def process_alert(self, city: str, weather_data: WeatherForecast, threshold: AlertThreshold):
        if await self.should_trigger_alert(city, weather_data, threshold):
            await self.send_alert(city, weather_data, threshold)
    
    async def should_trigger_alert(self, city: str, weather_data: WeatherForecast, threshold: AlertThreshold) -> bool:
        last_alert = await self.db.alerts.find_one({
            "city": city,
            "alert_type": threshold.alert_type,
            "timestamp": {"$gte": datetime.now(pytz.UTC) - timedelta(hours=1)}
        })
        return last_alert is None
    
    async def send_alert(self, city: str, weather_data: WeatherForecast, threshold: AlertThreshold):
        if threshold.email_notifications and threshold.notification_email:
            await self.send_email_alert(city, weather_data, threshold)
        
        await self.db.alerts.insert_one({
            "city": city,
            "alert_type": threshold.alert_type,
            "weather_data": weather_data.model_dump(),
            "threshold": threshold.model_dump(),
            "timestamp": datetime.now(pytz.UTC)
        })

    async def send_email_alert(self, city: str, weather_data: WeatherForecast, threshold: AlertThreshold):
        if not all([settings.EMAIL_USER, settings.EMAIL_PASSWORD]):
            logger.warning("Email credentials not configured")
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = settings.EMAIL_USER
            msg['To'] = threshold.notification_email
            msg['Subject'] = f"Weather Alert for {city}"
            
            body = self._create_alert_email_body(city, weather_data, threshold)
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(settings.EMAIL_HOST, settings.EMAIL_PORT) as server:
                server.starttls()
                server.login(settings.EMAIL_USER, settings.EMAIL_PASSWORD)
                server.send_message(msg)
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")

    def _create_alert_email_body(self, city: str, weather_data: WeatherForecast, threshold: AlertThreshold) -> str:
        return f"""
Weather Alert for {city}

Condition: {threshold.alert_type}
Current Value: {getattr(weather_data, threshold.alert_type, 'N/A')}
Threshold: {threshold.min_value if threshold.min_value is not None else 'N/A'} - {threshold.max_value if threshold.max_value is not None else 'N/A'}

Current Weather Conditions:
Temperature: {weather_data.temperature}°C
Humidity: {weather_data.humidity}%
Wind Speed: {weather_data.wind_speed} m/s

This is an automated alert. Please check your weather app for more details.
"""
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status_code": exc.status_code,
            "detail": exc.detail
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "detail": "Internal server error"
        }
    )
# API endpoints
@app.get("/api/weather/forecast/{city}")
async def get_weather_forecast(
    city: str,
    days: Annotated[int, Query(ge=1, le=7)] = 5,
) -> List[WeatherForecast]:
    """Get weather forecast for any city."""
    weather_service = WeatherService()
    forecasts = await weather_service.get_weather_data(city, days)
    return forecasts[:days * 8]  # 8 forecasts per day

@app.get("/api/weather/current/{city}")
async def get_current_weather(
    city: str,
) -> Optional[WeatherForecast]:
    """Get current weather for any city."""
    weather_service = WeatherService()
    forecasts = await weather_service.get_weather_data(city, 1)
    return forecasts[0] if forecasts else None

@app.get("/api/weather/statistics/{city}")
async def get_weather_statistics(
    city: str,
    start_date: datetime = None,
    end_date: datetime = None
)-> JSONResponse:
    """Get weather statistics for any city."""
    if start_date is None:
        start_date = datetime.now(pytz.UTC) - timedelta(days=7)
    if end_date is None:
        end_date = datetime.now(pytz.UTC)

    try:
        pipeline = [
            {
                "$match": {
                    "city": city,
                    "timestamp": {
                        "$gte": start_date,
                        "$lte": end_date
                    }
                }
            },
            {
                "$group": {
                    "_id": None,
                    "avg_temp": {"$avg": "$temperature"},
                    "max_temp": {"$max": "$temperature"},
                    "min_temp": {"$min": "$temperature"},
                    "avg_humidity": {"$avg": "$humidity"},
                    "avg_wind_speed": {"$avg": "$wind_speed"},
                    "conditions": {"$push": "$condition"}
                }
            }
        ]
        
        stats = await db.weather_records.aggregate(pipeline).to_list(None)
        if not stats:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "message": "No historical data available",
                    "city": city,
                    "period": {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat()
                    }
                }
            )
        
        stats = stats[0]
        conditions = stats.pop("conditions", [])
        condition_frequency = dict(Counter(conditions))
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                **{k: round(v, 2) if isinstance(v, (float, int)) else v 
                   for k, v in stats.items() if k != "_id"},
                "condition_frequency": condition_frequency,
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                }
            }
        )
    except Exception as e:
        logger.error(f"Error retrieving statistics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve weather statistics"
        )

@app.post("/api/alerts/configure/{city}")
async def configure_alert(
    city: str,
    alert_type: AlertType = Query(...),
    min_value: Optional[float] = Query(None),
    max_value: Optional[float] = Query(None),
    consecutive_readings: int = Query(default=2, ge=1, le=10),
    email_notifications: bool = Query(default=True),
    notification_email: Optional[EmailStr] = Query(None),
    api_key: str = Depends(verify_api_key)
) -> JSONResponse:
    """Configure weather alerts for any city."""
    try:
        # Create AlertThreshold object from query parameters
        threshold = AlertThreshold(
            alert_type=alert_type,
            min_value=min_value,
            max_value=max_value,
            consecutive_readings=consecutive_readings,
            email_notifications=email_notifications,
            notification_email=notification_email
        )
        
        # Validate the city exists by attempting to get its coordinates
        weather_service = WeatherService()
        await weather_service.geocoding_service.get_coordinates(city)
        
        # Proceed with alert configuration
        await db.alert_thresholds.update_one(
            {"city": city, "alert_type": threshold.alert_type},
            {"$set": threshold.model_dump()},
            upsert=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Alert configured successfully",
                "city": city,
                "alert_type": threshold.alert_type,
                "threshold": threshold.model_dump()
            }
        )
    except ValidationError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve)
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error configuring alert: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to configure alert: {str(e)}"
        )
@app.get("/api/alerts/{city}")
async def get_alerts(
    city: str,
    api_key: Optional[str] = Depends(verify_api_key)
)-> JSONResponse:
    """Get configured alerts for a city."""
    try:
        alerts = await db.alert_thresholds.find({"city": city}).to_list(None)
        if not alerts:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=[]
            )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=[AlertThreshold(**alert).model_dump() for alert in alerts]
        )
    except Exception as e:
        logger.error(f"Error retrieving alerts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve alerts"
        )

@app.delete("/api/alerts/{city}/{alert_type}")
async def delete_alert(
    city: str,
    alert_type: AlertType,
    api_key: Optional[str] = Depends(verify_api_key)
)-> JSONResponse:
    """Delete a configured alert for a city."""
    try:
        result = await db.alert_thresholds.delete_one({
            "city": city,
            "alert_type": alert_type
        })
        if result.deleted_count:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": f"Alert {alert_type} deleted for {city}"}
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error deleting alert: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete alert"
        )

# Background Tasks
async def check_weather_alerts():
    """Background task to check weather conditions against configured alerts."""
    try:
        weather_service = WeatherService()
        alert_manager = AlertManager(db)
        
        # Get all cities with configured alerts
        cities = await db.alert_thresholds.distinct("city")
        
        for city in cities:
            try:
                # Get current weather
                forecasts = await weather_service.get_weather_data(city, 1)
                if not forecasts:
                    continue
                
                current_weather = forecasts[0]
                
                # Get all alerts for this city
                alerts = await db.alert_thresholds.find({"city": city}).to_list(None)
                
                # Process each alert
                for alert in alerts:
                    threshold = AlertThreshold(**alert)
                    await alert_manager.process_alert(city, current_weather, threshold)
                    
            except Exception as e:
                logger.error(f"Error processing alerts for city {city}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in check_weather_alerts: {str(e)}")

async def store_weather_data():
    """Background task to store weather data for historical analysis."""
    try:
        weather_service = WeatherService()
        
        # Get all cities with configured alerts
        cities = await db.alert_thresholds.distinct("city")
        
        for city in cities:
            try:
                # Get current weather
                forecasts = await weather_service.get_weather_data(city, 1)
                if not forecasts:
                    continue
                
                current_weather = forecasts[0]
                
                # Store in database
                await db.weather_records.insert_one(
                    current_weather.model_dump()
                )
                
            except Exception as e:
                logger.error(f"Error storing weather data for city {city}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in store_weather_data: {str(e)}")

async def cleanup_old_data():
    """Background task to clean up old weather records and alerts."""
    try:
        # Keep only last 30 days of weather records
        thirty_days_ago = datetime.now(pytz.UTC) - timedelta(days=30)
        await db.weather_records.delete_many({
            "timestamp": {"$lt": thirty_days_ago}
        })
        
        # Keep only last 7 days of alert history
        seven_days_ago = datetime.now(pytz.UTC) - timedelta(days=7)
        await db.alerts.delete_many({
            "timestamp": {"$lt": seven_days_ago}
        })
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_data: {str(e)}")

# Application Startup and Shutdown Events
@app.on_event("startup")
async def startup_event():
    try:
        # Create indexes
        await db.weather_records.create_index(
            [("timestamp", -1), ("city", 1)],
            background=True
        )
        await db.alert_thresholds.create_index(
            [("city", 1), ("alert_type", 1)],
            unique=True,
            background=True
        )
        await db.alerts.create_index(
            [("timestamp", -1)],
            background=True
        )
        
        # Initialize alert manager
        app.state.alert_manager = AlertManager(db)
        
        # Schedule background tasks
        scheduler.add_job(
            check_weather_alerts,
            'interval',
            minutes=settings.ALERT_CHECK_INTERVAL,
            id='check_weather_alerts',
            replace_existing=True
        )
        
        scheduler.add_job(
            store_weather_data,
            'interval',
            minutes=settings.UPDATE_INTERVAL,
            id='store_weather_data',
            replace_existing=True
        )
        
        scheduler.add_job(
            cleanup_old_data,
            'cron',
            hour=0,  # Run at midnight
            id='cleanup_old_data',
            replace_existing=True
        )
        
        # Start scheduler
        scheduler.start()
        
        logger.info("Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    try:
        # Shutdown scheduler
        scheduler.shutdown()
        
        # Close MongoDB connection
        await client.close()
        
        logger.info("Application shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
        raise

# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "status_code": exc.status_code,
        "detail": exc.detail
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return {
        "status_code": 500,
        "detail": "Internal server error"
    }

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )