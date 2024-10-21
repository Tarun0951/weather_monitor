import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Optional
import altair as alt
from streamlit_option_menu import option_menu
import json
import numpy as np
from dataclasses import dataclass

@dataclass
class DashboardTheme:
    PRIMARY_COLOR = "#1E88E5"
    SECONDARY_COLOR = "#FFC107"
    BACKGROUND_COLOR = "#FFFFFF"
    CARD_BACKGROUND = "#F8F9FA"
    SUCCESS_COLOR = "#4CAF50"
    WARNING_COLOR = "#FF9800"
    DANGER_COLOR = "#F44336"
    TEXT_COLOR = "#212121"
    FONT_FAMILY = "Inter, sans-serif"

class DashboardConfig:
    def __init__(self):
        # Load configuration from session state or initialize defaults
        self.api_base_url = st.session_state.get('api_base_url', "http://localhost:8000/api")
        self.api_key = st.session_state.get('api_key', "")
        self.email_host = st.session_state.get('email_host', "smtp.gmail.com")
        self.email_port = st.session_state.get('email_port', 587)
        self.email_user = st.session_state.get('email_user', "")
        self.email_password = st.session_state.get('email_password', "")
        
        self.cities = [
           "Hyderabad",
           "Chennai",
           "Delhi",
           "Kolkata",
           "Mumbai",
           "Bangalore",
           "Pune",
           "Ahmedabad",
           "Jaipur",
           "Surat",
           "Lucknow",
           "Kanpur",
           "Nagpur",
           "Indore",
           "Thane",
           "Bhopal",
           "Visakhapatnam",
           "Patna",
           "Vadodara",
           "Agra",
           "Nashik",
           "Ranchi",
           "Gwalior",
           "Vijayawada",
           "Madurai",
           "Aurangabad",
           "Coimbatore",
           "Jodhpur",
           "Varanasi",
           "Meerut",
           "Allahabad"    
        ]
        
        self.weather_icons = {
            "Clear": "☀️",
            "Clouds": "☁️",
            "Rain": "🌧️",
            "Thunderstorm": "⛈️",
            "Snow": "❄️",
            "Mist": "🌫️",
            "Haze": "😶‍🌫️",
            "Drizzle": "🌦️",
            "Fog": "🌫️",
            "Smoke": "💨",
            "Dust": "😷",
            "Sand": "🏜️",
            "Ash": "🌋",
            "Squall": "💨",
            "Tornado": "🌪️"
        }

    @property
    def headers(self) -> Dict:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def save_config(self):
        st.session_state['api_base_url'] = self.api_base_url
        st.session_state['api_key'] = self.api_key
        st.session_state['email_host'] = self.email_host
        st.session_state['email_port'] = self.email_port
        st.session_state['email_user'] = self.email_user
        st.session_state['email_password'] = self.email_password

class WeatherAPI:
    def __init__(self, config: DashboardConfig):
        self.config = config

    def _make_request(self, endpoint: str, method: str = "GET", params: Dict = None) -> Dict:
        url = f"{self.config.api_base_url}/{endpoint}"
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=self.config.headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            st.error(f"API Error: {str(e)}")
            return {}

    def get_current_weather(self, city: str) -> Dict:
        return self._make_request(f"weather/current/{city}")

    def get_forecast(self, city: str, days: int = 5) -> List[Dict]:
        return self._make_request(f"weather/forecast/{city}", params={"days": days})

    def get_statistics(self, city: str, start_date: datetime, end_date: datetime) -> Dict:
        params = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
        return self._make_request(f"weather/statistics/{city}", params=params)

    def configure_alert(self, city: str, alert_config: Dict) -> Dict:
        return self._make_request(
            f"alerts/configure/{city}",
            method="POST",
            params=alert_config
        )

class Dashboard:
    def __init__(self):
        st.set_page_config(
            page_title="Weather Monitoring Dashboard",
            page_icon="🌤️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize config and API
        self.config = DashboardConfig()
        self.api = WeatherAPI(self.config)
        
        # Apply custom CSS
        self.apply_custom_css()

    def apply_custom_css(self):
        st.markdown(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family={DashboardTheme.FONT_FAMILY}&display=swap');
            
            /* Global Styles */
            body {{
                font-family: {DashboardTheme.FONT_FAMILY};
                color: {DashboardTheme.TEXT_COLOR};
                background-color: {DashboardTheme.BACKGROUND_COLOR};
            }}
            
            /* Weather Card Styles */
            .weather-card {{
                background: {DashboardTheme.CARD_BACKGROUND};
                border-radius: 15px;
                padding: 1.5rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
                transition: transform 0.2s;
            }}
            
            .weather-card:hover {{
                transform: translateY(-5px);
            }}
            
            .weather-card h3 {{
                color: {DashboardTheme.PRIMARY_COLOR};
                margin-bottom: 1rem;
            }}
            
            .weather-card .temperature {{
                font-size: 2.5rem;
                font-weight: bold;
                color: {DashboardTheme.TEXT_COLOR};
            }}
            
            .metric-container {{
                background: {DashboardTheme.CARD_BACKGROUND};
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            }}
            
            .metric-label {{
                font-size: 0.9rem;
                color: {DashboardTheme.TEXT_COLOR};
                opacity: 0.8;
            }}
            
            .metric-value {{
                font-size: 1.5rem;
                font-weight: bold;
                color: {DashboardTheme.PRIMARY_COLOR};
            }}
        </style>
        """, unsafe_allow_html=True)

    def create_weather_card(self, data: Dict):
        if not data:
            return
            
        icon = self.config.weather_icons.get(data.get('condition', ''), "🌡️")
        
        st.markdown(f"""
        <div class="weather-card">
            <h3>{data.get('city', 'Unknown')} {icon}</h3>
            <div class="temperature">{data.get('temperature', 0):.1f}°C</div>
            <p>Feels like: {data.get('feels_like', 0):.1f}°C</p>
            <div class="weather-details">
                <span>💧 {data.get('humidity', 0)}%</span> | 
                <span>💨 {data.get('wind_speed', 0)} m/s</span>
            </div>
            <p class="description">{data.get('description', '').capitalize()}</p>
        </div>
        """, unsafe_allow_html=True)

    def render_settings_page(self):
        st.title("⚙️ Dashboard Settings")
        
        with st.form("dashboard_settings"):
            st.subheader("API Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                api_base_url = st.text_input(
                    "API Base URL",
                    value=self.config.api_base_url,
                    help="The base URL for the weather API"
                )
                api_key = st.text_input(
                    "API Key",
                    value=self.config.api_key,
                    type="password",
                    help="Your API key for authentication"
                )
            
            st.subheader("Email Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                email_host = st.text_input(
                    "SMTP Host",
                    value=self.config.email_host,
                    help="SMTP server host for email notifications"
                )
                email_port = st.number_input(
                    "SMTP Port",
                    value=self.config.email_port,
                    help="SMTP server port"
                )
            
            with col2:
                email_user = st.text_input(
                    "Email Username",
                    value=self.config.email_user,
                    help="Email address for sending notifications"
                )
                email_password = st.text_input(
                    "Email Password",
                    value=self.config.email_password,
                    type="password",
                    help="Email account password or app-specific password"
                )
            
            if st.form_submit_button("Save Settings"):
                # Update config
                self.config.api_base_url = api_base_url
                self.config.api_key = api_key
                self.config.email_host = email_host
                self.config.email_port = email_port
                self.config.email_user = email_user
                self.config.email_password = email_password
                
                # Save to session state
                self.config.save_config()
                
                # Test API connection
                test_response = self.api.get_current_weather(self.config.cities[0])
                if test_response:
                    st.success("✅ Settings saved and API connection successful!")
                else:
                    st.warning("⚠️ Settings saved but API connection failed. Please check your configuration.")
    def render_overview_page(self):
        st.title("🌤️ Weather Overview")
        
        # City selection
        selected_cities = st.multiselect(
            "Select Cities to Monitor",
            self.config.cities,
            default=self.config.cities[:3]
        )
        
        if not selected_cities:
            st.warning("Please select at least one city to monitor")
            return
        
        # Create columns for weather cards
        cols = st.columns(3)
        for idx, city in enumerate(selected_cities):
            with cols[idx % 3]:
                weather_data = self.api.get_current_weather(city)
                self.create_weather_card(weather_data)

    def render_forecast_page(self):
        st.title("📅 Weather Forecast")
        
        # City selection
        selected_city = st.selectbox("Select City", self.config.cities)
        forecast_days = st.slider("Forecast Days", 1, 7, 5)
        
        # Get forecast data
        forecast_data = self.api.get_forecast(selected_city, days=forecast_days)
        
        if not forecast_data:
            st.warning("No forecast data available")
            return
        
        # Convert to DataFrame for plotting
        df = pd.DataFrame(forecast_data)
        
        # Temperature trend
        st.subheader("Temperature Forecast")
        fig = px.line(df, 
            x='timestamp', 
            y='temperature',
            title=f"Temperature Forecast for {selected_city}",
            labels={'temperature': 'Temperature (°C)', 'timestamp': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily details in expandable sections
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        for date in df['date'].unique():
            day_data = df[df['date'] == date].iloc[0]
            with st.expander(f"Detailed Forecast for {date}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Temperature", f"{day_data['temperature']}°C", 
                            f"Feels like: {day_data['feels_like']}°C")
                with col2:
                    st.metric("Humidity", f"{day_data['humidity']}%")
                with col3:
                    st.metric("Wind Speed", f"{day_data['wind_speed']} m/s")
                
                # Additional metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precipitation", f"{day_data['probability_of_precipitation']}%")
                with col2:
                    st.metric("Pressure", f"{day_data['pressure']} hPa")
                with col3:
                    st.metric("UV Index", day_data['uv_index'])
                
                st.write(f"Condition: {day_data['condition']} {self.config.weather_icons.get(day_data['condition'], '')}")
                st.write(f"Description: {day_data['description']}")

    def render_statistics_page(self):
        st.title("📊 Weather Statistics")
        
        # City and date range selection
        col1, col2 = st.columns([2, 3])
        with col1:
            selected_city = st.selectbox("Select City", self.config.cities)
            
        with col2:
            date_range = st.date_input(
                "Select Date Range",
                value=(datetime.now() - timedelta(days=7), datetime.now()),
                max_value=datetime.now()
            )
        
        if len(date_range) != 2:
            st.warning("Please select a date range")
            return
            
        start_date, end_date = date_range
        
        # Get statistics data
        stats_data = self.api.get_statistics(
            selected_city,
            datetime.combine(start_date, datetime.min.time()),
            datetime.combine(end_date, datetime.max.time())
        )
        
        if not stats_data:
            st.warning("No statistics data available")
            return
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Temperature",
                f"{stats_data.get('avg_temperature', 0):.1f}°C",
                delta=None
            )
        
        with col2:
            st.metric(
                "Average Humidity",
                f"{stats_data.get('avg_humidity', 0):.1f}%",
                delta=None
            )
        
        with col3:
            st.metric(
                "Average Wind Speed",
                f"{stats_data.get('avg_wind_speed', 0):.1f} m/s",
                delta=None
            )
        
        # Create time series plot for temperature trends
        if 'hourly_data' in stats_data:
            st.subheader("Temperature Trends")
            hourly_df = pd.DataFrame(stats_data['hourly_data'])
            fig = px.line(
                hourly_df,
                x='timestamp',
                y='temperature',
                title=f"Temperature Trends for {selected_city}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Weather conditions breakdown
        if 'condition_counts' in stats_data:
            st.subheader("Weather Conditions Breakdown")
            conditions_df = pd.DataFrame(
                list(stats_data['condition_counts'].items()),
                columns=['Condition', 'Count']
            )
            fig = px.pie(
                conditions_df,
                values='Count',
                names='Condition',
                title=f"Weather Conditions Distribution for {selected_city}"
            )
            st.plotly_chart(fig, use_container_width=True)

    def render_alert_settings(self):
        st.subheader("🚨 Alert Configuration")
        
        selected_city = st.selectbox("Select City", self.config.cities)
        
        with st.form("alert_settings"):
            alert_type = st.selectbox(
                "Alert Type",
                ["temperature", "humidity", "wind", "severe_weather"]
            )
            
            col1, col2 = st.columns(2)
            with col1:
                min_value = st.number_input("Minimum Value", value=0.0)
            with col2:
                max_value = st.number_input("Maximum Value", value=100.0)
            
            st.subheader("Notification Settings")
            col1, col2 = st.columns(2)
            
            with col1:
                email_notifications = st.checkbox("Email Notifications")
                if email_notifications:
                    notification_email = st.text_input("Notification Email")
            
            with col2:
                consecutive_readings = st.slider(
                    "Consecutive Readings",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="Number of consecutive readings needed to trigger alert"
                )
            
            if st.form_submit_button("Save Alert Configuration"):
                if email_notifications and not notification_email:
                    st.error("Please enter a notification email address")
                elif max_value <= min_value:
                    st.error("Maximum value must be greater than minimum value")
                else:
                    alert_config = {
                        "alert_type": alert_type,
                        "min_value": min_value,
                        "max_value": max_value,
                        "consecutive_readings": consecutive_readings,
                        "email_notifications": email_notifications,
                        "notification_email": notification_email if email_notifications else None
                    }
                    
                    response = self.api.configure_alert(selected_city, alert_config)
                    if response:
                        st.success("Alert configuration saved successfully!")
                    else:
                        st.error("Failed to save alert configuration")

    def run(self):
        with st.sidebar:
            selected = option_menu(
                "Navigation",
                ["Overview", "Forecast", "Statistics", "Settings"],
                icons=['house', 'cloud-sun', 'graph-up', 'gear'],
                menu_icon="cast",
                styles={
                    "container": {"padding": "5px"},
                    "icon": {"color": DashboardTheme.PRIMARY_COLOR},
                    "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px"}
                }
            )
        
        if selected == "Overview":
            self.render_overview_page()
        elif selected == "Forecast":
            self.render_forecast_page()
        elif selected == "Statistics":
            self.render_statistics_page()
        elif selected == "Settings":
            self.render_settings_page()
            self.render_alert_settings()

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run()