# Weather Monitoring Dashboard

This is a weather monitoring dashboard application that provides real-time weather information, forecasts, and alerts for various cities. The application consists of a backend API built with FastAPI and a frontend built with Streamlit.

## Features

- **Real-time Weather Data**: The application fetches current weather conditions for selected cities, including temperature, humidity, wind speed, and more.
- **Weather Forecasts**: The application provides a multi-day weather forecast for the selected cities, including details such as temperature, probability of precipitation, and weather conditions.
- **Weather Statistics**: The application displays historical weather statistics for a selected city, including average temperature, humidity, wind speed, and weather condition distribution.
- **Weather Alerts**: Users can configure weather-based alerts (e.g., temperature, humidity, wind) for specific cities, with the option to receive email notifications when the configured thresholds are exceeded.
- **Responsive and Interactive UI**: The Streamlit-based frontend provides a user-friendly and responsive interface for interacting with the weather data and configuring alerts.

## Backend (FastAPI)

The backend of the application is built using the FastAPI framework. It provides the following API endpoints:

1. `GET /api/weather/forecast/{city}`: Retrieves the weather forecast for a specified city.
2. `GET /api/weather/current/{city}`: Retrieves the current weather conditions for a specified city.
3. `GET /api/weather/statistics/{city}`: Retrieves historical weather statistics for a specified city.
4. `POST /api/alerts/configure/{city}`: Configures a weather alert for a specified city.
5. `GET /api/alerts/{city}`: Retrieves the configured weather alerts for a specified city.
6. `DELETE /api/alerts/{city}/{alert_type}`: Deletes a configured weather alert for a specified city and alert type.

To run the backend:

1. Ensure you have Python 3.7+ installed.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Set the necessary environment variables (e.g., API keys, database connection details) in a `.env` file.
4. Run the FastAPI application: `python main.py`

The backend will start running on `http://localhost:8000`.

## Frontend (Streamlit)

The frontend of the application is built using the Streamlit framework. It provides a user-friendly interface for interacting with the weather data and configuring alerts.

To run the frontend:

1. Ensure you have Python 3.7+ installed.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Streamlit application: `streamlit run dashboard.py`

The frontend will start running on `http://localhost:8501`.

## Running the Application Together

To run the entire application (backend and frontend), follow these steps:

1. Start the backend server by running `python main.py` in one terminal or command prompt.
2. Start the frontend server by running `streamlit run dashboard.py` in another terminal or command prompt.
3. Open your web browser and navigate to `http://localhost:8501` to access the Streamlit-based frontend.

The frontend will interact with the backend API to fetch and display the weather data and manage the weather alerts.



## Running the Application with Docker
To run the entire application (backend and frontend) using Docker, follow these steps:

# Build the Docker image:
 `docker build -t weather-app .`
This will build a Docker image named weather-app using the Dockerfile in the project directory.
# Run the Docker container:
`docker run -p 8000:8000 -p 8501:8501 weather-app`

This will start the Docker container and map the backend and frontend ports to the host system.

The backend will be accessible at http://localhost:8000.
The frontend will be accessible at http://localhost:8501.

## Configuration and Environment Variables

The application uses various configuration settings, which are managed through environment variables. These include:

- **API Keys**: `OPENWEATHER_API_KEY`, `API_KEY`, `GEODB_API_KEY`
- **Database Settings**: `MONGODB_URL`, `MONGODB_DB_NAME`
- **Email Settings**: `EMAIL_HOST`, `EMAIL_PORT`, `EMAIL_USER`, `EMAIL_PASSWORD`
- **Application Settings**: `UPDATE_INTERVAL`, `ALERT_CHECK_INTERVAL`, `TEMPERATURE_UNIT`, `ANOMALY_DETECTION_THRESHOLD`, `MAX_RETRIES`, `RETRY_DELAY`
- **Caching Settings**: `GEOCODING_CACHE_TTL`, `GEOCODING_CACHE_SIZE`

Make sure to set these environment variables correctly before running the application.

## Dependencies

The backend (FastAPI) and frontend (Streamlit) applications share some common dependencies, which are listed in the `requirements.txt` file. The main dependencies include:

- **FastAPI**: for building the backend API
- **Streamlit**: for building the frontend dashboard
- **motor**: for interacting with the MongoDB database
- **httpx**: for making HTTP requests to the weather API
- **pytz**: for handling timezone-related operations
- **cachetools**: for caching geocoding results
- **apscheduler**: for scheduling background tasks

## Future Improvements

- **Improved Error Handling and Logging**: Enhance the error handling and logging mechanisms to provide more detailed and user-friendly error messages.
- **User Authentication and Authorization**: Implement user authentication and authorization to allow multiple users to manage their own alerts and preferences.
- **Advanced Visualization**: Explore more advanced visualization techniques, such as interactive charts and graphs, to provide a more comprehensive and intuitive data presentation.
- **Notification Channels**: Expand the notification capabilities to include additional channels, such as SMS, push notifications, or integrations with popular communication platforms.
- **Anomaly Detection and Insights**: Implement more advanced data analysis and anomaly detection algorithms to provide users with meaningful insights and recommendations.
