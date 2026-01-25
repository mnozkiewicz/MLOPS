import os
import requests
from typing import Annotated, Optional, Union, Tuple
from fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("Weather Service")

API_KEY = os.environ.get("OPEN_METEO_API")

def get_coordinates(city: str, country: Optional[str] = None) -> Union[Tuple[float, float], str]:

    url = "http://api.openweathermap.org/geo/1.0/direct"
    query = f"{city},{country}" if country else city
    params = {
        "q": query,
        "limit": 1,
        "appid": API_KEY
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return f"Response: {response.text}"

        data = response.json()
        if not data:
            return f"Error: City '{query}' not found."
        return data[0]['lat'], data[0]['lon']

    except Exception as e:
        return f"Geocoding error: {e}"


@mcp.tool(description="Get daily weather forecast for a city, up to 16 days into the future.")
def get_daily_forecast(
    city: Annotated[str, "The city name to get the weather for."],
    country: Annotated[str, "The country the city is in (optional)"] = "",
    days: Annotated[int, "Number of days to forecast (1-16). Defaults to 1."] = 1
) -> str:

    coords = get_coordinates(city, country)
    if isinstance(coords, str):
        return coords
    
    lat, lon = coords
    url = "https://pro.openweathermap.org/data/2.5/forecast/daily"
    params = {
        "lat": 40,
        "lon": 40,
        "cnt": 1,
        "units": "metric",
        "appid": "d582cf96baa0aaf964b01af76b446eb1"
    }
    
    try:
        res = requests.get(url, params=params, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        report = []
        for day in data.get('list', []):
            date_str = datetime.fromtimestamp(day['dt']).strftime('%Y-%m-%d')
            desc = day['weather'][0]['description']
            temp = day['temp']['day']
            report.append(f"{date_str}: {desc}, {temp}°C")
            
        return "\n".join(report)

    except Exception as e:
        return f"Error fetching forecast: {str(e)}"


@mcp.tool(description="Get historical monthly weather averages. Use this for planning trips far in the future.")
def get_monthly_average(
    city: Annotated[str, "The city name."],
    month: Annotated[int, "The month number (1-12)."],
    country: Annotated[str, "The country (optional)."] = ""
) -> str:
    
    coords = get_coordinates(city, country)
    if not coords:
        return f"Error: Could not find coordinates for city: {city}"
    
    lat, lon = coords

    url = "http://history.openweathermap.org/data/2.5/aggregated/month"
    params = {
        "lat": lat,
        "lon": lon,
        "month": month,
        "appid": API_KEY
    }
    
    try:
        res = requests.get(url, params=params, timeout=10)

        res.raise_for_status()
        data = res.json()
        result = data.get('result', {})
        
        temp_mean = result.get('temp', {}).get('mean', 'N/A')
        precip_mean = result.get('precipitation', {}).get('mean', 'N/A')
        
        return (f"Averages for Month {month} in {city}:\n"
                f"Mean Temp: {temp_mean} K\n"
                f"Mean Precipitation: {precip_mean} mm")

    except Exception as e:
        return f"Error fetching averages: {str(e)}"

@mcp.tool(description="Returns the current date in 'YYYY-MM-DD' format.")
def get_current_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=8001, host="0.0.0.0")