import requests
import pandas as pd
from geopy.geocoders import Nominatim
from generate_report_assets import get_city_coordinates, get_trained_model


def predict_solar_with_ai(city_name, area_m2, efficiency=0.18, pr=0.75):
    # 1. Get automated coordinates
    lat, lon = get_city_coordinates(city_name)

    # 2. Train or Fetch the localized winning AI model for this specific city
    ai_model = get_trained_model(city_name)

    # 3. Get tomorrow's raw weather forecast features from Open-Meteo
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,relative_humidity_2m_max,cloud_cover_max,wind_speed_10m_max"
        f"&timezone=auto"
    )

    response = requests.get(url).json()

    # Index 1 corresponds to tomorrow's forecast slice
    tomorrow_date = response["daily"]["time"][1]

    # 4. Construct a pandas DataFrame matching the exact feature names used in training
    tomorrow_features = pd.DataFrame([{
        'Max_Temp': response["daily"]["temperature_2m_max"][1],
        'Humidity': response["daily"]["relative_humidity_2m_max"][1],
        'Cloud_Cover': response["daily"]["cloud_cover_max"][1],
        'Wind_Speed': response["daily"]["wind_speed_10m_max"][1]
    }])

    # 5. Use the best AI model to predict the True Irradiance (GHI)
    ai_predicted_ghi = ai_model.predict(tomorrow_features)[0]

    # 6. Objective 3 Translation Layer: Calculate final kWh generation
    predicted_kwh = area_m2 * efficiency * ai_predicted_ghi * pr

    # Print the output visualization dashboard
    print("\n" + "=" * 25)
    print(f"   AI SMART FORECAST FOR {city_name.upper()}")
    print("=" * 25)
    print(f" Target Date               : {tomorrow_date}")
    print(f" Panel Footprint Array Size : {area_m2} m²")
    print(f" AI Predicted Irradiance   : {ai_predicted_ghi:.2f} kWh/m²/day")
    print(f"⚡ Estimated Net Generation  : {predicted_kwh:.2f} Units (kWh)")
    print("=" * 25 + "\n")


if __name__ == "__main__":
    print("=== Production-Ready AI Solar Predictor ===")
    try:
        city = input("Enter place: ").strip()
        area = float(input("Enter Solar Panel Area (in m²): "))

        predict_solar_with_ai(city, area)
    except ValueError:
        print(" Error: Please enter a valid numeric value for the solar panel area.")
    except Exception as e:
        print(f" An error occurred: {e}")