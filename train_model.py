import os
import pandas as pd
from geopy.geocoders import Nominatim
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Import the download function we wrote earlier
from download_data import download_nasa_historical_data


def get_city_coordinates(city_name):
    """Automates city-to-coordinate lookup."""
    geolocator = Nominatim(user_agent="india_solar_automation")
    location = geolocator.geocode(f"{city_name}, India")
    if not location:
        raise ValueError(f"Could not find coordinates for {city_name}")
    # Round coordinates to 4 decimal places to match NASA format
    return round(location.latitude, 4), round(location.longitude, 4)


def get_trained_model(city_name):
    """Automates data checking, downloading, cleaning, and model training."""
    lat, lon = get_city_coordinates(city_name)
    csv_filename = f"historical_solar_data_{lat}_{lon}.csv"

    if not os.path.exists(csv_filename):
        print(f"🔍 Local data for {city_name} not found. Automating download...")
        download_nasa_historical_data(lat, lon, start_year=2021, end_year=2025)

    # 3. Load raw data
    df = pd.read_csv(csv_filename)

    print(f"🧹 Executing Data Cleaning Pipeline for {city_name}...")
    initial_rows = len(df)

    # A. Drop complete duplicates if they exist
    df = df.drop_duplicates()

    # B. Handle NASA's specific missing data flag (-999 or -999.0)
    # Replace them with NaN so pandas can track them, then drop or fill them
    df = df.replace([-999.0, -999], pd.NA)
    df = df.dropna(subset=["GHI_Target", "Max_Temp", "Humidity", "Cloud_Cover", "Wind_Speed"])

    # C. Domain-Specific Filtering (Physics Sanity Check)
    # Relative humidity cannot drop below 0% or exceed 100%
    df = df[(df["Humidity"] >= 0) & (df["Humidity"] <= 100)]
    # Cloud cover must mathematically sit between 0% and 100%
    df = df[(df["Cloud_Cover"] >= 0) & (df["Cloud_Cover"] <= 100)]
    # Target GHI cannot be negative (nighttime or zero radiation is 0)
    df = df[df["GHI_Target"] >= 0]

    cleaned_rows = len(df)
    print(f"✨ Cleaned {initial_rows - cleaned_rows} anomalous rows from dataset.")

    # Define Features (X) and Target (y)
    X = df[["Max_Temp", "Humidity", "Cloud_Cover", "Wind_Speed"]]
    y = df["GHI_Target"]

    # Split into Training and Testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # 5. Train and evaluate XGBoost
    print(f"\n🤖 Training XGBoost Regressor for {city_name}...")
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_rmse = root_mean_squared_error(y_test, xgb_preds)
    xgb_mape = mean_absolute_percentage_error(y_test, xgb_preds) * 100

    # 6. Train and evaluate Random Forest
    print(f"🌲 Training Random Forest Regressor for {city_name}...")
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_rmse = root_mean_squared_error(y_test, rf_preds)
    rf_mape = mean_absolute_percentage_error(y_test, rf_preds) * 100

    # 7. Print Comparative Showdown Dashboard
    print("\n" + "=" * 50)
    print(f"📊 MODEL COMPARISON SHOWDOWN FOR {city_name.upper()}")
    print("=" * 50)
    print(f"🚀 XGBoost       -> RMSE: {xgb_rmse:.3f} kWh/m²/day | MAPE: {xgb_mape:.2f}%")
    print(f"🌲 Random Forest -> RMSE: {rf_rmse:.3f} kWh/m²/day | MAPE: {rf_mape:.2f}%")
    print("=" * 50)

    # 8. Dynamically pick the winner based on lowest error rate (MAPE)
    if xgb_mape < rf_mape:
        print("🏆 Winner: XGBoost Regressor selected for deployment!")
        best_model = xgb_model
    else:
        print("🏆 Winner: Random Forest Regressor selected for deployment!")
        best_model = rf_model
    print("=" * 50 + "\n")

    return best_model

if __name__ == "__main__":
    test_city = input(
        "Enter location to test automated comparative training: "
    ).strip()
    trained_brain = get_trained_model(test_city)