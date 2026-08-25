import os
import matplotlib.pyplot as plt
import numpy as np
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


def plot_model_comparison(city_name, xgb_rmse, xgb_mape, rf_rmse, rf_mape):
    """Generates Figure 6.3: Model Comparison Bar Charts for RMSE and MAPE."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    models = ["XGBoost", "Random Forest"]
    rmse_values = [xgb_rmse, rf_rmse]
    mape_values = [xgb_mape, rf_mape]
    colors = ["#f39c12", "#2980b9"]

    # --- Plot RMSE Subplot ---
    axes[0].bar(models, rmse_values, color=colors, width=0.5)
    axes[0].set_ylabel("RMSE (kWh/m²/day)")
    axes[0].set_title(f"RMSE (kWh/m²/day) — {city_name}")
    axes[0].set_ylim(0, max(rmse_values) * 1.25)
    for i, v in enumerate(rmse_values):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

    # --- Plot MAPE Subplot ---
    axes[1].bar(models, mape_values, color=colors, width=0.5)
    axes[1].set_ylabel("MAPE (%)")
    axes[1].set_title(f"MAPE (%) — {city_name}")
    axes[1].set_ylim(0, max(mape_values) * 1.25)
    for i, v in enumerate(mape_values):
        axes[1].text(i, v + 0.2, f"{v:.2f}%", ha="center", fontweight="bold")

    plt.tight_layout()
    output_filename = f"figure_6_3_comparison_{city_name.lower()}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"📊 Saved comparison plot to: {output_filename}")
    plt.close()


def plot_residual_analysis(city_name, winner_name, y_test, predictions):
    """Generates Figure 6.2: Predicted vs Actual and Residual Plots for Winning Model."""
    residuals = predictions - y_test.values

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Predicted vs Actual Plot ---
    axes[0].scatter(y_test.values, predictions, alpha=0.5, color="#2980b9", edgecolors="none")
    max_val = max(max(y_test.values), max(predictions)) + 0.5
    axes[0].plot([0, max_val], [0, max_val], "r--", label="Ideal Fit (45°)")
    axes[0].set_xlabel("Actual GHI (kWh/m²/day)")
    axes[0].set_ylabel("Predicted GHI (kWh/m²/day)")
    axes[0].set_title(f"Predicted vs. Actual ({winner_name})")
    axes[0].legend()

    # --- Residual Plot ---
    axes[1].scatter(predictions, residuals, alpha=0.5, color="#2980b9", edgecolors="none")
    axes[1].axhline(0, color="r", linestyle="--")
    axes[1].set_xlabel("Predicted GHI (kWh/m²/day)")
    axes[1].set_ylabel("Residual (Predicted - Actual)")
    axes[1].set_title(f"Residual Plot ({city_name})")

    plt.tight_layout()
    output_filename = f"figure_6_2_residuals_{city_name.lower()}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"📈 Saved residual analysis plot to: {output_filename}")
    plt.close()


def get_trained_model(city_name):
    """Automates data checking, downloading, cleaning, model training, and visual generation."""
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
    df = df.replace([-999.0, -999], pd.NA)
    df = df.dropna(subset=["GHI_Target", "Max_Temp", "Humidity", "Cloud_Cover", "Wind_Speed"])

    # C. Domain-Specific Filtering (Physics Sanity Check)
    df = df[(df["Humidity"] >= 0) & (df["Humidity"] <= 100)]
    df = df[(df["Cloud_Cover"] >= 0) & (df["Cloud_Cover"] <= 100)]
    df = df[df["GHI_Target"] >= 0]

    cleaned_rows = len(df)
    print(f"Cleaned {initial_rows - cleaned_rows} anomalous rows from dataset.")

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
    print(f"\n Training XGBoost Regressor for {city_name}...")
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_rmse = root_mean_squared_error(y_test, xgb_preds)
    xgb_mape = mean_absolute_percentage_error(y_test, xgb_preds) * 100

    # 6. Train and evaluate Random Forest
    print(f" Training Random Forest Regressor for {city_name}...")
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_rmse = root_mean_squared_error(y_test, rf_preds)
    rf_mape = mean_absolute_percentage_error(y_test, rf_preds) * 100

    # 7. Print Comparative Showdown Dashboard
    print("\n" + "=" * 50)
    print(f" MODEL COMPARISON SHOWDOWN FOR {city_name.upper()}")
    print("=" * 50)
    print(f" XGBoost       -> RMSE: {xgb_rmse:.3f} kWh/m²/day | MAPE: {xgb_mape:.2f}%")
    print(f" Random Forest -> RMSE: {rf_rmse:.3f} kWh/m²/day | MAPE: {rf_mape:.2f}%")
    print("=" * 50)

    # 8. Dynamically pick the winner and set up plot values
    if xgb_mape < rf_mape:
        print(" Winner: XGBoost Regressor selected for deployment!")
        best_model = xgb_model
        best_preds = xgb_preds
        winner_name = "XGBoost Regressor"
    else:
        print(" Winner: Random Forest Regressor selected for deployment!")
        best_model = rf_model
        best_preds = rf_preds
        winner_name = "Random Forest Regressor"
    print("=" * 50 + "\n")

    # 9. GENERATE AND SAVE REPORT PLOTS AUTOMATICALLY
    plot_model_comparison(city_name, xgb_rmse, xgb_mape, rf_rmse, rf_mape)
    plot_residual_analysis(city_name, winner_name, y_test, best_preds)

    return best_model


if __name__ == "__main__":
    test_city = input(
        "Enter location to test automated comparative training: "
    ).strip()
    trained_brain = get_trained_model(test_city)