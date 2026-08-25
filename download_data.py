import pandas as pd
import requests


def download_nasa_historical_data(lat, lon, start_year=2021, end_year=2024):
    """Downloads historical daily weather and solar data from NASA POWER API."""
    print(f"🛰️ Requesting historical data from NASA for Lat: {lat}, Lon: {lon}...")

    # Define the parameters we need for training our Solar ML model
    # ALLSKY_SFC_SW_DWN = Global Horizontal Irradiance (GHI) - Our Target (y)
    # T2M_MAX = Max Temperature
    # RH2M = Relative Humidity
    # CLOUD_AMT = Cloud Amount/Coverage %
    # WS10M = Wind Speed at 10 meters
    parameters = "ALLSKY_SFC_SW_DWN,T2M_MAX,RH2M,CLOUD_AMT,WS10M"

    start_date = f"{start_year}0101"
    end_date = f"{end_year}1231"

    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters={parameters}&community=RE&longitude={lon}&latitude={lat}"
        f"&start={start_date}&end={end_date}&format=JSON"
    )

    response = requests.get(url)

    if response.status_code != 200:
        print("❌ Error fetching data from NASA API.")
        return None

    data = response.json()

    # Extract the time-series features
    records = []
    dates = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"].keys()

    for date in dates:
        records.append(
            {
                "Date": pd.to_datetime(date, format="%Y%m%d"),
                "GHI_Target": data["properties"]["parameter"][
                    "ALLSKY_SFC_SW_DWN"
                ][date],
                "Max_Temp": data["properties"]["parameter"]["T2M_MAX"][date],
                "Humidity": data["properties"]["parameter"]["RH2M"][date],
                "Cloud_Cover": data["properties"]["parameter"]["CLOUD_AMT"][
                    date
                ],
                "Wind_Speed": data["properties"]["parameter"]["WS10M"][date],
            }
        )

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Clean data: NASA uses -999 for missing values. Let's drop or handle them if any.
    df = df.replace(-999, pd.NA).dropna()

    filename = f"historical_solar_data_{lat}_{lon}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Success! Saved {len(df)} days of historical data to '{filename}'")

    return filename
