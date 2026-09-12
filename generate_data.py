## I USE AI TO GENERATE THE DATASET BELOW, BUT IT IS BASED ON REALISTIC LIFEGUARD GUIDANCE AND WEATHER PATTERNS IN EGYPT.

import numpy as np
import pandas as pd

np.random.seed(42)

YEARS = [2023, 2024, 2025]
SUMMER_MONTHS = [7, 8, 9]
DAYS_PER_MONTH = 30
rows = []

for year in YEARS:
    for month in SUMMER_MONTHS:
        for day in range(1, DAYS_PER_MONTH + 1):
            week_of_summer = ((month - 7) * 30 + day - 1) // 7 + 1

            base_temp = {7: 26.5, 8: 28.0, 9: 26.0}[month]
            sea_temp = np.random.normal(base_temp, 1.2)

            wave_height = np.abs(np.random.normal(0.6, 0.4))

            wind_speed = np.abs(np.random.normal(18, 8))

            current_strength = np.clip(
                (wave_height * 3) + (wind_speed / 10) + np.random.normal(0, 1.0),
                0, 10
            )

            weather_condition = np.random.choice(
                ["Clear", "Partly Cloudy", "Windy", "Stormy"],
                p=[0.55, 0.25, 0.15, 0.05]
            )

            if weather_condition == "Stormy":
                wave_height += np.random.uniform(0.5, 1.2)
                wind_speed += np.random.uniform(10, 20)
                current_strength = min(10, current_strength + np.random.uniform(1, 3))
            elif weather_condition == "Windy":
                wind_speed += np.random.uniform(5, 12)

            uv_index = np.clip(np.random.normal(9, 1.5), 4, 12)

            danger_score = (wave_height * 1.5) + (current_strength * 1.2) + (wind_speed / 15)
            incident_lambda = max(0.02, (danger_score - 3) * 0.35)
            historical_incidents = np.random.poisson(max(0, incident_lambda))

            crowd_level = np.random.choice(["Low", "Medium", "High"], p=[0.2, 0.5, 0.3])

            unsafe = (
                (wave_height > 1.3) |
                (current_strength > 6.5) |
                (wind_speed > 35) |
                (weather_condition == "Stormy") |
                (historical_incidents >= 2)
            )

            flip = np.random.random() < 0.04
            is_safe = (not unsafe) if not flip else unsafe

            rows.append({
                "year": year,
                "month": month,
                "day": day,
                "week_of_summer": week_of_summer,
                "sea_temp_c": round(sea_temp, 1),
                "wave_height_m": round(wave_height, 2),
                "wind_speed_kmh": round(wind_speed, 1),
                "current_strength": round(current_strength, 1),
                "weather_condition": weather_condition,
                "uv_index": round(uv_index, 1),
                "historical_incidents": historical_incidents,
                "crowd_level": crowd_level,
                "is_safe": int(is_safe)
            })

df = pd.DataFrame(rows)
df.to_csv("/home/claude/safe_sahel/data/swim_safety_data.csv", index=False)

print("Dataset created!")
print("Shape:", df.shape)
print(df["is_safe"].value_counts(normalize=True).rename("proportion"))
print(df.head())
