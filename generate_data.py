## I USE AI TO GENERATE THE DATASET BELOW, BUT IT IS BASED ON REALISTIC LIFEGUARD GUIDANCE AND WEATHER PATTERNS IN EGYPT.

import numpy as np
import pandas as pd

np.random.seed(42) 
YEARS = [2023, 2024, 2025]
SUMMER_MONTHS = [7, 8, 9]          # July, August, September
DAYS_PER_MONTH = 30                 
rows = []

for year in YEARS:
    for month in SUMMER_MONTHS:
        for day in range(1, DAYS_PER_MONTH + 1):
            week_of_summer = ((month - 7) * 30 + day - 1) // 7 + 1  # week 1..13

            # ---- Simulate weather/sea features using realistic Egypt ranges ----

            # Sea temperature (Celsius): warms up July -> peaks August -> cools Sept
            base_temp = {7: 26.5, 8: 28.0, 9: 26.0}[month]
            sea_temp = np.random.normal(base_temp, 1.2)

            # Wave height (meters): usually calm in summer, occasional swell
            wave_height = np.abs(np.random.normal(0.6, 0.4))

            # Wind speed (km/h): Etesian "Meltemi-like" winds can pick up in Aug
            wind_speed = np.abs(np.random.normal(18, 8))

            # Current strength (arbitrary scale 0-10, 0=none, 10=extreme rip current)
            # Correlated loosely with wind and wave height (rough seas = stronger currents)
            current_strength = np.clip(
                (wave_height * 3) + (wind_speed / 10) + np.random.normal(0, 1.0),
                0, 10
            )

            # Weather condition (categorical)
            weather_condition = np.random.choice(
                ["Clear", "Partly Cloudy", "Windy", "Stormy"],
                p=[0.55, 0.25, 0.15, 0.05]
            )
            # Stormy weather pushes wave height & wind up further
            if weather_condition == "Stormy":
                wave_height += np.random.uniform(0.5, 1.2)
                wind_speed += np.random.uniform(10, 20)
                current_strength = min(10, current_strength + np.random.uniform(1, 3))
            elif weather_condition == "Windy":
                wind_speed += np.random.uniform(5, 12)

            # UV index (0-11+), just informative, not safety-critical
            uv_index = np.clip(np.random.normal(9, 1.5), 4, 12)

            # Historical incidents in that WEEK (count of drowning/rescue reports)
            # Higher when current/wave/wind are high -> makes the label realistic
            danger_score = (wave_height * 1.5) + (current_strength * 1.2) + (wind_speed / 15)
            incident_lambda = max(0.02, (danger_score - 3) * 0.35)
            historical_incidents = np.random.poisson(max(0, incident_lambda))

            # Beach crowd level (more people = more reported near-misses, informative feature)
            crowd_level = np.random.choice(["Low", "Medium", "High"], p=[0.2, 0.5, 0.3])

            # ---- LABEL: is it safe to swim? (rule based on real lifeguard guidance) ----
            # Danger if: big waves, strong current, high wind, stormy weather,
            # or recent incidents reported.
            unsafe = (
                (wave_height > 1.3) |
                (current_strength > 6.5) |
                (wind_speed > 35) |
                (weather_condition == "Stormy") |
                (historical_incidents >= 2)
            )

            # Add a little randomness so the model doesn't just learn a perfect
            # if/else rule (real life always has some noise/uncertainty)
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
                "is_safe": int(is_safe)   # 1 = Safe, 0 = Not Safe (our label)
            })

df = pd.DataFrame(rows)
df.to_csv("/home/claude/safe_sahel/data/swim_safety_data.csv", index=False)

print("Dataset created!")
print("Shape:", df.shape)
print(df["is_safe"].value_counts(normalize=True).rename("proportion"))
print(df.head())
