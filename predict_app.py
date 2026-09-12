import joblib
import pandas as pd

MODEL_PATH = "model/safe_sahel_model.pkl"


def ask_float(prompt, min_val, max_val):
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"  Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("  That's not a number, try again.")


def ask_int(prompt, min_val, max_val):
    while True:
        try:
            value = int(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"  Please enter a whole number between {min_val} and {max_val}.")
        except ValueError:
            print("  That's not a whole number, try again.")


def ask_choice(prompt, options):
    options_str = "/".join(options)
    while True:
        value = input(f"{prompt} ({options_str}): ").strip().title()
        if value in options:
            return value
        print(f"  Please type one of: {options_str}")


def main():
    print("=" * 55)
    print("   SAFE SAHEL - Mediterranean Swim Safety Predictor")
    print("        Egypt North Coast | Summer Season")
    print("=" * 55)
    print("\nEnter this week's sea & weather conditions:\n")

    sea_temp = ask_float("Sea temperature in Celsius (e.g. 27.5): ", 15, 40)
    wave_height = ask_float("Wave height in meters (e.g. 0.6): ", 0, 6)
    wind_speed = ask_float("Wind speed in km/h (e.g. 18): ", 0, 120)
    current_strength = ask_float("Current strength, scale 0 (none) - 10 (extreme): ", 0, 10)
    weather_condition = ask_choice("Weather condition", ["Clear", "Partly Cloudy", "Windy", "Stormy"])
    uv_index = ask_float("UV index (e.g. 9): ", 0, 14)
    historical_incidents = ask_int("Reported incidents this week (0, 1, 2...): ", 0, 20)
    crowd_level = ask_choice("Beach crowd level", ["Low", "Medium", "High"])

    sea_roughness_score = wave_height * 2 + wind_speed / 10 + current_strength
    recent_incident_flag = 1 if historical_incidents > 0 else 0

    input_row = pd.DataFrame([{
        "sea_temp_c": sea_temp,
        "wave_height_m": wave_height,
        "wind_speed_kmh": wind_speed,
        "current_strength": current_strength,
        "uv_index": uv_index,
        "historical_incidents": historical_incidents,
        "sea_roughness_score": sea_roughness_score,
        "recent_incident_flag": recent_incident_flag,
        "weather_condition": weather_condition,
        "crowd_level": crowd_level
    }])

    model = joblib.load(MODEL_PATH)
    probability_safe = model.predict_proba(input_row)[0][1]
    prediction = "SAFE" if probability_safe >= 0.5 else "NOT SAFE"

    print("\n" + "=" * 55)
    print(f"  PREDICTION:        {prediction} to swim this week")
    print(f"  SAFETY PERCENTAGE: {probability_safe:.1%}")
    print("=" * 55)

    if prediction == "NOT SAFE":
        print("\n⚠️  Stay cautious - consider avoiding deep water, watch for")
        print("   lifeguard flags, and don't swim alone this week.")
    else:
        print("\n✅ Conditions look good, but always check local lifeguard")
        print("   flags on the day before swimming.")


if __name__ == "__main__":
    main()
