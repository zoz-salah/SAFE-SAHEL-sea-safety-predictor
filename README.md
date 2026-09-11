# Safe Sahel 🌊
### Mediterranean Sea Swimming Safety Predictor — Egypt (Summer Season)

A beginner-friendly, end-to-end machine learning project that predicts whether
it's safe to swim on Egypt's North Coast during **July, August, and September**.

---

## What's inside

| File | Purpose |
|---|---|
| `data/generate_data.py` | Step 1 — creates the dataset from scratch (no real dataset existed, so we simulate 3 realistic summers using known Mediterranean/Egypt weather ranges) |
| `eda.py` | Step 2 — Exploratory Data Analysis: stats, missing-value checks, boxplots, correlation heatmap |
| `train_model.py` | Steps 3–6 — preprocessing, feature engineering, model training (Logistic Regression), and evaluation |
| `predict_app.py` | Step 7 — a simple command-line front-end: answer a few questions, get a prediction |
| `safe_sahel_app.html` | A browser-based interactive version of the same predictor (drag sliders, see the result update live) — open it directly in any browser, no install needed |
| `model/safe_sahel_model.pkl` | The trained model, saved and ready to reuse |
| `model/model_export.json` | The trained model's weights, exported so the HTML version can run the exact same math in the browser |
| `data/swim_safety_data.csv` | The generated dataset (270 daily rows across 3 summers) |
| `data/eda_charts.png`, `data/correlation_heatmap.png` | Charts from the EDA step |

---

## How to run it (Python side)

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib

python3 data/generate_data.py   # 1. build the dataset
python3 eda.py                  # 2. explore it (saves charts)
python3 train_model.py          # 3. train + evaluate + save the model
python3 predict_app.py          # 4. try it! answer the questions in your terminal
```

## Or just open the app in a browser

Double-click **`safe_sahel_app.html`** — no installation needed. It uses the
exact same trained model weights, running as plain JavaScript.

---

## How the "safety" label was decided

Since no public dataset exists for this exact problem, the dataset is
**simulated** using realistic value ranges for Egypt's North Coast in summer,
and each day is labeled Safe/Not Safe using a rule based on real lifeguard
guidance:

> A day is marked **Not Safe** if wave height > 1.3m, OR current strength >
> 6.5/10, OR wind speed > 35 km/h, OR the weather is Stormy, OR 2+ incidents
> were reported that week.

A small amount of random noise (4%) is added so the model has to actually
*learn* the pattern rather than just memorizing a perfect if/else rule —
just like real-world data.

## The model

**Logistic Regression** — chosen deliberately because it's:
- Simple and easy to explain to a beginner
- Naturally outputs a probability (0–100%), which becomes our "safety percentage"
- Fast to train, no GPU or deep learning needed

On the held-out test set it reaches about **81% accuracy** (varies slightly
by random seed). The features that mattered most were **recent incidents**,
**wind speed**, and **stormy weather** — which matches real-world intuition.

## Ideas to extend this project
- Swap in real historical weather data (e.g. from a weather API) instead of simulated data
- Try a Decision Tree or Random Forest and compare accuracy
- Add real beach-specific data (Marina, Sidi Abdel Rahman, Montaza, etc.)
- Turn the HTML app into a full Streamlit or Flask web app with a backend
