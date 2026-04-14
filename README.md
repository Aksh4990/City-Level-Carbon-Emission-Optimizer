# City-Level Carbon Emission Optimizer

A Streamlit demo dashboard for forecasting and optimizing a synthetic city's carbon emissions.

The project generates monthly city data, trains two emission prediction models, runs a genetic algorithm over policy levers, and presents the results in an interactive dashboard.

## What It Does

- Generates a synthetic city emissions dataset for Jan 2020 to Dec 2024.
- Trains a Random Forest model for tabular emission prediction.
- Trains an LSTM model for 6-month emission forecasting.
- Runs a genetic algorithm to search for a lower-emission policy mix.
- Shows the results in a Streamlit dashboard with overview, forecast, optimization, comparison, and what-if simulator tabs.

## Project Structure

```text
dashboard/app.py                  Streamlit dashboard
data/carbon_emission_data.csv     Synthetic dataset
models/train_rf.py                Random Forest training script
models/train_lstm.py              LSTM training script
models/ga_optimizer.py            Genetic algorithm optimizer
models/*.pkl / *.pth / *.json     Saved model and optimizer artifacts
utils/generate_dataset.py         Synthetic dataset generator
```

## Run The Dashboard

From this project folder:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run dashboard\app.py
```

If PowerShell blocks activation, use:

```powershell
venv\Scripts\python -m streamlit run dashboard\app.py
```

Streamlit usually opens at:

```text
http://localhost:8501
```

## Rebuild The Artifacts

Run these from any working directory; the scripts resolve paths relative to the project folder.

```powershell
venv\Scripts\python utils\generate_dataset.py
venv\Scripts\python models\train_rf.py
venv\Scripts\python models\train_lstm.py
venv\Scripts\python models\ga_optimizer.py
```

## Demo Notes

This is a proof-of-concept using synthetic data, not a live city dataset. The LSTM is used for the 6-month no-action forecast, while the Random Forest is used for tabular what-if predictions and the genetic algorithm objective. The "economic viability" wording in the dashboard refers to bounded policy levers, not a separate cost model.
