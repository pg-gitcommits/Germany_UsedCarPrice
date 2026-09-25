# Germany Used Car Price Prediction

A machine learning app that estimates the resale price (in €) of a used car in Germany from its brand, model, age, mileage, fuel type, and a few other listing attributes.

The project covers the full workflow: data cleaning and feature engineering in a Jupyter notebook, a K-Nearest Neighbours regression model, and two ways to serve it: a **Streamlit** web UI and a **Flask + Swagger** REST API. It was built in June 2021.

- **Dataset:** [Used Cars Data (data.world / data-society)](https://data.world/data-society/used-cars-data), about 370k listings scraped from eBay Kleinanzeigen in March–April 2016
- **Deployed app (Heroku, 2021):** http://ucpp2.herokuapp.com/ (no longer live)

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
4. [Model and Results](#model-and-results)
5. [Running the Project](#running-the-project)
6. [API Reference](#api-reference)
7. [Deployment](#deployment)
8. [Future Work](#future-work)
9. [Author](#author)

---

## Repository Structure

| File | Purpose |
|---|---|
| `GUCPP.ipynb` | Main notebook: EDA, cleaning, feature engineering, training, and exporting the model and lookup files |
| `autos 2.csv` | Raw dataset (~371k rows, 20 columns, ~67 MB) |
| `classifier.pkl` | Trained `KNeighborsRegressor` saved with pickle (~72 MB) |
| `my_streamlit.py` | Streamlit web app: dropdown inputs → price prediction |
| `swagger.py` | Flask REST API with a Swagger UI (via `flasgger`) exposing `POST /predict` |
| `app.py` | Early Flask prototype of the prediction endpoint |
| `columns.json` | Ordered list of the 45 model input features |
| `brand.json` | Brand name → integer label mapping (38 brands) |
| `model.json` | Car model name → integer label mapping (247 models) |
| `gpdbandm.csv` | Valid brand/model combinations found in the cleaned data (for reference only; the app doesn't use it) |
| `requirements.txt` | Python dependencies |
| `Procfile`, `setup.sh` | Heroku deployment config for the Streamlit app |

---

## Dataset

The raw data has **371,528 listings** and 20 columns. Most values are in German:

| Column | Meaning |
|---|---|
| `price` | Asking price in € (**target**) |
| `brand`, `model` | Manufacturer and model (e.g. `volkswagen`, `golf`) |
| `vehicletype` | `limousine` (sedan), `kleinwagen` (small car/hatchback), `kombi` (station wagon), `bus`, `cabrio`, `coupe`, `suv`, `andere` (other) |
| `yearofregistration`, `monthofregistration` | First registration date |
| `gearbox` | `manuell` / `automatik` |
| `powerps` | Engine power in PS (metric horsepower) |
| `kilometer` | Odometer reading, already bucketed (5,000 – 150,000) |
| `fueltype` | `benzin` (petrol), `diesel`, `lpg`, `cng`, `hybrid`, `elektro`, `andere` |
| `notrepaireddamage` | `ja` / `nein`: whether the car has unrepaired damage |
| `offertype` | `Angebot` (offer) / `Gesuch` (request) |
| `seller`, `abtest`, `name`, `postalcode`, `nrofpictures`, `datecrawled`, `datecreated`, `lastseen` | Listing metadata (dropped) |

---

## Methodology

All steps are in `GUCPP.ipynb`.

### 1. Cleaning

| Step | Rows remaining |
|---|---|
| Raw data | 371,528 |
| Drop extreme prices (`price >= 500,000`) | 371,426 |
| Map `notrepaireddamage` `ja/nein` → `1/0`, fill missing values with the mode | 371,426 |
| Drop metadata columns (`seller`, `name`, `postalcode`, `abtest`, dates, `nrofpictures`) | – |
| Drop rows with any remaining nulls (`vehicletype`, `gearbox`, `model`, `fueltype`) | 299,827 |
| Keep `yearofregistration > 1980` (raw data had values from 1000 to 9999) | 297,229 |
| Keep `50 < powerps < 350` (raw data had values from 0 to 20,000) | 271,530 |
| Drop `monthofregistration == 0` (unknown month) | 261,367 |
| Drop `model == "andere"` ("other", which doesn't identify a model) | **242,901** |

`seller` was dropped because 99.999% of listings are private sellers, so it carries almost no signal.

### 2. Feature Engineering

- **Power binning:** `powerps` → `powerps_bin` using the bins `(50,100] → 1`, `(100,200] → 2`, `(200,250] → 3`, `(250,350] → 4`. The raw `powerps` column is then dropped.
- **One-hot encoding** (`pd.get_dummies`, `drop_first=True`) for `offertype`, `vehicletype`, `gearbox`, `kilometer`, `monthofregistration`, `fueltype`, `notrepaireddamage`, `powerps_bin`.
- **Label encoding** for the high-cardinality `brand` (38 values) and `model` (247 values), sorted alphabetically. The mappings are saved to `brand.json` and `model.json` so the apps can convert names to the same integers.
- `yearofregistration` is kept as a raw integer.

This gives **45 features** (listed in order in `columns.json`).

### 3. Train/Test Split

85/15 split with `train_test_split(test_size=0.15, random_state=1)`:
- Train: 206,465 rows
- Test: 36,436 rows

Feature scaling was tried and then skipped (the code is still in the notebook, commented out).

---

## Model and Results

The final model is a **K-Nearest Neighbours regressor with `k = 100`** (`sklearn.neighbors.KNeighborsRegressor`).

| Metric | Test set |
|---|---|
| RMSE | **≈ €3,001** |

For context, after removing nulls (before the final filters), the median price was about €3,500 and the standard deviation was about €8,950.

Random Forest, XGBoost, and KNN with `k = 9, 25, 50` were also explored during development; that code is kept in the notebook for reference.

The trained model is saved to `classifier.pkl`.

---

## Running the Project

### Prerequisites

- Python **3.8** (the notebook and cached bytecode were built with 3.8)
- scikit-learn **0.24.x** (the version `classifier.pkl` was trained with)

### Setup

```bash
git clone https://github.com/pg-gitcommits/Germany_UsedCarPrice.git
cd Germany_UsedCarPrice

python3.8 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install scikit-learn==0.24.2 flask flasgger
```

### Option A: Streamlit UI

```bash
streamlit run my_streamlit.py
```

This opens a form in your browser with dropdowns for brand, model, year, offer type, vehicle type, gearbox, mileage, registration month, fuel type, pending repairs, and power. Click **Predict** to see the estimated price.

### Option B: Flask + Swagger API

```bash
python swagger.py
```

Then open **http://127.0.0.1:5000/apidocs** to try the `/predict` endpoint in the Swagger UI.

### Re-training the Model

Open `GUCPP.ipynb` and run all cells. The notebook reads `autos 2.csv` and writes out `classifier.pkl`, `columns.json`, `brand.json`, `model.json`, and `gpdbandm.csv`.

---

## API Reference

### `POST /predict`

All parameters are sent as **query parameters**.

| Parameter | Type | Accepted values |
|---|---|---|
| `brand` | int | Brand label from `brand.json` (e.g. `36` = volkswagen) |
| `model` | int | Model label from `model.json` (e.g. `116` = golf) |
| `yearofregistration` | int | 1981 – 2018 |
| `offertype` | string | `Offer`, `Request` |
| `vehicletype` | string | `Bus`, `Convertible`, `Coupe`, `Hatchback`, `Station Wagon`, `limousine`, `SUV`, anything else = other |
| `gearbox` | string | `Manual`, `Automatic` |
| `kilometer` | string | `<10,000`, `10,000-20,000`, … , `100,000-125,000`, `125,000-150,000` |
| `monthofregistration` | string | `January` … `December` |
| `fueltype` | string | `Petrol`, `CNG`, `Diesel`, `Electric`, `Hybrid`, `LPG`, anything else = other |
| `notrepaireddamage` | string | `Yes`, `No` |
| `powerps` | string | `50-100`, `100-200`, `200-300`, `300-350` |

**Example**

```bash
curl -X POST "http://127.0.0.1:5000/predict?brand=36&model=116&yearofregistration=2010&offertype=Offer&vehicletype=Hatchback&gearbox=Manual&kilometer=100,000-125,000&monthofregistration=June&fueltype=Diesel&notrepaireddamage=No&powerps=100-200"
```

**Response** (plain text)

```
The Predicted Value is €<price>
```

---

## Deployment

The Streamlit app was deployed on **Heroku** with this config:

- `Procfile`: `web: sh setup.sh && streamlit run my_streamlit.py`
- `setup.sh` writes `~/.streamlit/config.toml` so Streamlit runs headless on Heroku's assigned `$PORT`.

---

## Future Work

- **Model improvements:** use cross-validation and hyperparameter tuning, and benchmark gradient-boosting models (XGBoost, LightGBM, CatBoost) against KNN using RMSE, MAE, and R².
- **Better categorical encoding:** replace label encoding for `brand` and `model` with target encoding or learned embeddings.
- **Richer features:** keep `powerps` as a continuous value, add vehicle age at listing time, and use the free-text `name` field (for example, trim levels like "TDI" or "AMG").
- **Smarter UI:** filter the model dropdown by the selected brand, using the valid pairs in `gpdbandm.csv`.
- **Fresher data:** retrain on recent listings to reflect today's used-car market.
- **Modern deployment:** combine the Streamlit and API apps, containerise them with Docker, and redeploy on a current platform (e.g. Streamlit Community Cloud or Render).

---

## Author

**Pranav Gowtham Bulusu**, June 2021
