# Deployment & Remaining Submission Steps

Everything that could be computed and validated in this sandbox is done — the notebook,
model, submission file, and Flask app are all real, executed artifacts. A few steps in the
project brief require accounts/hosting/tools this sandbox doesn't have (no internet
egress, no TensorFlow/MLflow installed, no ability to hold your GitHub/Heroku
credentials). This file gives you the exact commands to finish those on your own machine.

## 1. Push to GitHub

```bash
cd rossmann-sales-forecast   # put all the files from this delivery in one folder
git init
git add .
git commit -m "Rossmann sales forecasting: EDA, ML pipeline, LSTM ref, Flask app"
git branch -M main
git remote add origin https://github.com/<your-username>/rossmann-sales-forecast.git
git push -u origin main
```
Include `Rossmann_Sales_Forecasting.ipynb`, `01_eda.py`, `02_modeling.py`, `webapp/`,
`submission.csv`, and the `model-*.pkl`.

## 2. Track data versions with DVC (for the screenshot requirement)

```bash
pip install dvc
dvc init
dvc add train.csv test.csv store.csv
git add train.csv.dvc test.csv.dvc store.csv.dvc .gitignore
git commit -m "Track raw data with DVC"
dvc remote add -d storage <your-remote-e.g.-s3-or-gdrive>
dvc push
```
Make a small change (e.g. re-export a cleaned CSV), `dvc add` again, commit — that gives
you the "multiple data versions" screenshot the brief asks for.

## 3. MLflow tracking (for the screenshot requirement)

```bash
pip install mlflow
```
Use the snippet in the notebook's "2.7 — Serving predictions with MLflow" section inside
`02_modeling.py` (wrap the `pipeline.fit(...)` call in an `mlflow.start_run()` block).
Re-run training a few times with different hyperparameters (e.g. `n_estimators`,
`max_depth`) to log multiple runs, then:
```bash
mlflow ui
```
Open `http://localhost:5000` and screenshot the run comparison table — that's your
"multiple model versions" evidence.

## 4. Train the LSTM

```bash
pip install tensorflow statsmodels
```
Run the LSTM cell from the notebook (section "2.6 — Deep Learning approach") in Google
Colab (free GPU) or locally. It's a small, 2-layer network by design so it trains quickly.

## 5. Deploy the Flask app

Simplest current option is **Render** (Heroku's free tier no longer exists):
```bash
cd webapp
echo "flask
pandas
numpy
scikit-learn
joblib
matplotlib
gunicorn" > requirements.txt
echo "web: gunicorn app:app" > Procfile
git init && git add . && git commit -m "Flask sales forecast app"
```
Push this folder (including a copy of your `model-*.pkl`) to its own GitHub repo, then on
render.com: "New Web Service" → connect the repo → it auto-detects the `Procfile` and
deploys. You'll get a live URL to submit.

If your course specifically requires Heroku and you already have a paid/verified account:
```bash
heroku create rossmann-sales-forecast
git push heroku main
```

## 6. Final PDF "blog" write-up

Convert the executed notebook (after you've run the LSTM section for real) to a PDF:
```bash
pip install nbconvert
jupyter nbconvert --to pdf Rossmann_Sales_Forecasting.ipynb
```
or use `File → Download as → PDF` in Jupyter/Colab. Add 2-3 paragraphs of narrative
framing at the top (the "why") since a blog post reads better with prose bridges between
sections than a raw notebook does.
