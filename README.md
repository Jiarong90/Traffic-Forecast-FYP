
Moved ML to https://github.com/Jiarong90/Traffic-Forecast-ML

# Recommended Setup (Windows)

## 1) Create and activate venv
   
python -m venv venv
.\venv\Scripts\Activate.ps1

## 2) Install dependencies

pip install -r requirements.txt

## 3) Copy .env.example to .env and fill values

copy .env.example .env

## 4) Run the server
   
uvicorn main:app --reload

## 5) Open http://127.0.0.1:8000/

## 6) To access LTA API, request for the API Key in 
https://datamall.lta.gov.sg/content/datamall/en/request-for-api.html

And fill LTA_API_KEY in .env
