# Premier League Predictor

This project cleans Premier League match data and uses it in a machine learning match predictor model.

## Setup

Use Python 3.10 or newer.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Prepare the data

Run the scraper from the project root:

```powershell
python src/scraping.py
```

The script downloads the dataset from Kaggle into `data/`, cleans the match data, and writes the result to:

```text
data/cleaned_prem_matches.csv
```

The script expects the downloaded dataset to contain `England CSV.csv`.

## Project structure

```text
data/
src/
    main.py
    scraping.py
requirements.txt
README.md
```

## Dataset

The source data is downloaded from the Kaggle dataset [English Premier League and Championship Full Dataset](https://www.kaggle.com/datasets/panaaaaa/english-premier-league-and-championship-full-dataset).

Review the dataset's Kaggle license and terms before redistributing the raw or cleaned CSV files.
