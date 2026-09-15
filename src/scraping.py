import kagglehub
import pandas as pd

path = kagglehub.dataset_download(
    "panaaaaa/english-premier-league-and-championship-full-dataset",
    output_dir="./data"
)

# Read the csv files into pandas dataframes
df = pd.read_csv("./data/England CSV.csv")
# df2 = pd.read_csv("./data/England 2 CSV.csv")

# Column formatting
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(" ", "-")
df.columns = df.columns.str.replace("_", "-")

# Date data type to datetime
df["date"] = pd.to_datetime(df["date"])

# Clean team names
df["hometeam"] = df["hometeam"].str.lower().str.strip()
df["awayteam"] = df["awayteam"].str.lower().str.strip()
# For teams with inconsistent naming
team_map = {
    "brighton & hove albion": "brighton",
    "ipswich town": "ipswich"
}
df["hometeam"] = df["hometeam"].replace(team_map)
df["awayteam"] = df["awayteam"].replace(team_map)

# Drop duplicates and rows with missing values in important columns
df = df.drop_duplicates()
df = df.dropna(subset=["hometeam", "awayteam", "fth-goals", "fta-goals", "ft-result"])

# Create a season column. If month is above 7, season is current year and next year
# If month is below 8, season is previous year and current year
df["season"] = df["date"].apply(lambda x: f"{x.year}-{x.year + 1}" if x.month >= 8 else f"{x.year - 1}-{x.year}")

# Sort by date and team
df = df.sort_values(by=["date", "hometeam"]).reset_index(drop=True)

df.to_csv("./data/cleaned_prem_matches.csv", index=False)

