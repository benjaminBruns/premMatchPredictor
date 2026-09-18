import kagglehub
import pandas as pd

# path = kagglehub.dataset_download(
#     "panaaaaa/english-premier-league-and-championship-full-dataset",
#     output_dir="./data",
#     force_download=True
# )

# Combine the training and testing datasets together for cleaning
historical_df = pd.read_csv("./data/England CSV.csv")
current_season_df = pd.read_csv("./data/seasonData26-27.csv")
df = pd.concat([historical_df, current_season_df], ignore_index=True)

# Column formatting
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(" ", "-")
df.columns = df.columns.str.replace("_", "-")

# Date data type to datetime
df["date"] = pd.to_datetime(df["date"], dayfirst=True)

# Add day of week and day code columns to end
df['day-of-week'] = df['date'].dt.day_name()
df['day-code'] = df['date'].dt.weekday

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
df["season"] = df["date"].apply(lambda x: 
                                f"{x.year}-{x.year + 1}" 
                                if x.month >= 8 
                                else f"{x.year - 1}-{x.year}"
                                )

# Data is currently match-level, we need to convert it to team-level
home_df = pd.DataFrame({
    "date": df["date"],
    "season": df["season"],
    "day-of-week": df["day-of-week"],
    "day-code": df["day-code"],

    "team": df["hometeam"],
    "opponent": df["awayteam"],

    "venue": "home",

    "goals-for": df["fth-goals"],
    "goals-against": df["fta-goals"],

    "shots-for": df["h-shots"],
    "shots-against": df["a-shots"],

    "sot-for": df["h-sot"],
    "sot-against": df["a-sot"],

    "result": df["ft-result"].map({
        "H": "W",
        "D": "D",
        "A": "L"
    })
})

away_df = pd.DataFrame({
    "date": df["date"],
    "season": df["season"],
    "day-of-week": df["day-of-week"],
    "day-code": df["day-code"],

    "team": df["awayteam"],
    "opponent": df["hometeam"],

    "venue": "away",

    "goals-for": df["fta-goals"],
    "goals-against": df["fth-goals"],

    "shots-for": df["a-shots"],
    "shots-against": df["h-shots"],

    "sot-for": df["a-sot"],
    "sot-against": df["h-sot"],

    "result": df["ft-result"].map({
        "H": "L",
        "D": "D",
        "A": "W"
    })
})

team_df = pd.concat(
    [home_df, away_df],
    ignore_index = True
)

# Numeric encodings used by the team-level model.
team_names = pd.concat([team_df["team"], team_df["opponent"]]).unique()
team_codes = {team: code for code, team in enumerate(team_names)}
team_df["team-code"] = team_df["team"].map(team_codes)
team_df["opponent-code"] = team_df["opponent"].map(team_codes)
team_df["venue-code"] = team_df["venue"].map({"home": 1, "away": 0})

season_names = team_df["season"].sort_values().unique()
season_codes = {season: code for code, season in enumerate(season_names)}
team_df["season-code"] = team_df["season"].map(season_codes)
team_df["target"] = team_df["result"]


## Other features

# Goal difference
team_df["goal-difference"] = (
    team_df["goals-for"] - team_df["goals-against"]
)

# Points
team_df["points"] = team_df["result"].map({
    "W": 3,
    "D": 1,
    "L": 0
})

# Sort by team chronologically
team_df = team_df.sort_values(
    ["date", "team"]
).reset_index(drop=True)

# Update the cleaned data to new csv file
team_df.to_csv("./data/cleaned_prem_matches.csv", index=False)