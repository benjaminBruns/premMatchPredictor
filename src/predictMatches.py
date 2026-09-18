import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

matches = pd.read_csv("./data/cleaned_prem_matches.csv")

predictors = ["team-code", "opponent-code", "venue-code", "day-code", "season-code"]

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(5, closed="left").mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ["points", "goal-difference", "goals-for", "goals-against", "shots-for", "shots-against", "sot-for", "sot-against"]
new_cols = [f"{c}-rolling" for c in cols]

# groupby creates one dataframe for each team in the data. lambda applies the function to the teams
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling.droplevel("team")

# Unique index values
matches_rolling = matches_rolling.reset_index()


# Predicting Future Matches

def build_future_row(team, opponent, venue, date, season, matches, matches_rolling):
    # Define the codes
    team_code = matches[matches["team"] == team]["team-code"].iloc[0]
    opp_code = matches[matches["team"] == opponent]["team-code"].iloc[0]
    venue_code = 1 if venue == "home" else 0

    if season not in matches["season"].unique():
        season_code = matches["season-code"].max() + 1
    else:
        season_code = matches[matches["season"] == season]["season-code"].iloc[0]
    

    row = {
        "team-code": team_code,
        "opponent-code": opp_code,
        "venue-code": venue_code,
        "day-code": date.weekday(),
        "season-code": season_code,
    }

    # Fetch rolling stats for teams
    team_rolling = matches_rolling[matches_rolling["team"] == team].iloc[-1]
    opp_rolling = matches_rolling[matches_rolling["team"] == opponent].iloc[-1]

    for col in new_cols:
        if ("for" in col) or ("points" in col) or ("goal-difference" in col):
            row[col] = team_rolling[col]
        else:
            row[col] = opp_rolling[col] 

    return pd.DataFrame([row])

def predict_matches(fixtures):
    results = []
    for f in fixtures:
        row = build_future_row(
            f["team"], f["opponent"], f["venue"], f["date"], f["season"], matches, matches_rolling
        )
        pred = model.predict(row[predictors + new_cols])[0]
        if pred == "W":
            answer = f"{f['team']} wins"
        elif pred == "L":
            answer = f"{f['opponent']} wins"
        else:
            answer = "Draw"
        results.append(f"Prediction for {f['team']} vs {f['opponent']}: {answer}")
    return results

def train_model(data, predictors):
    train = data[data["date"] < "2026-07-01"]

    model = RandomForestClassifier(n_estimators=400, min_samples_split=5, random_state=1)
    model.fit(train[predictors], train["target"])
    return model


# Finally running the model
fixtures = [
{
    "team": "brentford",
    "opponent": "chelsea",
    "venue": "home",
    "date": pd.Timestamp("2026-09-18"),
    "season": "2026-2027",
},
{
    "team": "tottenham",
    "opponent": "aston villa",
    "venue": "home",
    "date": pd.Timestamp("2026-09-19"),
    "season": "2026-2027",
},
{
    "team": "brighton",
    "opponent": "arsenal",
    "venue": "home",
    "date": pd.Timestamp("2026-09-19"),
    "season": "2026-2027",
},
{
    "team": "everton",
    "opponent": "ipswich",
    "venue": "home",
    "date": pd.Timestamp("2026-09-19"),
    "season": "2026-2027",
},
{
    "team": "newcastle",
    "opponent": "hull",
    "venue": "home",
    "date": pd.Timestamp("2026-09-19"),
    "season": "2026-2027",
},
{
    "team": "nott'm forest",
    "opponent": "coventry",
    "venue": "home",
    "date": pd.Timestamp("2026-09-19"),
    "season": "2026-2027",
},
{
    "team": "bournemouth",
    "opponent": "liverpool",
    "venue": "home",
    "date": pd.Timestamp("2026-09-20"),
    "season": "2026-2027",
},
{
    "team": "leeds",
    "opponent": "crystal palace",
    "venue": "home",
    "date": pd.Timestamp("2026-09-20"),
    "season": "2026-2027",
},
{
    "team": "man city",
    "opponent": "sunderland",
    "venue": "home",
    "date": pd.Timestamp("2026-09-20"),
    "season": "2026-2027",
},
{
    "team": "fulham",
    "opponent": "man united",
    "venue": "home",
    "date": pd.Timestamp("2026-09-20"),
    "season": "2026-2027",
},
]

model = train_model(matches_rolling, predictors + new_cols)
predictions = predict_matches(fixtures)

# Clears the file
with open("./data/results/predictions.txt", "w") as file:
    pass

for pred in predictions:
    print(pred)
    with open("./data/results/predictions.txt", "a") as file:
        file.write(f"{pred}\n")
