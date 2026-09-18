import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report

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

def make_predictions(data, predictors):
    train = data[data["date"] < "2023-07-01"]
    test = data[data["date"] > "2023-07-01"]

    model = RandomForestClassifier(n_estimators=400, min_samples_split=5, class_weight="balanced", random_state=1)
    model.fit(train[predictors], train["target"])
    preds = model.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    prec = precision_score(test["target"], preds, average="weighted")
    report = classification_report(test["target"], preds)
    return combined, prec, report

combined, prec, report = make_predictions(matches_rolling, predictors + new_cols)
combined = combined.merge(matches_rolling[["date", "team", "opponent"]], left_index=True, right_index=True)

combined.to_csv("./data/results/combined.csv")
with open("./data/results/precisionScore.txt", "w") as file:
    file.write(f"{prec}")

with open("./data/results/classificationReport.csv", "w") as file:
    file.write(report)

merged = combined.merge(combined, left_on=["date", "team"], right_on=["date", "opponent"])
merged.to_csv("./data/results/merged.csv")