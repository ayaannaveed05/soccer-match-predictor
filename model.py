import pandas as pd
import requests
import os

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


#Automatic download of the current 2025/26 season
season_urls = {
    "SP1.csv": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "E0.csv": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "F1.csv": "https://www.football-data.co.uk/mmz4281/2526/F1.csv",
    "I1.csv": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "D1.csv": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
}

download_folder = os.path.join(os.path.dirname(__file__), "data")

for filename, url in season_urls.items(): 
    path = os.path.join(download_folder, filename) # creates the full file path to save the CSV
    try:
        response = requests.get(url, timeout=10)  # attempt to download file from URL, if takes more than 10s, stops trying
        response.raise_for_status()  # throws error if an issue

        try:
            with open(path, "wb") as f:  # opens in binary write form and saves the downloaded content
                f.write(response.content)
            print(f"✅ Downloaded {filename}")
        except PermissionError:
            print(f"⚠️ Skipped {filename} (file is open or locked — close it and rerun).")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {filename}: {e}")


files = {
    "SP1.csv": "Spain",  
    "SP1 (1).csv": "Spain",
    "SP1 (2).csv": "Spain",
    "E0.csv": "England",
    "E0 (1).csv": "England",
    "E0 (2).csv": "England",
    "F1.csv": "France",
    "F1 (1).csv": "France",
    "F1 (2).csv": "France",
    "I1.csv": "Italy",
    "I1 (1).csv": "Italy",
    "I1 (2).csv": "Italy",
    "D1.csv": "Germany",
    "D1 (1).csv": "Germany",
    "D1 (2).csv": "Germany"
}

dfs = []
for file, league in files.items():  # loops through file and league
    path = os.path.join(download_folder, file)
    if os.path.exists(path):  # checks if file exists
        temp_df = pd.read_csv(path)   # reads the CSV file into pandas df
        temp_df['league'] = league  # adds a new column 'leagues' to know what country the data came from
        dfs.append(temp_df)  #adds the df to the list of dfs
    else:
        print(f"⚠️ File not found: {file}")

df = pd.concat(dfs, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce') #converts date into datetime object, in European style
df = df.dropna(subset=['Date']) #drops rows where date is missing
df.rename(columns={'HomeTeam': 'home_team', 'AwayTeam': 'away_team',
                   'FTHG': 'home_goals', 'FTAG': 'away_goals', 'FTR': 'result'}, inplace=True)
df['winner'] = df['result'].map({'H': 'home', 'A': 'away', 'D': 'draw'}) #create a new column "winner" and converts the single letters into words


def get_valid_team(prompt, df):  # checks if name is valid
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    teams_lower = [t.lower() for t in teams]

    while True:
        team_input = input(prompt).strip()
        if team_input.lower() in teams_lower:
            return teams[teams_lower.index(team_input.lower())]
        else:
            print("⚠️ Team not found. Please enter a valid team name.")

def get_home_or_away(prompt):  # checks if team is home or away
    while True:
        ha = input(prompt).strip().lower()
        if ha in ['home', 'away']:
            return ha
        else:
            print("⚠️ Invalid input. Please type 'home' or 'away'.")


team_name = get_valid_team("Enter the team name you want to analyze: ", df)
team_league = df[df['home_team'] == team_name]['league'].iloc[0]

# Filter for the league of the team
df_league = df[df['league'] == team_league].copy()  # use .copy() to avoid SettingWithCopyWarning

# Prompt for opponent until they are in the same league
while True:
    next_opponent = get_valid_team(f"Enter opponent team name: ", df)
    opponent_league = df[df['home_team'] == next_opponent]['league'].iloc[0]
    if opponent_league == team_league:
        break
    else:
        print(f"⚠️ {next_opponent} is not in the same league as {team_name} ({team_league}). Please choose a team in the same league.")

home_or_away = get_home_or_away(f"Is your team playing at home or away vs {next_opponent}? (home/away): ")


df['opponent'] = df.apply(lambda row: row['away_team'] if row['home_team'] == team_name else row['home_team'], axis=1)

# axis=1 applies .apply() function to each row
# lambda row checks if my team is home; if not, it is away
# creates a new column "opponent" which lists the opposing team of every match my team faces

def get_recent_form(df, team, num_matches=5):  # calculates last 5 games
    team_home = df[df['home_team'] == team].tail(num_matches)  # get home matches of team
    team_away = df[df['away_team'] == team].tail(num_matches)  # get away matches of team
    recent_matches = pd.concat([team_home, team_away]).sort_values('Date', ascending=False).head(num_matches)
    # combines both home and away matches, sorts them by date (with newest first), and takes most recent 5

    goals_scored = []
    goals_conceded = []

    for _, row in recent_matches.iterrows():  
        if row['home_team'] == team:  # if home team, add home goals to scored, and away goals to conceded
            goals_scored.append(row['home_goals'])
            goals_conceded.append(row['away_goals'])
        else:
            goals_scored.append(row['away_goals'])
            goals_conceded.append(row['home_goals'])

    avg_scored = sum(goals_scored) / len(goals_scored)  
    avg_conceded = sum(goals_conceded) / len(goals_conceded)
    form = avg_scored - avg_conceded  #scoring more than conceded, positive if more goals scored than conceded

    return avg_scored, avg_conceded, form, recent_matches

le = LabelEncoder()
df['winner_encoded'] = le.fit_transform(df['winner'])
#a LabelEncoder which converts text labels into labels, so makes
#home=0, away=1, draw=2

# calculates rolling averages (how many goals this team has scored at home in last 5 games and averages it)
df_league['home_recent_goals'] = df_league.groupby('home_team')['home_goals'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
df_league['home_recent_conceded'] = df_league.groupby('home_team')['away_goals'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
df_league['away_recent_goals'] = df_league.groupby('away_team')['away_goals'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
df_league['away_recent_conceded'] = df_league.groupby('away_team')['home_goals'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)

df_league['home_form'] = df_league['home_recent_goals'] - df_league['home_recent_conceded']
df_league['away_form'] = df_league['away_recent_goals'] - df_league['away_recent_conceded']

# Calculate rolling home advantage (last 5 home games vs last 5 away games)
df_league['home_team_home_form'] = df_league.groupby('home_team')['home_form'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
df_league['away_team_away_form'] = df_league.groupby('away_team')['away_form'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)

# Home advantage = recent home form vs recent away form
df_league['home_advantage'] = df_league['home_team_home_form'] - df_league['away_team_away_form']

df = df.sort_values('Date')  # sort once so time flows forward

weights = [0.4, 0.3, 0.15, 0.1, 0.05]
h2h_home_goals, h2h_away_goals, h2h_home_conceded, h2h_away_conceded = [], [], [], []

for idx, row in df_league.iterrows():
    team = row['home_team']
    opponent = row['away_team']
    match_date = row['Date']

    # find only league-specific H2H matches before this match
    h2h = df_league[
        (
            ((df_league['home_team'] == team) & (df_league['away_team'] == opponent)) |
            ((df_league['home_team'] == opponent) & (df_league['away_team'] == team))
        ) & (df_league['Date'] < match_date)
    ].sort_values('Date', ascending=False).head(5)

    hg = ag = hc = ac = 0
    for i, (_, r) in enumerate(h2h.iterrows()):
        w = weights[i]
        if r['home_team'] == team:
            hg += r['home_goals'] * w
            hc += r['away_goals'] * w
        else:
            ag += r['away_goals'] * w
            ac += r['home_goals'] * w

    h2h_home_goals.append(hg)
    h2h_away_goals.append(ag)
    h2h_home_conceded.append(hc)
    h2h_away_conceded.append(ac)

df_league['h2h_home_goals'] = h2h_home_goals
df_league['h2h_away_goals'] = h2h_away_goals
df_league['h2h_home_conceded'] = h2h_home_conceded
df_league['h2h_away_conceded'] = h2h_away_conceded


features = [
    'home_recent_goals', 'away_recent_goals',
    'home_recent_conceded', 'away_recent_conceded',
    'home_form', 'away_form',
    'h2h_home_goals', 'h2h_away_goals',
    'h2h_home_conceded', 'h2h_away_conceded',
    'home_advantage'
]

df_league['winner_encoded'] = le.transform(df_league['winner'])

X = df_league[features]
y = df_league['winner_encoded'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
# model learns 80% from training, 20% is tested on unseen games
print("\n" + "="*60)
print("🎯 BASELINE COMPARISONS")
print("="*60)

# Baseline 1: Always predict home win
always_home_acc = (y_test == le.transform(['home'])[0]).sum() / len(y_test)
print(f"Always predict home win: {always_home_acc:.1%}")

# Baseline 2: Always predict most common class
most_common_class = y_train.value_counts().idxmax()
most_common_acc = (y_test == most_common_class).sum() / len(y_test)
print(f"Always predict most common class: {most_common_acc:.1%}")

# Baseline 3: Random guess
print(f"Random guessing: 33.3%")

print("\n" + "="*60)
print("🔍 HYPERPARAMETER TUNING (GridSearchCV)")
print("="*60)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

tscv = TimeSeriesSplit(n_splits=3)

print("Running grid search (this may take 1-2 minutes)...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=tscv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train, y_train)

print(f"\n✅ Best parameters found:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest cross-validation score: {grid_search.best_score_:.1%}")

model = grid_search.best_estimator_
# Note: Logistic Regression outperformed Random Forest in testing.
# In production, would use LR as the deployed model.

print("\n" + "="*60)
print("📊 MODEL PERFORMANCE")
print("="*60)

y_pred = model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Random Forest Test Accuracy: {rf_accuracy:.1%}")
print(f"   Improvement over 'always home': +{(rf_accuracy - always_home_acc)*100:.1f} percentage points")
print(f"   Improvement over random guess: +{(rf_accuracy - 0.333)*100:.1f} percentage points")

print("\nClassification Report:")
# Note: Low recall on draws (9%) is expected - draws are inherently difficult to predict
# due to their dependence on unpredictable late-game events (injuries, red cards, referee decisions).
print(classification_report(y_test, y_pred, target_names=le.classes_))

from sklearn.linear_model import LogisticRegression

# class_weight='balanced' helps reduce home-win bias
baseline = LogisticRegression(max_iter=1000, class_weight='balanced')
baseline.fit(X_train, y_train)

baseline_pred = baseline.predict(X_test)

baseline_accuracy = accuracy_score(y_test, baseline_pred)

print(f"\n📉 Baseline Logistic Regression: {baseline_accuracy:.1%}")
print(classification_report(y_test, baseline_pred, target_names=le.classes_))

print("\n" + "="*60)
print("📊 FEATURE IMPORTANCE ANALYSIS")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nWhat drives predictions (ranked by importance):\n")
for idx, row in feature_importance.iterrows():
    bar_length = int(row['importance'] * 50)
    bar = '█' * bar_length
    print(f"{row['feature']:25s} {bar} {row['importance']:.3f}")

top_3_importance = feature_importance.head(3)['importance'].sum()
print(f"\n💡 Insight: Top 3 features account for {top_3_importance:.1%} of predictions")

h2h_matches = df_league[((df_league['home_team'] == team_name) & (df_league['away_team'] == next_opponent)) |
                         ((df_league['home_team'] == next_opponent) & (df_league['away_team'] == team_name))] \
                        .sort_values('Date', ascending=False).head(5)

if not h2h_matches.empty:  # prints out the head to head matches
    print(f"\nMost recent head-to-head matches between {team_name} and {next_opponent}:")
    for _, row in h2h_matches.iterrows():
        print(f"{row['home_team']} {row['home_goals']} - {row['away_goals']} {row['away_team']} ({row['Date'].date()})")
else:
    print(f"\nNo head-to-head matches found between {team_name} and {next_opponent}.")


latest_match = df_league[(df_league['home_team'] == team_name) | (df_league['away_team'] == team_name)].iloc[-1]
# pulls the most recent match that involves them, .iloc[-1] takes the last rows

h2h_home_goals = latest_match['h2h_home_goals']
h2h_away_goals = latest_match['h2h_away_goals']
h2h_home_conceded = latest_match['h2h_home_conceded']
h2h_away_conceded = latest_match['h2h_away_conceded']

# end and beginning of season, knowing where to get the info out of
season_start = pd.to_datetime("2025-08-01")
season_end = pd.to_datetime("2026-05-31")

# Filter df_league for current season only
df_league_season = df_league[(df_league['Date'] >= season_start) & (df_league['Date'] <= season_end)]

# Get recent form from current season
home_recent_goals, home_recent_conceded, home_form, team_last_5 = get_recent_form(df_league_season, team_name)
away_recent_goals, away_recent_conceded, away_form, opponent_last_5 = get_recent_form(df_league_season, next_opponent)


print(f"\nLast 5 matches for {team_name}:")  # gets the last 5 matches for team and opponent
for _, row in team_last_5.iterrows():
    print(f"{row['home_team']} {row['home_goals']} - {row['away_goals']} {row['away_team']} ({row['Date'].date()})")

print(f"\nLast 5 matches for {next_opponent}:")
for _, row in opponent_last_5.iterrows():
    print(f"{row['home_team']} {row['home_goals']} - {row['away_goals']} {row['away_team']} ({row['Date'].date()})")

# Assign correct home/away orientation for the prediction
if home_or_away.lower() == 'home':
    next_home_team = team_name
    next_away_team = next_opponent
else:
    next_home_team = next_opponent
    next_away_team = team_name

# Get recent form ONLY from current season for prediction
# ensures we do not accidentally use future games
home_recent_goals, home_recent_conceded, home_form, _ = get_recent_form(df_league_season, next_home_team)
away_recent_goals, away_recent_conceded, away_form, _ = get_recent_form(df_league_season, next_away_team)

home_team_home_form = df_league_season[df_league_season['home_team'] == next_home_team]['home_form'].tail(5).mean()
away_team_away_form = df_league_season[df_league_season['away_team'] == next_away_team]['away_form'].tail(5).mean()
home_advantage = home_team_home_form - away_team_away_form

# row with 10 columns with the features
next_match_features = pd.DataFrame([{
    'home_recent_goals': home_recent_goals,
    'away_recent_goals': away_recent_goals,
    'home_recent_conceded': home_recent_conceded,
    'away_recent_conceded': away_recent_conceded,
    'home_form': home_form,
    'away_form': away_form,
    'h2h_home_goals': h2h_home_goals,
    'h2h_away_goals': h2h_away_goals,
    'h2h_home_conceded': h2h_home_conceded,
    'h2h_away_conceded': h2h_away_conceded,
    'home_advantage': home_advantage
}])

predicted_winner_encoded = model.predict(next_match_features)[0]
# uses the trained Random Forest model to predict the winner for the given features

predicted_probs = model.predict_proba(next_match_features)[0] 

predicted_winner = le.inverse_transform([predicted_winner_encoded])[0]

# predicts the result
print("\n" + "="*60)
print("⚽ MATCH PREDICTION")
print("="*60)
print(f"{next_home_team} vs {next_away_team}")
if predicted_winner == 'home':
    print(f"✅ Predicted Winner: {next_home_team}")
elif predicted_winner == 'away':
    print(f"✅ Predicted Winner: {next_away_team}")
else:
    print("🤝 Predicted Result: Draw")

prob_labels = {
    'home': next_home_team,
    'away': next_away_team,
    'draw': 'Draw'
}

print("\n📊 Probability Breakdown:")
for label, prob in zip(le.classes_, predicted_probs):
    print(f"{prob_labels[label]}: {prob*100:.1f}%")

print("\n" + "="*60)



