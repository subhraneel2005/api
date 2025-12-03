import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import os

# ------------------ LOAD DATA ------------------
df = pd.read_csv('matches.csv')

# Drop matches with missing winner or city
df.dropna(subset=['winner', 'city'], inplace=True)

# Standardize team names
team_mapping = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Pune Warriors': 'Rising Pune Supergiants',
    'Gujarat Lions': 'Kochi Tuskers Kerala',
}
for col in ['team1', 'team2', 'toss_winner', 'winner']:
    df[col] = df[col].replace(team_mapping)

# ------------------ FEATURE ENGINEERING ------------------

df['team1_win'] = (df['team1'] == df['winner']).astype(int)
df['toss_match_winner'] = (df['toss_winner'] == df['winner']).astype(int)

def get_matchup_key(row):
    teams = sorted([row['team1'], row['team2']])
    return f"{teams[0]}_vs_{teams[1]}"

df['matchup_key'] = df.apply(get_matchup_key, axis=1)

def calculate_h2h_win_rate(df, team1, team2, matchup_key):
    h2h_matches = df[df['matchup_key'] == matchup_key]
    team1_wins = h2h_matches[h2h_matches['winner'] == team1].shape[0]
    total_matches = h2h_matches.shape[0]
    return team1_wins / total_matches if total_matches > 0 else 0.5

df['team1_h2h_win_rate'] = df.apply(
    lambda row: calculate_h2h_win_rate(df, row['team1'], row['team2'], row['matchup_key']),
    axis=1
)

venue_stats = df.groupby(['venue', 'winner']).size().reset_index(name='wins')
venue_stats.rename(columns={'winner': 'team'}, inplace=True)

venue_team1 = df[['venue', 'team1']].rename(columns={'team1': 'team'})
venue_team2 = df[['venue', 'team2']].rename(columns={'team2': 'team'})
all_venue_team = pd.concat([venue_team1, venue_team2])
venue_total = all_venue_team.groupby(['venue', 'team']).size().reset_index(name='total_matches')

venue_win_rates = pd.merge(venue_total, venue_stats, on=['venue', 'team'], how='left').fillna(0)
venue_win_rates['win_rate'] = venue_win_rates['wins'] / venue_win_rates['total_matches']

def get_venue_win_rate(row, team_col):
    venue = row['venue']
    team = row[team_col]
    rate = venue_win_rates[
        (venue_win_rates['venue'] == venue) &
        (venue_win_rates['team'] == team)
    ]['win_rate'].values
    return rate[0] if len(rate) else 0.5

df['team1_venue_win_rate'] = df.apply(lambda row: get_venue_win_rate(row, 'team1'), axis=1)
df['team2_venue_win_rate'] = df.apply(lambda row: get_venue_win_rate(row, 'team2'), axis=1)
df['venue_rate_diff'] = df['team1_venue_win_rate'] - df['team2_venue_win_rate']

# ------------------ MODEL TRAINING ------------------

features = [
    'team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'city',
    'toss_match_winner', 'team1_h2h_win_rate', 'venue_rate_diff'
]
target = 'team1_win'

data = df[features + [target]].copy()

categorical_features = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'city']
data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=False)

X = data_encoded.drop(target, axis=1)
y = data_encoded[target]

# Save OHE columns for prediction
joblib.dump(X.columns.tolist(), "ohe_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nFinal Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "cricket_model.pkl")
print("Model saved as cricket_model.pkl")
print("OHE column order saved as ohe_columns.pkl")

# ------------------ EXPORT CLEANED DATA ------------------

engineered_cols = [
    'team1_win', 'toss_match_winner', 'team1_h2h_win_rate',
    'team1_venue_win_rate', 'team2_venue_win_rate', 'venue_rate_diff',
    'matchup_key'
]

final_df = df[df.columns.tolist() + engineered_cols].copy()
final_df.to_csv('cleaned_ipl_data_new.csv', index=False)
print("Cleaned dataset saved!")
