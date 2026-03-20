import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# 1. Load and Standardize Data
df = pd.read_csv('matches.csv')
df.dropna(subset=['winner', 'city'], inplace=True)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Grouping old team names into their modern franchises for consistency
team_mapping = {
    'Delhi Daredevils': 'Delhi Capitals',
    'Deccan Chargers': 'Sunrisers Hyderabad',
    'Kings XI Punjab': 'Punjab Kings',
    'Rising Pune Supergiant': 'Rising Pune Supergiants',
    'Pune Warriors': 'Rising Pune Supergiants',
    'Gujarat Lions': 'Gujarat Titans'
}
for col in ['team1', 'team2', 'toss_winner', 'winner']:
    df[col] = df[col].replace(team_mapping)

# 2. Hyper-Feature Engineering
df['team1_win'] = (df['team1'] == df['winner']).astype(int)

# Feature A: Venue Win Rate (How strong is Team 1 at this specific ground?)
venue_stats = df.groupby(['venue', 'team1'])['team1_win'].mean().reset_index()
venue_stats.columns = ['venue', 'team1', 't1_venue_strength']
df = df.merge(venue_stats, on=['venue', 'team1'], how='left')

# Feature B: Rolling Form (Win rate of the last 10 matches played by each team)
def get_rolling_form(team_name, date):
    past = df[(df['date'] < date) & ((df['team1'] == team_name) | (df['team2'] == team_name))].tail(10)
    if len(past) == 0: return 0.5
    wins = len(past[past['winner'] == team_name])
    return wins / len(past)

df['t1_form'] = df.apply(lambda x: get_rolling_form(x['team1'], x['date']), axis=1)
df['t2_form'] = df.apply(lambda x: get_rolling_form(x['team2'], x['date']), axis=1)

# Feature C: Toss Advantage
df['t1_toss_win'] = (df['toss_winner'] == df['team1']).astype(int)

# 3. Prepare for Training
# We keep these as strings; CatBoost handles the encoding internally for better accuracy
features = ['team1', 'team2', 'venue', 't1_venue_strength', 't1_form', 't2_form', 't1_toss_win', 'toss_decision']
X = df[features].fillna(0.5)
y = df['team1_win']

# Split by time: Train on 80% of older matches, test on the most recent 20%
train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# 4. CatBoost Training
# Telling the model which columns are categorical text
cat_features = ['team1', 'team2', 'venue', 'toss_decision']

model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.02,
    depth=7,
    loss_function='Logloss',
    cat_features=cat_features,
    verbose=200 # Shows progress every 200 iterations
)

model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

# 5. Evaluation
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"\n✅ Final Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the Cleaned Dataset
# Including the newly engineered features for your app
df.to_csv('cleaned_matches.csv', index=False)
print("Data saved as 'cleaned_matches.csv'")

joblib.dump(model, "cricket_model.pkl")
print("Model saved as cricket_model.pkl")