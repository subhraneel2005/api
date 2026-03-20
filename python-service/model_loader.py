import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model3", "cricket_model.pkl")
data_path = os.path.join(BASE_DIR, "model3", "cleaned_matches.csv")

model = joblib.load(model_path)

# load historical data
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# ---------- PRECOMPUTED TABLES ----------

# venue strength
venue_stats = df.groupby(['venue', 'team1'])['team1_win'].mean().reset_index()
venue_stats.columns = ['venue', 'team1', 't1_venue_strength']


def get_venue_strength(team, venue):
    row = venue_stats[
        (venue_stats['team1'] == team) &
        (venue_stats['venue'] == venue)
    ]
    return row['t1_venue_strength'].values[0] if len(row) else 0.5


def get_form(team):
    past = df[
        (df['team1'] == team) | (df['team2'] == team)
    ].tail(10)

    if len(past) == 0:
        return 0.5

    wins = len(past[past['winner'] == team])
    return wins / len(past)


# ---------- PREPROCESS ----------

features = [
    'team1', 'team2', 'venue',
    't1_venue_strength', 't1_form', 't2_form',
    't1_toss_win', 'toss_decision'
]


def preprocess(input_data):
    team1 = input_data['team1']
    team2 = input_data['team2']
    venue = input_data['venue']

    enriched = {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_decision": input_data["toss_decision"],

        # engineered features
        "t1_toss_win": 1 if input_data["toss_winner"] == team1 else 0,
        "t1_venue_strength": get_venue_strength(team1, venue),
        "t1_form": get_form(team1),
        "t2_form": get_form(team2),
    }

    df_input = pd.DataFrame([enriched]).fillna(0.5)
    return df_input[features]


# ---------- PREDICT ----------

def predict(input_data):
    df_input = preprocess(input_data)

    pred = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]

    # probability of team1 winning = proba[1]
    confidence = float(proba[1] if pred == 1 else proba[0])

    winner = input_data["team1"] if pred == 1 else input_data["team2"]

    return winner, confidence