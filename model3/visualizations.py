import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load cleaned dataset
df = pd.read_csv("cleaned_matches.csv")

# Load trained model
model = joblib.load("cricket_model.pkl")

sns.set_style("whitegrid")

# 1 Matches per season
plt.figure()
df['season'].value_counts().sort_index().plot(kind='bar')
plt.title("Matches per Season")
plt.xlabel("Season")
plt.ylabel("Matches")
plt.tight_layout()
plt.show()

# 2 Wins by team
plt.figure()
df['winner'].value_counts().plot(kind='bar')
plt.title("Wins by Team")
plt.xlabel("Team")
plt.ylabel("Wins")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 3 Toss decision distribution
plt.figure()
df['toss_decision'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Toss Decision Distribution")
plt.ylabel("")
plt.tight_layout()
plt.show()

# 4 Player of the match leaders
plt.figure()
df['player_of_match'].value_counts().head(15).plot(kind='barh')
plt.title("Top Player of the Match Awards")
plt.xlabel("Awards")
plt.tight_layout()
plt.show()

# 5 Matches by city
plt.figure()
df['city'].value_counts().head(15).plot(kind='bar')
plt.title("Matches by City")
plt.xlabel("City")
plt.ylabel("Matches")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 6 Result margin distribution
plt.figure()
df['result_margin'].dropna().plot(kind='hist', bins=30)
plt.title("Win Margin Distribution")
plt.xlabel("Margin")
plt.tight_layout()
plt.show()

# 7 Target runs distribution
plt.figure()
df['target_runs'].dropna().plot(kind='hist', bins=30)
plt.title("Target Runs Distribution")
plt.xlabel("Target Runs")
plt.tight_layout()
plt.show()

# 8 Team form vs win
plt.figure()
sns.scatterplot(x='t1_form', y='team1_win', data=df)
plt.title("Team Form vs Win")
plt.tight_layout()
plt.show()

# 9 Venue strength vs win
plt.figure()
sns.boxplot(x='team1_win', y='t1_venue_strength', data=df)
plt.title("Venue Strength vs Win")
plt.tight_layout()
plt.show()

# 10 Toss advantage vs win
plt.figure()
sns.barplot(x='t1_toss_win', y='team1_win', data=df)
plt.title("Toss Advantage vs Win")
plt.tight_layout()
plt.show()

# 11 Average target runs by season
plt.figure()
df.groupby('season')['target_runs'].mean().plot()
plt.title("Average Target Runs by Season")
plt.xlabel("Season")
plt.ylabel("Average Target")
plt.tight_layout()
plt.show()

# 12 Feature importance from model
features = ['team1','team2','venue','t1_venue_strength','t1_form','t2_form','t1_toss_win','toss_decision']
importance = model.get_feature_importance()

plt.figure()
plt.barh(features, importance)
plt.title("Model Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()