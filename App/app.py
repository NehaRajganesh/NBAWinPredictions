import streamlit as st
import pandas as pd
import joblib


# ---- App layout ---- #
st.set_page_config(page_title="NBA Win Forecast", page_icon="🏀")
st.title("🏀 NBA Game Win Forecaster")


# ---- Load model ---- #
@st.cache_resource
def load_model():
    return joblib.load('Models/nba_model.pkl')

model = load_model()

# ---- Load scaler ---- #
@st.cache_resource
def load_scaler():
    return joblib.load('Models/scaler.pkl')

scaler = load_scaler()


# ---- Load team averages ---- #
@st.cache_data
def load_team_stats():
    # Example: load average stats from a CSV
    team_stats = pd.read_csv('data/full_ts.csv')

    cols_to_drop = ['3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'Rk_x', 'Rk_y',
        'Unnamed: 0', 'Unnamed: 17', 'Unnamed: 22', 'Unnamed: 27', 'Arena', 'Attend.', 'FT/FGA.1', 'FTr', 'FT/FGA', 'eFG%.1',
        'W', 'L', 'PW', 'PL', 'SOS', 'SRS', 'MOV', 'TOV', 'PF', 'TOV%.1',
        'MP', 'G', 'FG', 'FGA', 'ORB', 'DRB', 'TRB']

    team_stats = team_stats.drop(columns = cols_to_drop)

    last10 = pd.read_csv('data/nba_last10_win.csv')
    team_abbreviations = {
    'Atlanta Hawks': 'ATL',
    'Boston Celtics': 'BOS',
    'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI',
    'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW',
    'Houston Rockets': 'HOU',
    'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC',
    'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP',
    'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL',
    'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHO',
    'Portland Trail Blazers': 'POR',
    'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
}

    # Replace for Home and Visitor columns
    last10['Team'] = last10['Team'].replace(team_abbreviations)

    last10.drop(columns = ['L10', 'L10_W', 'L10_L'], inplace = True)

    team_stats = pd.merge(team_stats, last10, how = 'left', left_on = 'Team', right_on = 'Team').rename(columns = {'L10_Win_Pct' : 'Win Pct'})
    team_stats.set_index('Team', inplace=True)

    return team_stats

team_stats = load_team_stats()



st.write("""
Pick two NBA teams and forecast who is more likely to win based on regular season stats.
""")

# Team selection
teams = team_stats.index.tolist()
home_team = st.selectbox("Select Home Team", teams)
visitor_team = st.selectbox("Select Visitor Team", teams)

st.image(f"NBA/{home_team}.png", width=100, caption=f"Home: {home_team}")
st.image(f"NBA/{visitor_team}.png", width=100, caption=f"Visitor: {visitor_team}")

if st.button("Predict Winner"):
    if home_team == visitor_team:
        st.error("Home and Visitor teams cannot be the same.")
    else:
        home_stats = team_stats.loc[home_team]
        visitor_stats = team_stats.loc[visitor_team]
        feature_diff = home_stats - visitor_stats

        # Now, rename columns to match model training
        feature_diff.index = [f"{col} Difference" for col in feature_diff.index]

        # Optional: clip if needed before scaling
        feature_diff = feature_diff.clip(lower=-5, upper=5)

        # Rename to match training format
        feature_diff_df = pd.DataFrame([feature_diff])

        # Normalize
        feature_diff_scaled = scaler.transform(feature_diff_df)

        # Predict using calibrated model
        proba = model.predict_proba(feature_diff_scaled)[0]

        home_win_proba = proba[1]
        visitor_win_proba = proba[0]

        # Display
        st.subheader(f"Prediction for {home_team} vs {visitor_team}")
        st.write(f"🏠 {home_team} Win Probability: {home_win_proba:.2%}")
        st.write(f"🛫 {visitor_team} Win Probability: {visitor_win_proba:.2%}")


