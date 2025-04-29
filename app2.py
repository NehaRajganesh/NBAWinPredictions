from tensorflow.keras.models import load_model
import streamlit as st
import pandas as pd

# Load the trained model
model = load_model('nba_win_predictor.h5')

# Load the playoff games you want to predict on
df = pd.read_csv('/Users/neharajganesh/Desktop/Projects/NBAWinPredictions/data/playoff.csv')

# Preprocess data (remove unnecessary columns, scale features if needed)
X_playoff = df.drop(columns=["Result", "Date", "Team1", "Team2"])  # adjust columns you don't want

# Make predictions
predictions = model.predict(X_playoff)
predicted_classes = (predictions > 0.5).astype("int32").flatten()

# Add predictions back to DataFrame
df["Predicted_Win"] = predicted_classes

# Build Streamlit app
st.title('🏀 NBA Playoff Game Win Predictions')

selected_date = st.date_input('Select a playoff date')

df['Date'] = pd.to_datetime(df['Date'])
filtered_games = df[df['Date'] == pd.to_datetime(selected_date)]

if not filtered_games.empty:
    st.write("Predicted Results for Games:")
    for idx, row in filtered_games.iterrows():
        team1 = row['Team1']
        team2 = row['Team2']
        prediction = row['Predicted_Win']
        
        if prediction == 1:
            winner = team1
        else:
            winner = team2
        
        st.markdown(f"<h3 style='text-align: center; color: green;'>🏆 Predicted Winner: {winner}</h3>", unsafe_allow_html=True)
        st.markdown("---")
else:
    st.write("No games found for that date.")
