import streamlit as st
import pandas as pd

# Loading in data
df = pd.read_csv('/Users/neharajganesh/Desktop/Projects/NBAWinPredictions/data/playoff.csv')

# Title 
st.markdown("<h1 style='text-align: center; color: #0047AB;'>🏀 NBA Playoff Games Viewer 🏀</h1>", unsafe_allow_html=True)

#Pick a date
selected_date = st.date_input('Select a playoff game date:')

# datetime format 
df['Date'] = pd.to_datetime(df['Date'])

# filtering games by the selected date
filtered_games = df[df['Date'] == pd.to_datetime(selected_date)]

# Display
if not filtered_games.empty:
    st.write(f"Playoff games on {selected_date}:")
    st.dataframe(filtered_games)
else:
    st.write(f"No playoff games found for {selected_date}.")