import streamlit as st
import pandas as pd

st.write("""
# My first app
Hello *World!*
""")

# Loading in data
df = pd.read_csv('/Users/neharajganesh/Desktop/Projects/NBAWinPredictions/data/playoff.csv')

#title
st.title('NBA Playoff Games Viewer')

#Pick a date
selected_date = st.date_input('Select a playoff game date')

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