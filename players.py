#Load in CSV dataset 
import pandas as pd
df=pd.read_csv("players.csv")
print(df.head(5))

#Group by team and sort by PER in descending order to get top 5 players per team 
df_sorted = df.sort_values(by=["Team", "PER"], ascending=[True, False])
top5_per_team = df_sorted.groupby("Team").head(5)
print(top5_per_team[["Team", "Player", "PER"]])

#Avergae PER of top 5 players per team 
avg_per_per_team = top5_per_team.groupby("Team")["PER"].mean().reset_index()
avg_per_per_team.columns = ["Team", "Avg_TOP5_PER"]
print(avg_per_per_team)