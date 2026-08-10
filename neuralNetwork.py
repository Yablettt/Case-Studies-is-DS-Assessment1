import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
RANDOM_STATE = 50

#load data
cols = [
    "gameDate", "numMinutes", "points", "assists", "reboundsTotal", "fieldGoalsAttempted", "threePointersAttempted",
    "freeThrowsAttempted", "turnovers", "steals", "blocks", "home", "win", "playerteamName",]
df = pd.read_csv("csv/PlayerStatistics.csv", usecols=cols, low_memory=False)

print("Shape: ", df.shape)

#filter for 2003-2022 so both models are similar
df["gameDate"] = pd.to_datetime(df["gameDate"])
df["numMinutes"] = pd.to_numeric(df["numMinutes"], errors="coerce")

df = df[(df["gameDate"] >= "2003-01-01") & (df["gameDate"] <= "2022-12-31")]
print("Shape after filtering 2003-2022: ", df.shape)