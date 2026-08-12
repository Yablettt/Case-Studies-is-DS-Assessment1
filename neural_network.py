#References
# Mlpregressor (no date) scikit. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html (Accessed: 08 August 2026). 
# 1.17. neural network models (supervised) (no date) scikit. Available at: https://scikit-learn.org/stable/modules/neural_networks_supervised.html#neural-networks-supervised (Accessed: 08 August 2026). 
#22. neural networks with Scikit (no date) 22. Neural Networks with Scikit | Machine Learning. Available at: https://python-course.eu/machine-learning/neural-networks-with-scikit.php (Accessed: 09 August 2026). 


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
RANDOM_STATE = 50

#load data
cols = ["gameDate", "numMinutes", "points", "assists", "reboundsTotal", "fieldGoalsAttempted", "threePointersAttempted", "freeThrowsAttempted", "turnovers", "steals", "blocks", "home", "win", "playerteamName",]
df = pd.read_csv("csv/PlayerStatistics.csv", usecols=cols, low_memory=False)

print("Shape: ", df.shape)

#filter for 2003-2022 so both models are similar
df["gameDate"] = pd.to_datetime(df["gameDate"])
df["numMinutes"] = pd.to_numeric(df["numMinutes"], errors="coerce")

# Reference: Asked Claude to help me with date and time sorting. Prompt - "i need to filter a dataset by date how do i do that if the data in the column is yyyy-mm-dd and ive already converted the column to datetime"
df = df[(df["gameDate"] >= "2003-01-01") & (df["gameDate"] <= "2022-12-31")]
print("Shape after filtering 2003-2022: ", df.shape)

# keep player stats for players that have played over 10 mins
df = df.dropna(subset=["numMinutes"])
df = df[df["numMinutes"] >= 10]
print("Shape after dropping players who played less than 10 minutes of gametime: ", df.shape)

# the feature columns and target
feature_cols = [ "numMinutes", "assists", "reboundsTotal", "fieldGoalsAttempted", "threePointersAttempted", "freeThrowsAttempted", "turnovers", "steals", "blocks", "home"]
X = df[feature_cols]
y = df["points"]

# train/test
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(X, y, df, test_size=0.2, random_state=RANDOM_STATE)

#train the neural network
model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", max_iter=300, early_stopping=True, random_state=RANDOM_STATE)
model.fit(X_train, y_train)
 
y_pred = model.predict(X_test)

#evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


#league
print("\nEvaluation")
print("--------------------------------------------------------------------")
print("Test set size: ", len(y_test))
print("Training iterations run: ", model.n_iter_)
print("MAE: ", round(mae, 4))
print("RMSE: ", round(rmse, 4))
print("R2 Score: ", round(r2, 4))
 
print("\nPredictions vs actual (10 rows):")
comparison = pd.DataFrame({"actual_points": y_test.values[:10], "predicted_points": np.round(y_pred[:10], 1)})
print(comparison.to_string(index=False))

#just lakers players
lakers_test = df_test["playerteamName"] == "Lakers"
lakers_test_idx = df_test[lakers_test].index
 
if len(lakers_test_idx) > 0:
    X_lakers = X_test.loc[lakers_test_idx]
    y_lakers = y_test.loc[lakers_test_idx]
    y_lakers_pred = model.predict(X_lakers)
 
    mae_la = mean_absolute_error(y_lakers, y_lakers_pred)
    rmse_la = np.sqrt(mean_squared_error(y_lakers, y_lakers_pred))
    r2_la = r2_score(y_lakers, y_lakers_pred)
 
    print("\nLakers Evaluation")
    print("--------------------------------------------------------------------")
    print("Lakers player-game rows in test set: ", len(y_lakers))
    print("MAE: ", round(mae_la, 4))
    print("RMSE: ", round(rmse_la, 4))
    print("R2: ", round(r2_la, 4))
else:
    print("\nNo Lakers rows found in this test split.")

#run model on full lakers ds and not just test splits
lakers_full = df["playerteamName"] == "Lakers"
df_lakers_full = df[lakers_full]
X_lakers_full = df_lakers_full[feature_cols]
y_lakers_full = df_lakers_full["points"]
y_lakers_full_pred = model.predict(X_lakers_full)
 
mae_lf = mean_absolute_error(y_lakers_full, y_lakers_full_pred)
rmse_lf = np.sqrt(mean_squared_error(y_lakers_full, y_lakers_full_pred))
r2_lf = r2_score(y_lakers_full, y_lakers_full_pred)

print("\nLakers Evaluation (all seasons)")
print("--------------------------------------------------------------------")
print("Test set size: ", len(y_lakers))
print("MAE: ", round(mae_lf, 4))
print("RMSE: ", round(rmse_lf, 4))
print("R2 Score: ", round(r2_lf, 4))