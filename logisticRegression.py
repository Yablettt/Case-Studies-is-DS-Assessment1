import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report)

RANDOM_STATE = 50

#load data
df = pd.read_csv("csv/games.csv")

print("Shape: ", df.shape)

print("Nulls: \n",df.isnull().sum())

# drop rows with missing stats
df = df.dropna(subset=["PTS_home", "FG_PCT_home", "FT_PCT_home", "FG3_PCT_home", "AST_home", "REB_home", "PTS_away", "FG_PCT_away", "FT_PCT_away", "FG3_PCT_away", "AST_away", "REB_away",])

print("After removing missing rows: ", df.shape)

# create features
df["FG_PCT_diff"] = df["FG_PCT_home"] - df["FG_PCT_away"]
df["FT_PCT_diff"] = df["FT_PCT_home"] - df["FT_PCT_away"]
df["FG3_PCT_diff"] = df["FG3_PCT_home"] - df["FG3_PCT_away"]
df["AST_diff"] = df["AST_home"] - df["AST_away"]
df["REB_diff"] = df["REB_home"] - df["REB_away"]

feature_cols = ["FG_PCT_diff", "FT_PCT_diff", "FG3_PCT_diff", "AST_diff", "REB_diff"]

X = df[feature_cols]
y = df["HOME_TEAM_WINS"]

# train test split
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(X, y, df, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

# train LR
model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
model.fit(X_train, y_train)
 
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
 
# evaluate predictions for all 30 teams
print("League Evaluation")
print("----------------------------------------")
print("Test set size: ", len(y_test))
print("Accuracy: ", round(accuracy_score(y_test, y_pred), 4))
print("Precision: ", round(precision_score(y_test, y_pred), 4))
print("Recall: ", round(recall_score(y_test, y_pred), 4))
print("F1 Score: ", round(f1_score(y_test, y_pred), 4))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Home Loss", "Home Win"]))
 
print("\nFeature coefficients:")
coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_[0]}).sort_values("coefficient", key=abs, ascending=False)
print(coef_df.to_string(index=False))
 
# evaluate the lakers
lakers_mask_test = (df_test["TEAM_ID_home"] == 1610612747) | (df_test["TEAM_ID_away"] == 1610612747)
lakers_test_idx = df_test[lakers_mask_test].index
 
if len(lakers_test_idx) > 0:
    X_lakers = X_test.loc[lakers_test_idx]
    y_lakers = y_test.loc[lakers_test_idx]
    y_lakers_pred = model.predict(X_lakers)
 
    print("\nLakers Evaluation")
    print("----------------------------------------")
    print("Lakers games in test set: ", len(y_lakers))
    print("Accuracy: ", round(accuracy_score(y_lakers, y_lakers_pred), 4))
    if len(set(y_lakers)) > 1:
        print("Precision: ", round(precision_score(y_lakers, y_lakers_pred), 4))
        print("Recall: ", round(recall_score(y_lakers, y_lakers_pred), 4))
        print("F1 Score: ", round(f1_score(y_lakers, y_lakers_pred), 4))
    print("\nLakers Confusion Matrix:")
    print(confusion_matrix(y_lakers, y_lakers_pred))
else:
    print("\nNo Lakers games in this test split")
 
# if run model on all the lakers games from 2003:
lakers_mask_full = (df["TEAM_ID_home"] == 1610612747) | (df["TEAM_ID_away"] == 1610612747)
df_lakers_full = df[lakers_mask_full]
X_lakers_full = df_lakers_full[feature_cols]
y_lakers_full = df_lakers_full["HOME_TEAM_WINS"]
y_lakers_full_pred = model.predict(X_lakers_full)
 

print("\nLakers Evaluation from 2003")
print("----------------------------------------")
print("Total Lakers games (2003-2022): ", len(y_lakers_full))
print("Accuracy: ", round(accuracy_score(y_lakers_full, y_lakers_full_pred), 4))
print("Precision: ", round(precision_score(y_lakers_full, y_lakers_full_pred), 4))
print("Recall: ", round(recall_score(y_lakers_full, y_lakers_full_pred), 4))
print("F1 Score: ", round(f1_score(y_lakers_full, y_lakers_full_pred), 4))
print("\nConfusion Matrix (Lakers, all seasons):")
print(confusion_matrix(y_lakers_full, y_lakers_full_pred))
 
print("----------------------------------------")