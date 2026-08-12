# NBA Case Study — Model Scripts
 
## Requirements
- Created on Python 3.11
- install pandas numpy scikit-learn
 
## Files
- `neural_network.py` — MLPRegressor, predicts player `points` (PlayerStatistics.csv)
- `logistic_regression.py` — Logistic Regression, predicts `HOME_TEAM_WINS` (games.csv)
 
## Setup
Place `games.csv` and `PlayerStatistics.csv` in the same folder, a folder named "csv".
`PlayerStatistics.csv` (approx 390MB) is not included in github repo because file size is too big. Can be downloaded from kaggle - https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores?select=PlayerStatistics.csv 
 
## Run
`
python3 neural_network.py
python3 logistic_regression.py
`
 
Each script prints evaluation metrics that include league wide results, and lakers specific breakdowns.