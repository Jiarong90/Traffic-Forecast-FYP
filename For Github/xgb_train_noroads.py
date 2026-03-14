import numpy as np
import pandas as pd
import duckdb
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score


FEAT = r""

# Train model with no road inputs. To test how well a naive baseline model works.

con = duckdb.connect()

min_ts, max_ts = con.execute(f"""
    SELECT MIN(snapshot_ts), MAX(snapshot_ts)
    FROM read_parquet('{FEAT}')                          
""").fetchone()

split_ts = max_ts - 2 * 86400

# Take only a fraction of data, as data size is too large to be loaded into memory..
TRAIN_SAMPLE_FRAC = 0.1
TEST_SAMPLE_FRAC = 0.1

train_df = con.execute(f"""
    SELECT sb, sb_tm5, sb_tm10, sb_tm15, hour, dow, y_tp30, snapshot_ts
    FROM read_parquet('{FEAT}')
    WHERE snapshot_ts < {split_ts}
    AND sb BETWEEN 1 AND 8
        AND sb_tm5 BETWEEN 1 AND 8
        AND sb_tm10 BETWEEN 1 AND 8
        AND sb_tm15 BETWEEN 1 AND 8
        AND y_tp30 BETWEEN 1 AND 8
    USING SAMPLE {int(TRAIN_SAMPLE_FRAC * 100)} PERCENT                 
""").df()

test_df = con.execute(f"""
    SELECT sb, sb_tm5, sb_tm10, sb_tm15, hour, dow, y_tp30, snapshot_ts
    FROM read_parquet('{FEAT}')
    WHERE snapshot_ts >= {split_ts}
        AND sb BETWEEN 1 AND 8
        AND sb_tm5 BETWEEN 1 AND 8
        AND sb_tm10 BETWEEN 1 AND 8
        AND sb_tm15 BETWEEN 1 AND 8
        AND y_tp30 BETWEEN 1 AND 8
""").df()

X_train = train_df[["sb", "sb_tm5", "sb_tm10", "sb_tm15"]]

X_test = test_df[["sb", "sb_tm5", "sb_tm10", "sb_tm15"]]

y_train = train_df["y_tp30"].astype(int) - 1
y_test = test_df["y_tp30"].astype(int) - 1
num_class = 8

# Use XGBoost. To do: to optimize more to maximize GPU benefit..
model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=8,
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,           
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",      
    device="cuda",          
    n_jobs=-1,
    random_state=42,
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

# Check accuracy, mae and within1
# To measure how far off are the model's predictions
acc = accuracy_score(y_test, pred)
mae = np.mean(np.abs(pred - y_test))
within1 = np.mean(np.abs(pred - y_test) <= 1)

base_pred = test_df["sb"].astype(int).to_numpy() - 1
base_acc = np.mean(base_pred == y_test.to_numpy())
base_mae = np.mean(np.abs(base_pred - y_test.to_numpy()))
base_within1 = np.mean(np.abs(base_pred - y_test.to_numpy()) <= 1)

# Check within speedbands that actually changed speed, how many of such speedbands
# were identified by the model (typically 0 for naive models)
change_mask = (base_pred != y_test.to_numpy())
if change_mask.any():
    acc_change = np.mean(pred[change_mask] == y_test.to_numpy()[change_mask])
    base_acc_change = np.mean(base_pred[change_mask] == y_test.to_numpy()[change_mask])
else:
    acc_change = float("nan")
    base_acc_change = float("nan")

print("Results:")
print(f"Model acc = {acc:.4f} within1 = {within1:.4f} MAE = {mae:.4f}")
print(f"Persistence acc={base_acc:.4f} within1={base_within1:.4f} MAE={base_mae:.4f}")
print(f"Change-rows accuracy: model={acc_change:.4f} | persistence={base_acc_change:.4f}")
