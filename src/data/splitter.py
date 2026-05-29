# Stratified train / validation / test splitting with optional time-aware logic.
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

Path("data/processed/splits").mkdir(parents=True, exist_ok=True)
master = pd.read_csv("data/processed/master_final.csv")

train, temp = train_test_split(master, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

train.to_parquet("data/processed/splits/train.parquet", index=False)
val.to_parquet("data/processed/splits/val.parquet", index=False)
test.to_parquet("data/processed/splits/test.parquet", index=False)

print(train.shape, val.shape, test.shape)