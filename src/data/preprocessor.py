
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
from pathlib import Path

url = "https://raw.githubusercontent.com/Wichkey/final-project-EADA/main/data/processed/master.csv"
master = pd.read_csv(url)

# Labels

def mule_1():
    labels = master[["cuenta"]].copy()
    labels["is_mule"] = 1
    labels.to_csv("data/raw/labels.csv", index=False)
    return labels

def merging(labels):
    master_labelled = master.merge(labels, on="cuenta", how="left")
    master_labelled.to_csv("data/processed/master_labelled.csv", index=False)

labels = mule_1()
merging(labels)

# Imputation 

print(list(Path("data/processed").glob("*")))
master_l = pd.read_csv("data/processed/master_labelled.csv")
print(master_l.dtypes)

def select_columns(df):
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include="object").columns
    boolean_cols = df.select_dtypes(include="bool").columns
    return numeric_cols, categorical_cols, boolean_cols

def fill_numeric(df, numeric_cols):
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df

def fill_categorical(df, categorical_cols):
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")
    return df

def fill_bool(df, boolean_cols):
    for col in boolean_cols:
        df[col] = df[col].fillna(0)
    return df

numeric_cols, categorical_cols, boolean_cols = select_columns(master_l)
master_l = fill_numeric(master_l, numeric_cols)
master_l = fill_categorical(master_l, categorical_cols)
master_l = fill_bool(master_l, boolean_cols)

master_l.to_csv("data/processed/master_imputed.csv", index=False)

print(f" The final nulls are:", master_l.isnull().sum().sum())

#Encoding

master_e = pd.read_csv("data/processed/master_imputed.csv")

def encode(df):
    df = df.drop(columns=["cuenta", "dato_persona_titular"])
    low = [col for col in df.select_dtypes(include="object").columns if df[col].nunique() < 20]
    high = [col for col in df.select_dtypes(include="object").columns if df[col].nunique() >= 20]
    df = pd.get_dummies(df, columns=low)
    for col in high:
        df[col] = df[col].map(df[col].value_counts() / len(df))
    return df
    
master = pd.read_csv("data/processed/master_imputed.csv")
master_encoded = encode(master)
master_encoded.to_csv("data/processed/master_encoded.csv", index=False)
print(master_encoded.shape)

#Scaling

#Robuts scaler for amount or money features

robust_cols = ["total_credit", "total_debit", "gross_volume", "mean_abs_amount", "max_abs_amount", "net_flow", "balance_2025_12_31"]

def scale(master):
    robust_cols = ["total_credit", "total_debit", "gross_volume", "mean_abs_amount", "max_abs_amount", "net_flow", "balance_2025_12_31"]
    standard_cols = [col for col in master.select_dtypes(include="number").columns if col not in robust_cols]
    master[robust_cols] = RobustScaler().fit_transform(master[robust_cols])
    master[standard_cols] = StandardScaler().fit_transform(master[standard_cols])
    return master


master.to_csv("data/processed/master_final.csv", index=False)

#Reusable transformer for when we have the final csv


    