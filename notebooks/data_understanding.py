import pandas as pd
import os

# Current directory
current_dir = os.path.dirname(__file__)

# Dataset path
dataset_path = os.path.join(
    current_dir,
    "..",
    "dataset",
    "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# Load dataset
df = pd.read_csv(dataset_path)

print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(df.head())

print("\n")

print("=" * 60)
print("LAST 5 ROWS")
print("=" * 60)
print(df.tail())

print("\n")

print("=" * 60)
print("SHAPE OF DATASET")
print("=" * 60)
print(df.shape)

print("\n")

print("=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(df.columns.tolist())

print("\n")

print("=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

print("\n")

print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\n")

print("=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)
print(df.duplicated().sum())

print("\n")

print("=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())