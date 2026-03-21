import pandas as pd

df = pd.read_parquet("data/parquet/chunk_0.parquet")

print("Columns:")
print(df.columns)

print("\nFirst row:")
print(df.iloc[0])