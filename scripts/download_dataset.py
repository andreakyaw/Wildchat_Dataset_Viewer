from datasets import load_dataset
import os

print("Downloading WildChat-1M dataset...")
dataset = load_dataset("allenai/WildChat-1M")

print("Saving as parquet chunks...")
os.makedirs("data/parquet", exist_ok=True)

chunk_size = 100_000
data = dataset["train"]

for i in range(0, len(data), chunk_size):
    print(f"Processing chunk {i}...")
    chunk = data.select(range(i, min(i + chunk_size, len(data))))
    chunk.to_parquet(f"data/parquet/chunk_{i}.parquet")

print("Done!")