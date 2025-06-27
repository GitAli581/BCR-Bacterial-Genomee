import pandas as pd

input_file = "input.csv"
rows_per_chunk = 100000  # Adjust this depending on your system and file size

for i, chunk in enumerate(pd.read_csv(input_file, chunksize=rows_per_chunk)):
    chunk.to_csv(f"input_part_{i+1}.csv", index=False)
    print(f"✅ Saved input_part_{i+1}.csv")

