from pathlib import Path
import pandas as pd

data_dir = Path("/Users/alfredyuan/Dropbox/GS_Network/Data/final_estimation_data/demand")

parquet_file = data_dir / "df.parquet"
dta_file = parquet_file.with_suffix(".dta")

df = pd.read_parquet(parquet_file)
df.to_stata(dta_file, write_index=False, version=118)

print(f"Saved {dta_file}")
