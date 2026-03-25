import pandas as pd
from pathlib import Path

#json file location
json_path = Path("data/raw/acn_caltech_sessions_2019_2021.json")

#load JSON
df = pd.read_json(json_path)

#flatten nested structure
df = pd.json_normalize(df.to_dict(orient="records"))

#save as CSV
output_path = Path("data/raw/acn_data.csv")
df.to_csv(output_path, index=False)

print(f"Saved CSV to {output_path}")
print("Shape:", df.shape)
print("Columns:", list(df.columns))