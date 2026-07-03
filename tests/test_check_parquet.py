import pandas as pd

p = "dataset/MMDocIR_Train_Dataset/parquet/ArxivQA_filter.parquet"
df = pd.read_parquet(p)
print(df.columns.tolist())
print(df.dtypes)
row = df.iloc[0]
print(row["file_name"], row["page"], row["domain"])
print(type(row["image"]), len(row["image"]), row["image"][:8])
