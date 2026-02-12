import pandas as pd

def gold():
    df = pd.read_parquet("../data/transformed_data.parquet")
    print(df.head())