import pandas as pd

df = pd.read_csv("data/raw/home_credit/application_train.csv")

print(df.shape)
print(df.head())
print(df.info())
print(df["TARGET"].value_counts(normalize=True))
print(df.isnull().mean().sort_values(ascending=False).head(30))