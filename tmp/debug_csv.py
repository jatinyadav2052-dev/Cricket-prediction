import pandas as pd
df = pd.read_csv(r'c:\Users\HP\OneDrive\Desktop\project\Project_Cricket_first\data\matches.csv')
print("Columns:")
for col in df.columns:
    print(f"- {col}")
print("\nUnique values in 'winner':", df['winner'].nunique())
print("Value counts for 'winner':")
print(df['winner'].value_counts().head(10))
print("\nMissing values:")
print(df.isnull().sum())
