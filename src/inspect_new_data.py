import pandas as pd

# Load the Excel file
df = pd.read_excel('dataset/The extended Algerian forest fires dataset.xlsx', sheet_name='DONAT_data_set')
print("--- Shape ---")
print(df.shape)

print("\n--- Columns ---")
print(df.columns.tolist())

print("\n--- Classes Value Counts ---")
print(df['Classes'].value_counts())

print("\n--- Region Column Value Counts ---")
print(df.iloc[:, -1].value_counts())

print("\n--- Date Column Sample ---")
print(df['DATE'].head(10).tolist())