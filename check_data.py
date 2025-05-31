import pandas as pd


df = pd.read_csv("data/cat_breeds_clean.csv", sep=";")


print("Все столбцы в датасете:")
print(df.columns.tolist())


duplicates = df.columns[df.columns.duplicated()].tolist()
print("\nДублирующиеся столбцы:")
print(duplicates)


df = df.loc[:, ~df.columns.duplicated()]

print("\nСтолбцы после удаления дубликатов:")
print(df.columns.tolist())


df.to_csv("data/cat_breeds_clean.csv", sep=";", index=False)
print("\nФайл успешно обновлен!") 