import pandas as pd


df = pd.read_csv("data/cat_breeds_clean.csv", sep=";")


print("Исходные столбцы:")
print(df.columns.tolist())


if 'Latitude' in df.columns and 'Longitude' in df.columns:
    df = df.drop(['Latitude', 'Longitude'], axis=1)
    print("\nСтолбцы после удаления:")
    print(df.columns.tolist())
    

    df.to_csv("data/cat_breeds_clean.csv", sep=";", index=False)
    print("\nФайл успешно обновлен!")
else:
    print("\nСтолбцы Latitude и Longitude не найдены в датасете.") 