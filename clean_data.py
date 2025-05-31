import pandas as pd

# Читаем исходный файл
df = pd.read_csv("data/cat_breeds_clean.csv", sep=";")

# Выводим информацию о столбцах
print("Исходные столбцы:")
print(df.columns.tolist())

# Удаляем столбцы Latitude и Longitude
if 'Latitude' in df.columns and 'Longitude' in df.columns:
    df = df.drop(['Latitude', 'Longitude'], axis=1)
    print("\nСтолбцы после удаления:")
    print(df.columns.tolist())
    
    # Сохраняем обновленный файл
    df.to_csv("data/cat_breeds_clean.csv", sep=";", index=False)
    print("\nФайл успешно обновлен!")
else:
    print("\nСтолбцы Latitude и Longitude не найдены в датасете.") 