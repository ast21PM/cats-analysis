import pandas as pd

# Читаем исходный файл
df = pd.read_csv("data/cat_breeds_clean.csv", sep=";")

# Выводим информацию о столбцах
print("Все столбцы в датасете:")
print(df.columns.tolist())

# Проверяем дубликаты столбцов
duplicates = df.columns[df.columns.duplicated()].tolist()
print("\nДублирующиеся столбцы:")
print(duplicates)

# Удаляем дублирующиеся столбцы
df = df.loc[:, ~df.columns.duplicated()]

# Проверяем оставшиеся столбцы
print("\nСтолбцы после удаления дубликатов:")
print(df.columns.tolist())

# Сохраняем очищенный датасет
df.to_csv("data/cat_breeds_clean.csv", sep=";", index=False)
print("\nФайл успешно обновлен!") 