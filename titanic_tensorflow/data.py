import os
import csv
from io import StringIO
import numpy as np
from sklearn.impute import SimpleImputer

# пока не добавляю энкодинг категориальных признаков, 
# просто удаляю их, если результаты не будут хорошими, то можно будет добавить энкодинг

# преобразуем csv в numpy массив, удаляя ненужные столбцы
# def csv_to_data(pth: str, column_to_deletes: list = [], labelencode_columns: list = []) -> np.ndarray:
#     fname = os.path.join(pth)
#     with open(fname) as f:
#         data = f.read()
#     lines = data.split("\n")
#     header = lines[0].split(",")
#     lines = lines[1:]
#     np_tensor = np.zeros(shape=(len(lines), len(header) - len(column_to_deletes)))
#     #print(np_tensor.shape)
#     for i, line in enumerate(lines):
#         if line == "":
#             print("empty line, skipping")
#             continue
#         strio = StringIO()
#         reader = csv.reader(strio)
#         header = next(reader)
#         values = next(reader)
#         print(values, len(values))
#         cnt = 0
#         for j, value in enumerate(values):
#             print(f"processing value: {value} at row {i}, column {j}")
#             if j in column_to_deletes:
#                 continue
#             if value == "":
#                 value = None
#                 np_tensor[i][cnt] = value
#                 cnt+=1
#                 continue
#             if j in labelencode_columns:
#                 print(f"label encoding value: {value} at row {i}, column {j}")
#                 value = value.strip() # удаляем пробелы, чтобы labelencoder не выдавал ошибку из-за них
#                 continue
#             else:
#                 value = float(value)
#             print(f"assigning value: {value} to np_tensor at row {i}, column {cnt}")
#             np_tensor[i][cnt] = value
#             cnt += 1

#     # передадим в labelencoder только те столбцы, которые нужно закодировать, а не все подряд
#     for column in labelencode_columns:
#         le = LabelEncoder()
#         np_tensor[:, column] = le.fit_transform(np_tensor[:, column])
#     return np_tensor

lable_encode_dict = {"male": 0, "female": 1}

def csv_to_data(pth: str, column_to_deletes: list = [], labelencode_columns: list = []) -> np.ndarray:
    fname = os.path.join(pth)
    
    with open(fname, newline='') as f:
        reader = csv.reader(f)
        
        # Чтение заголовка
        header = next(reader)
        
        # Определение количества строк и столбцов
        num_columns = len(header) - len(column_to_deletes)
        np_tensor = np.zeros(shape=(0, num_columns))
        
        # Чтение данных
        for i, row in enumerate(reader):
            #print(f"processing row {i}: {row}")
            if not row:  # Пропустить пустые строки
                print("empty line, skipping")
                continue

            # Преобразование строки в numpy массив
            np_row = []
            cnt = 0

            for j, value in enumerate(row):
                #print(f"processing value: {value} at row {i}, column {j}")
                
                if j in column_to_deletes:  # Пропускаем столбцы для удаления
                    continue
                

                if value == "":  # Если значение пустое
                    value = np.nan # Используем NaN для пропущенных значений
                    np_row.append(value)
                    cnt += 1
                    continue
                
                if j in labelencode_columns:  # Если столбец должен быть закодирован
                    #print(f"label encoding value: {value} at row {i}, column {j}")
                    value = value.strip()  # Убираем пробелы
                    np_row.append(lable_encode_dict[value]) # Преобразуем в числовое значение
                else:
                    try:
                        value = float(value)  # Преобразуем в число
                        np_row.append(value)
                    except ValueError:
                        print(f"Non-numeric value {value} in row {i}, column {j}, skipping.")
            
            # Добавляем строку в numpy массив
            #print(f"assigning row {i} to np_tensor: {np_row}")
            np_tensor = np.vstack([np_tensor, np.array(np_row)])
    
    # Применяем LabelEncoder к нужным столбцам
    # for column in labelencode_columns:
    #     le = LabelEncoder()
    #     np_tensor[:, column] = le.fit_transform(np_tensor[:, column])

    return np_tensor

# нормализуем данные, чтобы они были в одном масштабе, это поможет в обучении модели
def normalize_and_impute_data(data: np.ndarray) -> np.ndarray:
    imputer = SimpleImputer(strategy='most_frequent')
    data = imputer.fit_transform(data)
    #print(data)
    for i in range(data.shape[1]):
        column = data[:, i]

        if np.issubdtype(column.dtype, np.number):
            # Если есть NaN (они могут остаться после SimpleImputer), заменим их на среднее
            if np.any(np.isnan(column)):
                nan_mean = np.nanmean(column)  # Используем nanmean, чтобы игнорировать NaN в расчете
                column = np.nan_to_num(column, nan=nan_mean)
        
        mean = np.mean(column)
        data[:, i] = (column - mean)
        std = np.std(column)
        if std == 0:
            std = 1e-8
        data[:, i] = column / std
    return data

train_data = csv_to_data(pth="titanic_tensorflow\\data\\train.csv", column_to_deletes=[0, 3, 8, 10, 11], labelencode_columns=[4])
train_targets = train_data[:, 0] # выделим целевую переменную, она находится в первом столбце, так как мы удалили первый столбец с id
train_data = np.delete(train_data, 0, axis=1) # удалим столбец с целевой переменной из обучающих данных
train_data = normalize_and_impute_data(train_data)

print("train targets shape:", train_targets.shape)

print("train data shape:", train_data.shape)

print("train data sample:", train_data[:5], "\ntrain targets sample:", train_targets[:5])
test_data = csv_to_data(pth="titanic_tensorflow\\data\\test.csv", column_to_deletes=[0, 2, 7, 9, 10], labelencode_columns=[3])
test_data = normalize_and_impute_data(test_data)


print("test data shape:", test_data.shape)

test_targets = csv_to_data(pth="titanic_tensorflow\\data\\gender_submission.csv", column_to_deletes=[0])

if test_targets.shape != (test_data.shape[0],):
    test_targets = test_targets.flatten()  # Преобразуем в одномерный массив, если это необходимо
print("test targets shape:", test_targets.shape)