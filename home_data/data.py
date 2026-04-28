import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# нормализуем данные, чтобы они были в одном масштабе, это поможет в обучении модели
def normalize_and_impute_data(data_train: pd.DataFrame, data_test: pd.DataFrame) -> pd.DataFrame:

    # заполняем пропущенные значения средним по кол-ву встречаемых значений в столбце, так как это может быть более информативно, чем просто среднее по столбцу
    data_train = data_train.fillna(data_train.mode().iloc[0]) # заполняем пропущенные значения средним по столбцу
    data_test = data_test.fillna(data_train.mode().iloc[0]) # заполняем пропущенные значения средним по столбцу
    for columns in data_train.select_dtypes(include=["object", "string"]).columns:
        le = LabelEncoder()
        le.fit(data_train[columns].astype(str))
        data_train[columns] = le.transform(data_train[columns].astype(str))
        data_test[columns] = le.transform(data_test[columns].astype(str))

    mean = data_train.mean()
    std = data_train.std().replace(0, 1) # заменим нули на единицы, чтобы избежать деления на ноль
    data_train = (data_train - mean) / std # нормализуем данные, чтобы они были в одном масштабе, это поможет в обучении модели
    data_test = (data_test - mean) / std # нормализуем данные, чтобы они были в одном масштабе, это поможет в обучении модели
    return data_train, data_test

train_data = pd.read_csv("home_data\\data\\train.csv")
train_targets = train_data.iloc[:, -1]
train_data.drop(columns=train_data.columns[-1], inplace=True) # удалим столбец с ценой продажи дома
test_data = pd.read_csv("home_data\\data\\test.csv")
test_targets = pd.read_csv("home_data\\data\\sample_submission.csv").iloc[:, -1]

train_targets = np.log1p(train_targets)
test_targets = np.log1p(test_targets)

train_data, test_data = normalize_and_impute_data(train_data, test_data)
train_data = train_data.drop(columns=['Id']) # удалим столбец с идентификатором, так как он не несет полезной информации для модели
test_data = test_data.drop(columns=['Id']) # удалим столбец с идентифик



# посчитаем стандартное отклонение для каждого столбца, чтобы найти колонки с почти постоянными значениями, которые могут мешать обучению модели

# stds = train_data.std()

# # колонки с почти постоянными значениями
# low_std_cols = stds[stds < 1e-3].index
# print(low_std_cols)

# # удалить их
# train_data.drop(columns=low_std_cols, inplace=True)
# test_data.drop(columns=low_std_cols, inplace=True)