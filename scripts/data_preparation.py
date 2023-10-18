import pandas as pd
import tensorflow as tf

# преобразование файлов csv в датафреймы
df_X_train = pd.read_csv('data/raw/X_train.csv')
df_X_test = pd.read_csv('data/raw/X_test.csv')
df_Y_train = pd.read_csv('data/raw/Y_train.csv')
df_Y_test = pd.read_csv('data/raw/Y_test.csv')

# преобразование pandas.dataframe в numpy.ndarray
X_train = df_X_train.to_numpy()
X_test = df_X_test.to_numpy()
Y_train = df_Y_train.to_numpy()
Y_test = df_Y_test.to_numpy()

# нормализация данных
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# преобразование меток в формат one hot encoding
Y_train = tf.keras.utils.to_categorical(Y_train, 10)
Y_test = tf.keras.utils.to_categorical(Y_test, 10)

# преобразование numpy.ndarray в pandas.dataframe
df_X_train = pd.DataFrame(X_train)
df_X_test = pd.DataFrame(X_test)
df_Y_train = pd.DataFrame(Y_train)
df_Y_test = pd.DataFrame(Y_test)

# запись датафреймов в формате csv
df_X_train.to_csv('data/prepared/X_train_prepared.csv', index=False)
df_X_test.to_csv('data/prepared/X_test_prepared.csv', index=False)
df_Y_train.to_csv('data/prepared/Y_train_prepared.csv', index=False)
df_Y_test.to_csv('data/prepared/Y_test_prepared.csv', index=False)
