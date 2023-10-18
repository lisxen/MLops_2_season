import yaml
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# гиперпараметры модели
hypers = yaml.safe_load(open('params.yaml'))

# преобразование файлов .csv в датафреймы
df_X_train = pd.read_csv('data/prepared/X_train_prepared.csv')
df_Y_train = pd.read_csv('data/prepared/Y_train_prepared.csv')

# создание последовательной модели
model = Sequential()

# добавление уровней сети
model.add(Dense(800, input_dim=784, activation='relu', kernel_initializer='normal'))
model.add(Dense(10, activation='softmax', kernel_initializer='normal'))

# информация о модели
model.summary()

#компиляция модели
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

# обучение модели
model.fit(df_X_train, df_Y_train,
          batch_size=hypers['batch_size'],
          epochs=hypers['epochs'],
          validation_split=hypers['validation_split'],
          verbose=hypers['verbose'])

# сохранение модели
model.save('mnist_simple_model')
