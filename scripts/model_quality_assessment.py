import json
import pandas as pd
from tensorflow import keras

# преобразование файлов .csv в датафреймы
df_X_test = pd.read_csv('data/prepared/X_test_prepared.csv')
df_Y_test = pd.read_csv('data/prepared/Y_test_prepared.csv')

# загрузка обученной модели
model = keras.models.load_model('mnist_simple_model')

# оценка качества модели
scores = model.evaluate(df_X_test, df_Y_test, verbose=1)

# сохранение результатов в json
results = {'test_loss':scores[0], 'test_accuracy':scores[1]}
with open('results.json', 'w') as file:
	json.dump(results, file)
