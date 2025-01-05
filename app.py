from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Разрешаем CORS для всех маршрутов

# Загрузка моделей
model = joblib.load('crime_model.pkl')
scaler = joblib.load('scaler.pkl')

selected_features = ['assaultPerPop', 'rapes', 'robbbPerPop', 'rapesPerPop', 
                            'PctKidsBornNeverMar', 'murdPerPop', 'NumKidsBornNeverMar',
                            'OwnOccMedVal', 'MalePctNevMarr', 'HousVacant', 'racePctHisp',
                            'PctFam2Par', 'PctPersOwnOccup', 'larcPerPop']

@app.route('/predict', methods=['POST'])
def predict():
    # Получаем данные в формате JSON
    data = request.get_json()

    # Проверяем наличие всех необходимых признаков
    missing_features = [feature for feature in selected_features if feature not in data]
    if missing_features:
        return jsonify({'error': f'Отсутствуют следующие признаки: {missing_features}'}), 400

    # Преобразуем данные в DataFrame
    input_data = pd.DataFrame([data], columns=selected_features)

    # Преобразуем типы данных в float
    input_data = input_data.astype(float)

    # Масштабируем данные
    input_data_scaled = scaler.transform(input_data)

    # Предсказание
    prediction = model.predict(input_data_scaled)

    # Обратное преобразование (если применяли np.sqrt к целевой переменной)
    prediction = prediction ** 2

    # Возвращаем результат
    return jsonify({'prediction': prediction[0]})


if __name__ == '__main__':
    app.run(debug=True)
