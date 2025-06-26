from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)

# Configurar el registro de errores
logging.basicConfig(level=logging.DEBUG)

# Cargar modelos y configuración
try:
    model = joblib.load('modelo_diamantes.pkl')
    scaler = joblib.load('modelo_scaler.pkl')
    columnas_entrenamiento = joblib.load('columnas_entrenamiento.pkl')

    app.logger.debug('✅ Modelos y configuración cargados correctamente.')

except Exception as e:
    app.logger.error(f'❌ Error al cargar modelos: {str(e)}')
    model = None
    scaler = None
    columnas_entrenamiento = None

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None or columnas_entrenamiento is None:
            return jsonify({'error': 'Modelos no disponibles'}), 500

        # Obtener datos del formulario
        carat = float(request.form['carat'])
        x = float(request.form['x'])
        y_val = float(request.form['y'])  # y es palabra reservada
        clarity = request.form['clarity']
        color = request.form['color']

        # Crear DataFrame
        data_dict = {
            'carat': carat,
            'x': x,
            'y': y_val,
            'clarity': clarity,
            'color': color
        }

        data_df = pd.DataFrame([data_dict])

        app.logger.debug(f'Datos recibidos:\n{data_df}')

        # One-hot encoding
        data_df = pd.get_dummies(data_df)

        # Alinear con columnas de entrenamiento
        data_df = data_df.reindex(columns=columnas_entrenamiento, fill_value=0)

        app.logger.debug(f'DataFrame alineado:\n{data_df}')

        # Escalar solo columnas numéricas
        numeric_cols = data_df.select_dtypes(include=['float64', 'int64']).columns
        data_df[numeric_cols] = scaler.transform(data_df[numeric_cols])

        # Predicción
        prediction = model.predict(data_df)
        precio_diamante = round(max(0, float(prediction[0])), 2)

        app.logger.debug(f'Predicción generada: ${precio_diamante}')

        return jsonify({'precio_diamante': precio_diamante})

    except ValueError as ve:
        app.logger.error(f'❌ Error de valor: {str(ve)}')
        return jsonify({'error': 'Valores de entrada inválidos. Asegúrate de ingresar datos válidos.'}), 400
    except Exception as e:
        app.logger.error(f'❌ Error interno: {str(e)}')
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        status = {
            'status': 'OK' if model and scaler and columnas_entrenamiento else 'ERROR',
            'model_status': 'OK' if model else 'ERROR',
            'scaler_status': 'OK' if scaler else 'ERROR',
            'columns_status': 'OK' if columnas_entrenamiento else 'ERROR',
            'system_type': 'Diamond Price Prediction System'
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'ERROR', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
