from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar los modelos entrenados
try:
    # Cargar el modelo de red neuronal para diamantes
    model = joblib.load('modelo_diamantes.pkl')
    app.logger.debug('Modelo de predicción de precios de diamantes cargado correctamente.')
    
    # Cargar el scaler
    scaler = joblib.load('modelo_scaler.pkl')
    app.logger.debug('Scaler cargado correctamente.')
    
except Exception as e:
    app.logger.error(f'Error al cargar los modelos: {str(e)}')
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Verificar que los modelos estén cargados
        if model is None or scaler is None:
            return jsonify({'error': 'Los modelos no están disponibles'}), 500
        
        # Obtener los datos enviados en el request
        carat = float(request.form['carat'])
        x = float(request.form['x'])
        y = float(request.form['y'])
        clarity = float(request.form['clarity'])
        color = float(request.form['color'])

        # Columnas completas para el scaler (sin price porque no se escala)
        expected_columns = [
            'carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'
        ]
        
        # Características que usamos para la predicción
        features = ['y', 'carat', 'x', 'clarity', 'color']

        # Crear diccionario con todos los datos (completar con valores promedio los que no tenemos)
        data_dict = {
            'carat': carat,
            'cut': 3.0,           # Valor promedio para cut (1-5 scale)
            'color': color,
            'clarity': clarity,
            'depth': 61.5,        # Valor promedio típico para depth
            'table': 57.0,        # Valor promedio típico para table
            'x': x,
            'y': y,
            'z': 0.0              # Completar con 0 ya que no lo tenemos
        }

        # Crear DataFrame con todas las columnas esperadas por el scaler
        data_df = pd.DataFrame([data_dict], columns=expected_columns)
        
        app.logger.debug(f'DataFrame creado: {data_df}')
        
        # Escalar los datos usando el scaler entrenado
        data_scaled = scaler.transform(data_df)
        app.logger.debug(f'Datos escalados: {data_scaled}')
        
        # Crear DataFrame con los datos escalados
        data_scaled_df = pd.DataFrame(data_scaled, columns=expected_columns)
        
        # Seleccionar solo las características que usamos para la predicción
        # Mapear las características a las columnas escaladas
        feature_mapping = {
            'y': 'y',
            'carat': 'carat', 
            'x': 'x',
            'clarity': 'clarity',
            'color': 'color'
        }
        
        X = data_scaled_df[[feature_mapping[f] for f in features]]
        
        app.logger.debug(f'Datos seleccionados para predicción: {X}')

        # Realizar predicción con el modelo de red neuronal
        prediction = model.predict(X)
        
        # La predicción puede ser un array, tomar el primer valor
        precio_diamante = float(prediction[0])
        
        # Redondear a 2 decimales y asegurar que sea positivo
        precio_diamante = round(max(0, precio_diamante), 2)
                
        app.logger.debug(f'Predicción de precio del diamante: ${precio_diamante}')

        # Devolver la predicción como respuesta JSON
        return jsonify({'precio_diamante': precio_diamante})
        
    except ValueError as ve:
        app.logger.error(f'Error de valor en la predicción: {str(ve)}')
        return jsonify({'error': 'Valores de entrada inválidos. Asegúrate de ingresar números válidos.'}), 400
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': f'Error interno del servidor: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado de la aplicación y los modelos"""
    try:
        model_status = "OK" if model is not None else "ERROR"
        scaler_status = "OK" if scaler is not None else "ERROR"
        
        return jsonify({
            'status': 'OK' if model_status == 'OK' and scaler_status == 'OK' else 'ERROR',
            'model_status': model_status,
            'scaler_status': scaler_status,
            'system_type': 'Diamond Price Prediction System'
        })
    except Exception as e:
        return jsonify({'status': 'ERROR', 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
