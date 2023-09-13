from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


from websocket import create_connection
import websocket
import json
import time
import requests
import math

session = requests.session()
session.proxies = {
    'http': 'http://127.0.0.1:10900',
    'https': 'http://127.0.0.1:10900',
}
session.auth = ("", "")
ws_endpoint = "wss://api.hitbtc.com/api/3/ws/public"




def on_message(ws, message):
    data = json.loads(message)
    ticker_data = data["data"]["XRPUSDT_PERP"]

    # Remove the first and last keys from the dictionary
    del ticker_data[first_key]
    del ticker_data[last_key]

    input_df = pd.DataFrame([ticker_data])
    
    # Specify the numerical columns for feature scaling
    numerical_cols = [ 'a', 'A', 'b', 'B', 'o', 'c', 'h', 'l', 'v', 'q', 'p', 'P']
    
    # Normalize numerical features using MinMaxScaler
    scaler = MinMaxScaler()
    input_df[numerical_cols] = scaler.fit_transform(input_df[numerical_cols])
    
    # Make predictions using the loaded model
    prediction = model.predict(input_df)

    # Prepare the order action based on the predictioin result        
    print("the prediction for the result is:", prediction)

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, close_status_code, close_msg):
    print("Closed:", close_status_code, close_msg)

def on_open(ws):
    print("Subscribe to the XRPUSDT ticker data")
    ws.send('{"method": "subscribe", "ch": "ticker/3s", "params": {"symbols": ["XRPUSDT_PERP"]}}, "id": 13579")')

def run_websocket():
    try:
        while True:
            try:
                ws = websocket.WebSocketApp(ws_endpoint,
                                            on_message=on_message,
                                            on_error=on_error,
                                            on_close=on_close,
                                            on_open=on_open)
                ws.run_forever(http_proxy_host="127.0.0.1", http_proxy_port="10900", proxy_type="http")

            except Exception as e:
                print(f"WebSocket connection error: {e}")
                print("Attempting to reconnect in 5 seconds...")
                time.sleep(5)
    except KeyboardInterrupt:
        print("WebSocket connection terminated due to Ctrl+C.")
        sys.exit(0)

    print("WebSocket connection will not attempt to reconnect after one day.")


def make_prediction(ticker_data, model):
    # Remove the first and last keys from the dictionary
    del ticker_data['t']
    del ticker_data['L']

    # Create a DataFrame directly from the incoming data
    input_df = pd.DataFrame([ticker_data])

    # Specify the numerical columns for feature scaling
    numerical_cols = ['a', 'A', 'b', 'B', 'o', 'c', 'h', 'l', 'v', 'q', 'p', 'P']

    # Normalize numerical features using MinMaxScaler
    scaler = MinMaxScaler()
    input_df[numerical_cols] = scaler.fit_transform(input_df[numerical_cols])

    # Make predictions using the loaded model
    prediction = model.predict(input_df)

    return prediction


if __name__ == "__main__":
    model = tf.keras.models.load_model('my_trained_model.h5')  # Create an instance of the TradingAlgorithm class
    # run_websocket()
 
    # Single shot test for the model
    ticker_data = {'t': 1694588912203, 'a': '0.4755', 'A': '84843', 'b': '0.4750', 'B': '213336', 'o': '0.4726', 'c': '0.4752', 'h': '0.4861', 'l': '0.4701', 'v': '1784130', 'q': '850966.9709', 'p': '0.0026', 'P': '0.5501481168006771', 'L': 748659284}
    print("the model indicate that", make_prediction(ticker_data, model))
    

    # ticker_data = {'t': 1694588912203, 'a': '0.4755', 'A': '84843', 'b': '0.4750', 'B': '213336', 'o': '0.4726', 'c': '0.4752', 'h': '0.4861', 'l': '0.4701', 'v': '1784130', 'q': '850966.9709', 'p': '0.0026', 'P': '0.5501481168006771', 'L': 748659284}

    # # Remove the first and last keys from the dictionary
    # del ticker_data['t']
    # del ticker_data['L']


    # input_df = pd.DataFrame([ticker_data])
    
    # # Specify the numerical columns for feature scaling
    # numerical_cols = [ 'a', 'A', 'b', 'B', 'o', 'c', 'h', 'l', 'v', 'q', 'p', 'P']

    
    # # Normalize numerical features using MinMaxScaler
    # scaler = MinMaxScaler()
    # input_df[numerical_cols] = scaler.fit_transform(input_df[numerical_cols])
    
    # # Make predictions using the loaded model
    # prediction = model.predict(input_df)
        

    # print("the prediction for the result is:", prediction)