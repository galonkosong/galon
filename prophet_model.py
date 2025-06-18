from prophet import Prophet
import pandas as pd

def train_prophet(df):
    df_prophet = df.rename(columns={'open_time': 'ds', 'close': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    return model

# Contoh penggunaan:
# model = train_prophet(df)
# future = model.make_future_dataframe(periods=24, freq='h')
# forecast = model.predict(future)
