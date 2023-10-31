import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import geopandas as gpd
import warnings
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")

# Load your dataset into a pandas DataFrame
# Replace 'your_dataset.csv' with the actual file path
data = pd.read_csv('D:/Crop_Yield/crop_yield.csv')

# Load state-level geospatial data from an online source (GeoJSON format)
# Replace 'online_geojson_url' with the actual URL of your GeoJSON data source.
online_geojson_url = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/united-states.geojson'
state_geospatial_data = gpd.read_file(online_geojson_url)

# Merge geospatial data with crop yield data
data = pd.merge(data, state_geospatial_data, left_on='State', right_on='name', how='left')

# Data Preprocessing
data.drop(columns=['geometry'], inplace=True)
data.fillna(0, inplace=True)
data = pd.get_dummies(data, columns=['Crop', 'Season', 'State'])
data['Historical_Yield_Trend'] = np.arange(len(data))

# Data Splitting
X = data.drop(['Yield'], axis=1)
y = data['Yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
}

# Ensemble Method (Stacking)
stacked_model = StackingRegressor(
    estimators=[('rf', models['Random Forest']),
                ('lr', models['Linear Regression']),
                ('dt', models['Decision Tree'])],
    final_estimator=GradientBoostingRegressor(n_estimators=10)
)

models['Stacked Model'] = stacked_model

from keras.layers import Conv1D, MaxPooling1D, Flatten

# Model Training and Evaluation
y_test_actual = y_test  # Actual yield values
y_test_predicted = {}  # Create an empty dictionary to store model predictions

for model_name, model in models.items():
    print(f"Model: {model_name}")

    if 'Neural Network' in model_name:
        # Modify the Neural Network training block as previously mentioned
        X_train_nn = X_train.values
        y_train_nn = y_train.values
        X_test_nn = X_test.values

        # Scale features
        scaler = StandardScaler()
        X_train_nn = scaler.fit_transform(X_train_nn)
        X_test_nn = scaler.transform(X_test_nn)

        # Train the model
        model.fit(X_train_nn, y_train_nn)

        # Make predictions
        y_pred_cv = model.predict(X_test_nn)
    elif 'Recurrent Neural Network' in model_name:
        X_train_rnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
        y_train_rnn = y_train.values
        X_test_rnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Create and compile the RNN model
        rnn_model = Sequential()
        rnn_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_rnn.shape[1], 1)))
        rnn_model.add(LSTM(units=50, return_sequences=False))

        rnn_model.add(Dense(units=25))
        rnn_model.add(Dense(units=1))
        rnn_model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the RNN model
        rnn_model.fit(X_train_rnn, y_train_rnn, epochs=50, batch_size=32)

        # Make predictions
        y_pred_cv = rnn_model.predict(X_test_rnn)
    elif 'Convolutional Neural Network' in model_name:
        X_train_cnn = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
        y_train_cnn = y_train.values
        X_test_cnn = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Create and compile the CNN model
        cnn_model = Sequential()
        cnn_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_cnn.shape[1], 1)))
        cnn_model.add(MaxPooling1D(pool_size=2))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(units=50, activation='relu'))
        cnn_model.add(Dense(units=1))
        cnn_model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the CNN model
        cnn_model.fit(X_train_cnn, y_train_cnn, epochs=50, batch_size=32)

        # Make predictions
        y_pred_cv = cnn_model.predict(X_test_cnn)
    else:
        # For non-neural network models
        y_pred_cv = cross_val_predict(model, X, y, cv=5)  # 5-fold cross-validation

    # Store model predictions
    y_test_predicted[model_name] = y_pred_cv

    # Evaluate the model or calculate appropriate evaluation metrics
    if 'Neural Network' in model_name or 'Recurrent Neural Network' in model_name:
        # If using neural network models, you should evaluate on the test data
        mae_cv = mean_absolute_error(y_test, y_pred_cv)
        mse_cv = mean_squared_error(y_test, y_pred_cv)
        rmse_cv = np.sqrt(mse_cv)
        r2_cv = r2_score(y_test, y_pred_cv)
    else:
        # For non-neural network models, you can use the entire dataset
        mae_cv = mean_absolute_error(y, y_pred_cv)
        mse_cv = mean_squared_error(y, y_pred_cv)
        rmse_cv = np.sqrt(mse_cv)
        r2_cv = r2_score(y, y_pred_cv)

    print(f'Cross-Validation Mean Absolute Error: {mae_cv}')
    print(f'Cross-Validation Mean Squared Error: {mse_cv}')
    print(f'Cross-Validation Root Mean Squared Error: {rmse_cv}')
    print(f'Cross-Validation R-squared: {r2_cv}')

    print('-' * 40)

# Create a DataFrame for time series analysis
time_series_data = X_test.copy()
time_series_data['Actual_Yield'] = y_test_actual

# Plot time series data for actual and predicted yields

plt.figure(figsize=(12, 6))
plt.plot(time_series_data['Crop_Year'], time_series_data['Actual_Yield'], label='Actual Yield', marker='o')
for model_name in y_test_predicted:
    y_pred = y_test_predicted[model_name]
    if len(y_pred) == len(time_series_data['Crop_Year']):
        plt.plot(time_series_data['Crop_Year'], y_pred, label=f'Predicted Yield - {model_name}', linestyle='--', marker='x')
plt.xlabel('Crop Year')
plt.ylabel('Yield')
plt.title('Actual vs. Predicted Crop Yield Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Analyze trends using statistical methods, e.g., trend decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

# Assuming 'Crop_Year' is a numeric column representing years, convert it to a datetime format.
time_series_data['Crop_Year'] = pd.to_datetime(time_series_data['Crop_Year'], format='%Y')

# Group by 'Crop_Year' and aggregate the data (e.g., using the mean of 'Actual_Yield' values).
time_series_data = time_series_data.groupby('Crop_Year').agg({'Actual_Yield': 'mean'})

# Fill missing values in 'Actual_Yield' column with a specific value (e.g., 0).
time_series_data['Actual_Yield'].fillna(0, inplace=True)

# Set the 'Crop_Year' column as the index of the DataFrame and specify the frequency as 'A' (for annual data).
time_series_data = time_series_data.asfreq('A')

# Perform seasonal decomposition
# Decompose the actual yield time series data
decomposition_actual = seasonal_decompose(time_series_data['Actual_Yield'], model='additive')
# Decompose the predicted yield for a specific model (replace 'Your_Model_Name' with the actual model name)
# decomposition_predicted = seasonal_decompose(time_series_data['Predicted_Yield_Your_Model_Name'], model='additive')

# Plot decomposed time series components
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(decomposition_actual.trend, label='Actual Yield Trend', color='blue')
plt.plot(decomposition_actual.seasonal, label='Actual Yield Seasonality', color='red')
plt.plot(decomposition_actual.resid, label='Actual Yield Residuals', color='green')
plt.title('Decomposition of Actual Yield')
plt.legend()

# Uncomment this section if you want to perform decomposition for the 'Predicted_Yield_Neural Network'
decomposition_predicted = seasonal_decompose(time_series_data['Predicted_Yield_Neural Network'], model='additive')

# Plot decomposed time series components
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(decomposition_actual.trend, label='Actual Yield Trend', color='blue')
plt.plot(decomposition_actual.seasonal, label='Actual Yield Seasonality', color='red')
plt.plot(decomposition_actual.resid, label='Actual Yield Residuals', color='green')
plt.title('Decomposition of Actual Yield')
plt.legend()

# Uncomment this section if you have predicted yield for the 'Neural Network'
plt.subplot(2, 1, 2)
plt.plot(decomposition_predicted.trend, label='Predicted Yield Trend - Neural Network', color='blue')
plt.plot(decomposition_predicted.seasonal, label='Predicted Yield Seasonality - Neural Network', color='red')
plt.plot(decomposition_predicted.resid, label='Predicted Yield Residuals - Neural Network', color='green')
plt.title('Decomposition of Predicted Yield - Neural Network')
plt.legend()

plt.tight_layout()
plt.show()

