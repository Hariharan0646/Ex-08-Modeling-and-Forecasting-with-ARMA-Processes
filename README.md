# Ex-08: Modeling and Forecasting with ARMA Processes
## NAME : Hariharan S
## REG NO : 2305001009
## Aim
To simulate an ARMA time series, fit ARMA models, forecast future values, and compare model performance using AIC, MAE, MSE, and RMSE.

## Objectives
To generate an ARMA time series.
To plot the series and its ACF and PACF.
To split data into training and testing sets.
To fit and compare different ARMA models.
To forecast future values.
To calculate MAE, MSE, and RMSE.
To generate an ARMA(2,1) process.
## Program
```python
# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 2: Generate ARMA(1,1) Data
np.random.seed(42)
ar = np.array([1, -0.7])
ma = np.array([1, 0.4])
data = ArmaProcess(ar, ma).generate_sample(200)
series = pd.Series(data)

# Step 3: Plot Time Series
plt.plot(series)
plt.title("ARMA(1,1) Time Series")
plt.show()

# Step 4: Plot ACF and PACF
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plot_acf(series, lags=20, ax=ax[0])
plot_pacf(series, lags=20, method="ywm", ax=ax[1])
plt.show()

# Step 5: Split Data
train = series[:160]
test = series[160:]
print("Training Data:", len(train))
print("Testing Data:", len(test))

# Step 6: Fit ARMA(1,1) Model
model = ARIMA(train, order=(1, 0, 1)).fit()
print(model.summary())

# Step 7: Forecast Future Values
forecast = model.forecast(steps=40)
print("Forecast:")
print(forecast)

# Step 8: Calculate Forecast Errors
print("MAE:", mean_absolute_error(test, forecast))
mse = mean_squared_error(test, forecast)
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))

# Step 9: Plot Actual and Forecast Values
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, forecast, label="Forecast")
plt.title("Actual vs Forecast")
plt.legend()
plt.show()

# Step 10: Compare Different ARMA Models
orders = [(1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1)]

for order in orders:
    fit = ARIMA(train, order=order).fit()
    print("ARMA Order:", order)
    print("AIC:", fit.aic)
    print("BIC:", fit.bic)

# Step 11: Generate ARMA(2,1) Data
ar2 = np.array([1, -0.5, -0.2])
ma2 = np.array([1, 0.4])
data2 = ArmaProcess(ar2, ma2).generate_sample(200)

plt.plot(data2)
plt.title("ARMA(2,1) Time Series")
plt.show()

# Step 12: Fit ARMA(2,1) and Forecast
model2 = ARIMA(data2, order=(2, 0, 1)).fit()
forecast2 = model2.forecast(steps=10)

print("ARMA(2,1) Forecast:")
print(forecast2)
```
## Output
<img width="546" height="435" alt="image" src="https://github.com/user-attachments/assets/58749faf-7b46-4891-90ff-e13455893740" />
<img width="847" height="374" alt="image" src="https://github.com/user-attachments/assets/88485b27-f047-4a29-991c-a9a73feb6730" />
<img width="780" height="452" alt="image" src="https://github.com/user-attachments/assets/f4bc3e21-2048-456b-ab66-3c68f6334dfc" />
<img width="287" height="692" alt="image" src="https://github.com/user-attachments/assets/28fd675c-35b8-40b0-ab17-7e2a5373b884" />
<img width="192" height="62" alt="image" src="https://github.com/user-attachments/assets/b8b3aee1-add6-4696-a154-fa673f56369c" />
<img width="548" height="435" alt="image" src="https://github.com/user-attachments/assets/dbbda771-7c0b-4eeb-bd06-32b7797a45cf" />
<img width="226" height="207" alt="image" src="https://github.com/user-attachments/assets/88ddeb92-2612-4d7a-a8a8-bdef712e6663" />
<img width="546" height="435" alt="image" src="https://github.com/user-attachments/assets/893e820a-8856-4854-96b4-2a787aa15906" />
<img width="492" height="67" alt="image" src="https://github.com/user-attachments/assets/8c123a3e-dbcd-4055-9188-3bc1108a74bf" />

## Result
Thus, ARMA processes were generated and analyzed, different ARMA models were fitted and compared using AIC and BIC, and forecasting performance was evaluated using MAE, MSE, and RMSE. The best forecasting model was identified based on the lowest RMSE, and an ARMA(2,1) process was successfully generated and forecasted.
