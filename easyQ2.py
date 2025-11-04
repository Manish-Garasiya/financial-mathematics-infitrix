import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#when you want to download data from yfinance uncomment the below lines
# history = yf.download('RELIANCE.NS', period='2y', interval='1d')
# history.to_csv('reliance_data_2y.csv')
# hist = pd.read_csv('reliance_data_2y.csv',skiprows=[1,2]) 

#without downloading csv file(directly from yfinance)
hist = yf.Ticker('RELIANCE.NS').history(period='2y', interval='1d')
daily_log_returns = hist['Close'].pct_change().apply(lambda x: np.log(1 + x)) 
length_of_data = len(daily_log_returns)
# print(daily_log_returns)

portfolio_value = 100000  # assuming pv in INR
# Calculate VaR at 95% and 99% confidence levels
conf_level1 = 0.95
var_95 = daily_log_returns.quantile(1 - conf_level1) #quantile function will auomatically sort the data then find the VaR value at required percentile
conf_level2 = 0.99
var_99 = daily_log_returns.quantile(1 - conf_level2)

# Converting log returns to percentage VaR

percent_var_95 = -(np.exp(var_95)-1) * 100  # converting log returns to percentage
percent_var_99 = -(np.exp(var_99)-1) * 100  # -ve sign for making VaR positive

# Calculating dollar VaR
dollar_var_95 = (percent_var_95 / 100) * portfolio_value
dollar_var_99 = (percent_var_99/ 100) * portfolio_value

print(f"The values of VaR in percentage and magnitude of dollar VaR are: ")
print(f"at 95% confidence level: {percent_var_95:.2f}% or INR {dollar_var_95:.2f}")
print(f"at 99% confidence level: {percent_var_99:.2f}% or INR {dollar_var_99:.2f}")

# plotting histogram log returns with VaR graph with VaR thresholds
plt.figure(figsize=(10,6))
plt.hist(daily_log_returns.dropna(), bins='auto', alpha=0.65, color='blue', edgecolor='black')
plt.axvline(var_95,color = 'red',linestyle = '--', label='VaR at 95% ,confidence level')
plt.axvline(var_99,color = 'purple',linestyle = '--', label='VaR at 99% ,confidence level')
plt.title('Histogram for daily log returns with VaR threshholds')
plt.xlabel('Daily Log Returns')
plt.ylabel('frequency')
plt.legend()
plt.show()

#now calculating rolling VaR
window_size = 200
rolling_var_95_data= daily_log_returns.rolling(window=window_size) # it will group data in rolling window of size 200
rolling_var_95 = rolling_var_95_data.quantile(1 - conf_level1) # calculate VaR by quantile function
rolling_var_99_data = daily_log_returns.rolling(window=window_size)
rolling_var_99 = rolling_var_95_data.quantile(1 - conf_level2)

#plotting rolling VaR graph
plt.figure(figsize=(12,6))
plt.plot(rolling_var_95, color='red', label='Rolling VaR at 95% ,confidence level')
plt.plot(rolling_var_99, color='purple', label='Rolling VaR at 99% ,confidence level')
plt.title('Rolling VaR with time')
plt.xlabel('Date')
plt.ylabel('Rolling VaR')       
plt.legend()
plt.show()

#Now backtesting our VaR model
newhist = yf.Ticker('RELIANCE.NS').history(period='3y', interval='1d') #testing for longer period
new_daily_returns = newhist['Close'].pct_change() #calculate daily returns for new data
len_newdata = len(new_daily_returns)
newhist['VaR_95'] = new_daily_returns < -percent_var_95 / 100  #if actual loss exceeds VaR, if true then exception
newhist['VaR_99'] = new_daily_returns < -percent_var_99 / 100  #-ve sign to convert VaR to negative for comparison
exceptions_95 = newhist['VaR_95'].sum() #counting total exceptions
exceptions_99 = newhist['VaR_99'].sum()
print(f"\nThe real probability when actual loss exceeded the VaR estimates in backtesting period of {len_newdata} days: ")
print(f"at 95% confidence level: {(exceptions_95/len_newdata)*100:.2f}%","our model predicted 5% ")
print(f"at 99% confidence level: {(exceptions_99/len_newdata)*100:.2f}%","our model predicted 1% ")

