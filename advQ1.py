import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize._minimize as minimize

tesla_data = yf.Ticker('TSLA').history(period='1y', interval='1d')
google_data = yf.Ticker('GOOGL').history(period='1y', interval='1d')
vanguard_data = yf.Ticker('VOO').history(period='1y', interval='1d')

l1 = len(tesla_data)
l2 = len(google_data)
l3 = len(vanguard_data)
l= min(l1,l2,l3) #to make sure all dataframes have same length(here they all are same and 250)
w1 = w2 = w3 = 1/3 # initialiing by doing equal weightage
weight_matrix = np.array([w1,w2,w3]).T  # making weight matrix column vector

# Calculate daily log returns
tesla_log_returns = tesla_data['Close'].pct_change().apply(lambda x: np.log(1 + x)).iloc[-l:]
google_log_returns = google_data['Close'].pct_change().apply(lambda x: np.log(1 + x)).iloc[-l:]
vanguard_log_returns = vanguard_data['Close'].pct_change().apply(lambda x: np.log(1 + x)).iloc[-l:]

# Calculate mean of daily log returns
mean_tesla = tesla_log_returns.mean()
mean_google = google_log_returns.mean() 
mean_vanguard = vanguard_log_returns.mean()
mean_returns = np.array([mean_tesla, mean_google, mean_vanguard]).T  # making mean returns column vector

#expected daily log returns will be mean of log returns(because mean is the most likely outcome in normal distribution)
print(f"expected daily log returns are: tesla: {mean_tesla:.5f}, google: {mean_google:.5f}, vanguard: {mean_vanguard:.5f}")

#now caculating covariance matrix of log returns
# Combine log returns into a DataFrame
log_returns_df = pd.DataFrame({
    'TSLA': tesla_log_returns,
    'GOOGL': google_log_returns,
    'VOO': vanguard_log_returns
})
# calculating covariance pairwise by cov() function
cov_matrix = log_returns_df.cov()

#just applying the formula and calculating negative sharpe ratio(because we will minimize it)
def negative_sharpe_ratio(weights, mean_returns = mean_returns, cov_matrix = cov_matrix, daily_risk_free_rate=0.02/l): #taking risk free rate as 2%
    portfolio_return = np.dot(weights.T, mean_returns)  # portfolio expected return
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix , weights)))  # portfolio standard deviation
    sharpe_ratio = (portfolio_return - daily_risk_free_rate) / portfolio_std_dev
    return -sharpe_ratio*(np.sqrt(l)) #annualizing the sharpe ratio by multiplying with trading days of that year

#constraints - weights sum is equals to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  #constraint argument for minimize function

#bounds - weights are between 0 and 1
bounds = tuple((0, 1) for _ in range(3))  #bounds argument for minimize function

#optimizing the negative sharpe ratio function
max_in_neg_sharpe_ratio = minimize.minimize(negative_sharpe_ratio, weight_matrix, method='SLSQP', bounds=bounds, constraints=constraints) #applying minimize function

print("The maximum Sharpe ratio is:", -max_in_neg_sharpe_ratio.fun.round(4))

optimal_weights = max_in_neg_sharpe_ratio.x
print(f"The optimal weights for the portfolio are: tesla: {optimal_weights[0]:.4f}, google: {optimal_weights[1]:.4f}, vanguard: {optimal_weights[2]:.4f}")

#plotting the efficient frontier graph (return vs risk)
#making function to calculate portfolio return and standard deviation further used in efficient frontier function
def portfolio_objectives(weights):  
    portfolio_return = np.dot(weights.T, mean_returns)  #calculating portfolio return
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix , weights)))  #calculating portfolio standard deviation
    return portfolio_return, portfolio_std_dev 

#function for calculating efficient frontier
def efficient_frontier(num_portfolios):
    results = np.zeros((3,num_portfolios)) #making result matrix(initializing all with zero) to store std dev, return and sharpe ratio in rows for each portfolio in columns
    for i in range(num_portfolios):
        weights = np.random.random(3) #choosing random weights for 3 assets
        weights /= np.sum(weights)    #for making their sum equal to 1
        portfolio_return, portfolio_std_dev = portfolio_objectives(weights)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - 0.02/l) / portfolio_std_dev * np.sqrt(l)  # Sharpe Ratio
    return results

num_portfolios = 5000 #number of random portfolios to simulate
results = efficient_frontier(num_portfolios)
plt.figure(figsize=(10,6))
plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.5)
plt.colorbar(label='Sharpe Ratio')  #for sharpe ratio we using colorbar for better visualization

#plotting optimal portfolio point
plt.scatter(np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix , optimal_weights))), 
            np.dot(optimal_weights.T, mean_returns), color='red', marker='.', s=200, label='Optimal Portfolio')

plt.title('Efficient Frontier Graph')
plt.xlabel('Standard Deviation(or Risk)')
plt.ylabel('Portfolio Return')
plt.legend()
plt.show()

#plotting piechart for optimal weights
plt.figure(figsize=(5,5))
plt.pie(optimal_weights, labels=['tesla', 'google', 'vanguard'], autopct='%1.3f%%', startangle=90)
plt.title('Optimal Portfolio Weights Distribution')
plt.show()

#for showing performance comparison table
#function to get row elements for comparison table
def table_row_elements(weights):
    portfolio_return, portfolio_std_dev = portfolio_objectives(weights)
    sharpe_ratio = (portfolio_return - 0.02/l) / portfolio_std_dev * np.sqrt(l)
    return portfolio_return, portfolio_std_dev, sharpe_ratio

#performance comparison table
w_equal = np.array([1/3, 1/3, 1/3])
w_tesla = np.array([1, 0, 0])
w_google = np.array([0, 1, 0])
w_vanguard = np.array([0, 0, 1])

result_table = {}
result_table['optimal_weights'] = table_row_elements(optimal_weights)
result_table['equal_weights'] = table_row_elements(w_equal)
result_table['tesla_only'] = table_row_elements(w_tesla)
result_table['google_only'] = table_row_elements(w_google)
result_table['vanguard_only'] = table_row_elements(w_vanguard)

comp_table = pd.DataFrame(result_table, index=['Expected Return', 'Standard Deviation', 'Sharpe Ratio'])
print("\nPerformance Comparison Table:")
print(comp_table)

#sensitivity analysis for risk free rate
risk_free_rates = [0.01, 0.02, 0.03, 0.04, 0.05]  # different annual risk free rates
optimal_sharpe_ratios = []

for rfr in risk_free_rates:
    negative_sharpe_ratio_rfr = lambda weights: negative_sharpe_ratio(weights, daily_risk_free_rate=rfr/l)
    result = minimize.minimize(negative_sharpe_ratio_rfr, weight_matrix, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_sharpe_ratios.append(-result.fun)  #minimize returns negative sharpe ratio then we converting it back to positive then appending to list which tells optimal sharpe ratio for that risk free rate
plt.figure(figsize=(10,6))
plt.plot(risk_free_rates, optimal_sharpe_ratios, marker='o')
plt.title('Sensitivity Analysis of Optimal Sharpe Ratio to Risk-Free Rate')
plt.xlabel('Annual Risk-Free Rate')
plt.ylabel('Optimal Sharpe Ratio')
plt.grid()
plt.show()

#sensitivity analysis for expected returns(or say mean)
mean_return_multipliers = [0.8, 0.9, 1.0, 1.1, 1.2]  # multipliers for mean returns(we are taking only+-20% change)
optimal_sharpe_ratios_mean = []
for m in mean_return_multipliers:
    adjusted_mean_returns = mean_returns * m
    negative_sharpe_ratio_mean = lambda weights: negative_sharpe_ratio(weights, mean_returns=adjusted_mean_returns)
    result = minimize.minimize(negative_sharpe_ratio_mean, weight_matrix, method='SLSQP', bounds=bounds, constraints=constraints)
    optimal_sharpe_ratios_mean.append(-result.fun)  #minimize returns negative sharpe ratio then we converting it back to positive then appending to list which tells optimal sharpe ratio for that mean return multiplier
plt.figure(figsize=(10,6))
plt.plot(mean_return_multipliers, optimal_sharpe_ratios_mean, marker='o')
plt.title('Sensitivity Analysis of Optimal Sharpe Ratio to Mean/Expected Returns')
plt.xlabel('Mean/expected Return Multiplier')
plt.ylabel('Optimal Sharpe Ratio')
plt.grid()
plt.show()    

#from output graphs from above both sensitivity models we can conclude that mean/expected return is directly proportional to sharpe ratio while risk free rate is inversely proportional to sharpe ratio and gives linear relationship