import numpy as np
from scipy.optimize import minimize
import time
import pandas as pd
import os,sys
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
os.chdir(current_directory)

# Define the sub-function f(x)
def f(x, coeffs):
    a, b, c, d = coeffs
    return a + b*x + c*x**2 + d*x**3

# Define the objective function
def objective(x, n):
    # Generate random coefficients for each sub-function
    coeffs_list = [np.random.rand(4) for i in range(n)]
    # Evaluate each sub-function at x
    f_vals = [f(x[i], coeffs_list[i]) for i in range(n)]
    # Calculate y as max(f_vals) - min(f_vals)
    y = np.max(f_vals) - np.min(f_vals)
    return y


reslist=[]


for i in range(1,200):
    
    # Find the minimum of the objective function
    n = i  # user-defined dimensionality of the problem
    # Define the bounds for x
    bounds = [(0, 1) for i in range(n)]




    x0 = np.random.rand(n)  # initial guess for x

    recordTemp=[]
    for __i in range (6):
        start=time.time()
        result = minimize(objective, x0, args=(n,),method='SLSQP', bounds=bounds)
        timeconsume=round(time.time()-start,4)
        recordTemp.append(timeconsume)
    recordTemp.remove( max(recordTemp) )
    recordTemp.remove( min(recordTemp) )
    res= round(  sum(recordTemp)/len(recordTemp), 4  ) 


    reslist.append(res)
    
    # print(f'time consume:{timeconsume}')
    # Print the result
    # print("Minimum value of y:", result.fun)
    # print("Optimal x values:", result.x)
print(reslist)


df=pd.DataFrame(reslist)
df.to_csv('solve_balance_problem.csv',)















# import numpy as np
# from scipy.optimize import minimize

# # Define the sub-function f(x)
# def f(x, coeffs):
#     a, b, c, d = coeffs
#     return a + b*x + c*x**2 + d*x**3

# # Define the objective function
# def objective(x, f_count):
#     # Generate random coefficients for each sub-function
#     coeffs_list = [np.random.rand(4) for i in range(f_count)]
#     # Evaluate each sub-function at x
#     f_vals = [f(x, coeffs_list[i]) for i in range(f_count)]
#     # Calculate y as max(f_vals) - min(f_vals)
#     y = np.max(f_vals) - np.min(f_vals)
#     return y

# # Define the bounds for x
# bounds = [(0, 1) for i in range(4)]

# # Find the minimum of the objective function
# f_count = 500  # user-defined number of sub-functions
# x0 = np.random.rand(4)  # initial guess for x
# result = minimize(objective, x0, args=(f_count,), bounds=bounds)

# # Print the result
# print("Minimum value of y:", result.fun)
# print("Optimal x values:", result.x)
