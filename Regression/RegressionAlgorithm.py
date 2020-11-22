from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def create_dataset(hm, variance, step=2, correlation=False): # create random data se for testing
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val  += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [ i for i in range(len(ys)) ]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope_and_intercept(xs, ys): # calculate best fit slope and Y intercept
    m = ( ( ( mean(xs) * mean(ys) ) - mean(xs * ys) ) / 
            ( (mean(xs) * mean(xs)) - mean(xs * xs) ) )
    
    b = mean(ys) - (m * mean(xs) )

    return m, b

def squared_error(ys_orig, ys_line): # calculate squared error distance between line and points
    return sum( (ys_line - ys_orig)**2 )

def coefficient_of_determination(ys_orig, ys_line): # calculate coefficient of determination, how much better our line from mean line of ys 
    y_mean_line = [ mean(ys_orig) for y in ys_orig ]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)


xs, ys = create_dataset(40, 10, 2, correlation='pos')

m,b = best_fit_slope_and_intercept(xs, ys)

regression_line = [ (m*x)+b for x in xs ] # line set we get from best fit slope and y intercept  

predict_x = 8 # prediction x just for example 
predict_y = (m * predict_x)+b # prediction example

r_squared = coefficient_of_determination(ys, regression_line) # lets see how much better our line is
print(r_squared) 

## Plot
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, c='g', s=80)
plt.plot(xs, regression_line)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()

#print(m, b)