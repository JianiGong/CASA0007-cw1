# CASA0007-cw1
## code in paper
import numpy as np
import scipy.stats as sps
import pandas as pd
import matplotlib.pyplot as plt
data_filename = 'condition_x.csv'

#read in the data
data = pd.read_csv('D:/CASA0007/cw1/coursework_1_data_2019.csv')[['2008_ratio','2013_ratio','2018_ratio']]
#show the first few lines
data.head()

# Store each sample separately:
data1 = data['2008_ratio']
data2 = data['2013_ratio']
data3 = data['2018_ratio']

# Store some useful values

min1 = data1.min()
min2 = data2.min()
min3 = data3.min()

max1 = data1.max()
max2 = data2.max()
max3 = data3.max()

mean1 = data1.mean()
mean2 = data2.mean()
mean3 = data3.mean()

std1 = data1.std()
std2 = data2.std()
std3 = data3.std()

n1 = len(data1)
n2 = len(data2)
n3 = len(data3)


# And print some summary information:

data.describe()



#plotting histogram
output_filename1 = 'histogram_2008.png'
output_filename2 = 'histogram_2013.png'
output_filename3 = 'histogram_2018.png'
figure_width, figure_height = 7,7
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(figure_width,figure_height))
bin_info = list(np.arange(0.0003,0.009,0.0001))
# The next lines create and save the plot:
plt.hist(data1,bins=bin_info)
plt.savefig(output_filename1)

plt.hist(data2,bins=bin_info)
plt.savefig(output_filename2)

plt.hist(data3,bins=bin_info)
plt.savefig(output_filename3)


#hypothesis testing
# H0: mean ratio in 2008 =  Mean ratio in 2018
# H1: mean ratio in 2008 <  Mean ratio in 2018
# Set significance level:
alpha=0.05
std_ratio = std1/std3

print("std_ratio =", std_ratio)

if std_ratio > 0.5 and std_ratio < 2:
    print("Can assume equal population standard deviations.")
    equal_stds = True
else:
    print("Cannot assume equal population standard deviations.")
    equal_stds = False

test_stat, p_value = sps.ttest_ind(data1, data3, equal_var = equal_stds)
print("p-value =", p_value)

# Reach a conclusion:

if p_value < alpha:
    print("p-value < significance threshold.")
    print("Reject H0. Accept H1.")
   
elif p_value >= alpha:
    print("p-value >= significance threshold.")
    print("No significant evidence to reject H0.")
   
#relationship between males and females 2008
filename = 'D:/CASA0007/cw1/coursework_1_data_2019.csv'
output_filename = 'data_2008.png'
figure_width, figure_height = 7,7

import matplotlib.pyplot as plt
import statsmodels.api as sms
import numpy as np

data = np.genfromtxt(data_filename,delimiter = ',')

x_values = data[1:,29]
y_values = data[1:,32]
print(x_values)
print(y_values)
# These lines perform the regression procedure:
X_values = sms.add_constant(x_values)
regression_model_a = sms.OLS(y_values, X_values)
regression_model_b = regression_model_a.fit()
# and print a summary of the results:
print(regression_model_b.summary())
print() # blank line

# Now we store all the relevant values:
gradient  = regression_model_b.params[1]
intercept = regression_model_b.params[0]
Rsquared  = regression_model_b.rsquared
MSE       = regression_model_b.mse_resid
pvalue    = regression_model_b.f_pvalue

# And print them:
print("gradient  =", regression_model_b.params[1])
print("intercept =", regression_model_b.params[0])
print("Rsquared  =", regression_model_b.rsquared)
print("MSE       =", regression_model_b.mse_resid)
print("pvalue    =", regression_model_b.f_pvalue)

# This line creates the endpoints of the best-fit line:
x_lobf = [min(x_values),max(x_values)]
y_lobf = [x_lobf[0]*gradient + intercept,x_lobf[1]*gradient + intercept]

# This line creates the figure. 
plt.figure(figsize=(figure_width,figure_height))

# The next lines create and save the plot:
plt.plot(x_values,y_values,'b.',x_lobf,y_lobf,'r--')
plt.savefig(output_filename)




#relationship between males and females 2013
filename = 'D:/CASA0007/cw1/coursework_1_data_2019.csv'
output_filename = 'data_2013.png'
figure_width, figure_height = 7,7

import matplotlib.pyplot as plt
import statsmodels.api as sms
import numpy as np

data = np.genfromtxt(data_filename,delimiter = ',')

x_values = data[1:,30]
y_values = data[1:,33]
print(x_values)
print(y_values)
# These lines perform the regression procedure:
X_values = sms.add_constant(x_values)
regression_model_a = sms.OLS(y_values, X_values)
regression_model_b = regression_model_a.fit()
# and print a summary of the results:
print(regression_model_b.summary())
print() # blank line

# Now we store all the relevant values:
gradient  = regression_model_b.params[1]
intercept = regression_model_b.params[0]
Rsquared  = regression_model_b.rsquared
MSE       = regression_model_b.mse_resid
pvalue    = regression_model_b.f_pvalue

# And print them:
print("gradient  =", regression_model_b.params[1])
print("intercept =", regression_model_b.params[0])
print("Rsquared  =", regression_model_b.rsquared)
print("MSE       =", regression_model_b.mse_resid)
print("pvalue    =", regression_model_b.f_pvalue)

# This line creates the endpoints of the best-fit line:
x_lobf = [min(x_values),max(x_values)]
y_lobf = [x_lobf[0]*gradient + intercept,x_lobf[1]*gradient + intercept]

# This line creates the figure. 
plt.figure(figsize=(figure_width,figure_height))

# The next lines create and save the plot:
plt.plot(x_values,y_values,'b.',x_lobf,y_lobf,'r--')
plt.savefig(output_filename)





#relationship between males and females 2018
filename = 'D:/CASA0007/cw1/coursework_1_data_2019.csv'
output_filename = 'data_2018.png'
figure_width, figure_height = 7,7

import matplotlib.pyplot as plt
import statsmodels.api as sms
import numpy as np

data = np.genfromtxt(data_filename,delimiter = ',')

x_values = data[1:,31]
y_values = data[1:,34]
print(x_values)
print(y_values)
# These lines perform the regression procedure:
X_values = sms.add_constant(x_values)
regression_model_a = sms.OLS(y_values, X_values)
regression_model_b = regression_model_a.fit()
# and print a summary of the results:
print(regression_model_b.summary())
print() # blank line

# Now we store all the relevant values:
gradient  = regression_model_b.params[1]
intercept = regression_model_b.params[0]
Rsquared  = regression_model_b.rsquared
MSE       = regression_model_b.mse_resid
pvalue    = regression_model_b.f_pvalue

# And print them:
print("gradient  =", regression_model_b.params[1])
print("intercept =", regression_model_b.params[0])
print("Rsquared  =", regression_model_b.rsquared)
print("MSE       =", regression_model_b.mse_resid)
print("pvalue    =", regression_model_b.f_pvalue)

# This line creates the endpoints of the best-fit line:
x_lobf = [min(x_values),max(x_values)]
y_lobf = [x_lobf[0]*gradient + intercept,x_lobf[1]*gradient + intercept]

# This line creates the figure. 
plt.figure(figsize=(figure_width,figure_height))

# The next lines create and save the plot:
plt.plot(x_values,y_values,'b.',x_lobf,y_lobf,'r--')
plt.savefig(output_filename)
