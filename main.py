import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from scipy.stats import norm

dataName = "incomeThailand"
df = pd.read_csv(f'data/{dataName}.csv')
cov_df = df.cov()
print(df)
eigenvalues, eigenvectors = np.linalg.eig(cov_df)
standard = preprocessing.scale(df)
x_data = df['income']
x_label = 'Income'
y_data = df['expense']
y_label = 'Expense'
plt.figure(figsize=(16,9))

#FIGURE 1
plt.subplot(2,3,1)
mean,std = norm.fit(x_data)
plt.hist(x_data, bins=20, density=True, alpha= 0.3, color='y')
xmin, xmax=plt.xlim()
x=np.linspace(xmin,xmax,100)
p=norm.pdf(x, mean, std)
plt.plot(x,p,'k',linewidth=1)
plt.title('Gaussian Function with %s [mean = %.3f and standard deviation =%.3f]'%(x_label,mean,std))
plt.grid(True)

#FIGURE 2
plt.subplot(2,3,3)
mean,std = norm.fit(y_data)
plt.hist(y_data, bins=20, density=True, alpha= 0.3, color='y')
xmin, xmax=plt.xlim()
x=np.linspace(xmin,xmax,100)
p=norm.pdf(x, mean, std)
plt.plot(x,p,'k',linewidth=1)
plt.title('Gaussian Function with %s [mean = %.3f and standard deviation =%.3f]'%(y_label,mean,std))
plt.grid(True)

#FIGURE 3
plt.subplot(2,3,4)
plt.scatter(x_data, y_data)
plt.title(f'{dataName} Raw Data')
plt.grid(True)
plt.xlabel(f'{x_label}')
plt.ylabel(f'{y_label}')

#FIGURE 4
plt.subplot(2,3,5)
np.random.seed(0)
sample=np.random.multivariate_normal(mean=df.mean(),cov=cov_df,size=df.shape[0])
col= [f'Sample_{i}' for i in range (sample.shape[1])]
gen_df =pd.DataFrame(sample,columns=col)
plt.scatter(sample[:,0],sample[:,1])
plt.title(f"{dataName} Co-Varince")
plt.xlabel(f'{x_label}')
plt.ylabel(f'{y_label}')
plt.grid(True)

#FIGURE 5
plt.subplot(2,3,6)
plt.scatter(standard[:, 1], standard[:, 0], alpha=0.5)
plt.title(f"{dataName} Standardized Data with eigenvectors")
plt.arrow(0,0,eigenvectors[0,0] ,eigenvectors[0,1],head_width=0.1,head_length=0.2,color='r')
plt.arrow(0,0,eigenvectors[1,0]*3 ,eigenvectors[1,1]*3,head_width=0.1,head_length=0.2,color='g')
plt.xlabel(f'Standardize {x_label}')
plt.ylabel(f'Standardize {y_label}')
plt.grid(True)


print(f" Covarince Matrix : {cov_df}")
print(f" EigenValues : {eigenvalues}")
print(f" EigenVectors : {eigenvectors}")
plt.show()

