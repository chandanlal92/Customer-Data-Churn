import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
print (matplotlib.rcParams['backend'])
# try:
#   from ydata_profiling import ProfileReport
# except:
#   !pip install ydata_profiling
#   from ydata_profiling import ProfileReport

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
report= ProfileReport(data, title="Telco Customer Churn Report", explorative=True)
report.to_file("Telco_Customer_Churn_Report.html")
# Data distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Churn', data=data)
plt.title('Churn Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(data.corr(numeric_only=True),annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap')
plt.show()

# Pairplot
sns.pairplot(data, hue='Churn')
plt.show()