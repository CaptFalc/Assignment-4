import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('online_retail.csv')
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

data.dropna(inplace=True)

features = ['Country', 'Quantity','InvoiceDate','UnitPrice']
x = data[features]

scaler = StandardScaler()
x_Scaled = scaler.fit_transform(x[['UnitPrice', 'Quantity']])

kMean = KMeans(n_clusters=5, random_state=42)
clusters = kMean.fit_predict(x_Scaled)

plt.scatter(x_Scaled[:, 0], x_Scaled[:, 1], c=clusters, cmap='viridis')
plt.xlabel('Prices (Scaled)')
plt.ylabel('Quantity (Scaled)')
plt.title('Clusters of Retail Data')
plt.show()