import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Đọc dữ liệu
retail = pd.read_csv('../model/train/OnlineRetail_Train.csv', sep=",", encoding="ISO-8859-1", header=0)

# Chuyển CustomerID thành chuỗi và tạo cột Amount
retail['CustomerID'] = retail['CustomerID'].astype(str)
retail['Amount'] = retail['Quantity'] * retail['UnitPrice']

# Tính các chỉ số RFM
# Tính Monetary - Tổng số tiền mua hàng của mỗi khách hàng
rfm_m = retail.groupby('CustomerID')['Amount'].sum().reset_index()

# Tính Frequency - Số lần mua hàng của mỗi khách hàng
rfm_f = retail.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
rfm_f.columns = ['CustomerID', 'Frequency']

# Tính Recency - Số ngày kể từ lần mua hàng cuối cùng
retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format='%d-%m-%Y %H:%M')
max_date = retail['InvoiceDate'].max()  # Lấy ngày lớn nhất trong InvoiceDate
retail['Diff'] = (max_date - retail['InvoiceDate']).dt.days
rfm_p = retail.groupby('CustomerID')['Diff'].min().reset_index()  # Tính Recency là ngày cách xa nhất
rfm_p.columns = ['CustomerID', 'Recency']

# Merge các chỉ số lại với nhau
rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')

# Kiểm tra kết quả tính toán RFM
print(rfm.head())

# Xử lý ngoại lệ (outliers)
rfm = rfm.select_dtypes(include=[np.number])
Q1 = rfm.quantile(0.05)
Q3 = rfm.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm['Amount'] >= Q1['Amount'] - 1.5*IQR['Amount']) & (rfm['Amount'] <= Q3['Amount'] + 1.5*IQR['Amount'])]
rfm = rfm[(rfm['Recency'] >= Q1['Recency'] - 1.5*IQR['Recency']) & (rfm['Recency'] <= Q3['Recency'] + 1.5*IQR['Recency'])]
rfm = rfm[(rfm['Frequency'] >= Q1['Frequency'] - 1.5*IQR['Frequency']) & (rfm['Frequency'] <= Q3['Frequency'] + 1.5*IQR['Frequency'])]

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Amount', 'Frequency', 'Recency']])

# Chuyển sang DataFrame để dễ dàng thao tác sau này
rfm_scaled = pd.DataFrame(rfm_scaled, columns=['Amount', 'Frequency', 'Recency'])

# Phương pháp Elbow để xác định số cụm tối ưu
wss = []  # Within Sum of Squares

# Iterate over possible values of k
for k in range(2, 10 + 1):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    wss.append(kmeans.inertia_)

# Phương pháp Elbow distances để tìm k tối ưu
nPoints = len(wss)
allCoord = np.vstack((range(nPoints), wss)).T
firstPoint = allCoord[0]
lineVec = allCoord[-1] - allCoord[0]
lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
vecFromFirst = allCoord - firstPoint
scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
vecToLine = vecFromFirst - vecFromFirstParallel
distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))

# Tìm điểm gập góc (elbow) - chỉ số k tương ứng
elbow_k = 2 + np.argmax(distToLine)  # Thêm 2 vì bắt đầu từ k=2

# Huấn luyện mô hình KMeans với số cụm k tối ưu
kmeans = KMeans(n_clusters=elbow_k, random_state=42)
kmeans.fit(rfm_scaled)

# Dự đoán các cụm
rfm['Cluster_Id'] = kmeans.predict(rfm_scaled)

# Xem kết quả phân cụm

print("Hi",rfm.columns)  # In ra danh sách các cột trong rfm


plt.figure(figsize=(10, 6))
sns.stripplot(x='Cluster_Id', y='Amount', data=rfm, hue='Cluster_Id', palette='Set2')
plt.title("Phân bố Số Tiền Mua Sắm Theo Các Nhóm")
plt.xlabel("Nhóm Khách Hàng")
plt.ylabel("Số Tiền Mua Sắm (Amount)")
plt.show()

# Lưu mô hình KMeans
with open('../model/kmeans_model3.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
