from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import seaborn as sns
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy.matlib
app = Flask(__name__)
model = None  # Initialize model as None

def init_model(model_name):
    global model

    if model_name == "model1":
        model = pickle.load(open('model/kmeans_model1.pkl', 'rb'))
    elif model_name == "model2":
        model = pickle.load(open('model/kmeans_model2.pkl', 'rb'))
    else:
        raise ValueError("Model name not recognized")

def generate_plots(df_with_id):
    # Biểu đồ số tiền theo cụm
    plt.figure(figsize=(10, 6))
    sns.stripplot(x='Cluster_Id', y='Amount', data=df_with_id, hue='Cluster_Id', palette='Set2')
    plt.title("Phân bố Số Tiền Theo Các Nhóm")
    plt.xlabel("Nhóm Khách Hàng")
    plt.ylabel("Số Tiền Mua Sắm (Amount)")
    amount_img_path = 'static/ClusterId_Amount.png'
    plt.savefig(amount_img_path)
    plt.clf()

    # Biểu đồ tần suất theo cụm
    plt.figure(figsize=(10, 6))
    sns.stripplot(x='Cluster_Id', y='Frequency', data=df_with_id, hue='Cluster_Id', palette='Set2')
    plt.title("Phân bố Tần Suất Giao Dịch Theo Các Nhóm")
    plt.xlabel("Nhóm Khách Hàng")
    plt.ylabel("Tần Suất Giao Dịch")
    freq_img_path = 'static/ClusterId_Frequency.png'
    plt.savefig(freq_img_path)
    plt.clf()

    # Biểu đồ ngày mua sắm gần nhất theo cụm
    plt.figure(figsize=(10, 6))
    sns.stripplot(x='Cluster_Id', y='Recency', data=df_with_id, hue='Cluster_Id', palette='Set2')
    plt.title("Phân bố Ngày Mua Sắm Gần Nhất Theo Các Nhóm")
    plt.xlabel("Nhóm Khách Hàng")
    plt.ylabel("Ngày Mua Sắm Gần Nhất (Recency)")
    recency_img_path = 'static/ClusterId_Recency.png'
    plt.savefig(recency_img_path)
    plt.clf()

    return amount_img_path, freq_img_path, recency_img_path


def determine_best_k(data, max_k=10):
    wss = []  # Within Sum of Squares
    silhouette_avg = []  # Silhouette Score

    # Iterate over possible values of k
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        wss.append(kmeans.inertia_)
        silhouette_avg.append(silhouette_score(data, kmeans.labels_))

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

    # Silhouette Score: Best k is the one with the highest silhouette score
    silhouette_k = range(2, max_k + 1)[silhouette_avg.index(max(silhouette_avg))]

    # Vẽ biểu đồ Elbow
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_k + 1), wss, marker='o')
    plt.title("Elbow Method - WSS vs. k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Within-cluster Sum of Squares")
    plt.axvline(x=elbow_k, color='r', linestyle='--', label=f'Optimal k={elbow_k}')
    plt.grid(True)
    elbow_img_path = 'static/elbow_method.png'
    plt.savefig(elbow_img_path)
    plt.clf()

    # Vẽ biểu đồ Silhouette Score
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_k + 1), silhouette_avg, marker='o', color='orange')
    plt.title("Silhouette Score vs. k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    silhouette_img_path = 'static/silhouette_score.png'
    plt.savefig(silhouette_img_path)
    plt.clf()

    # Chọn k tối ưu dựa trên cả Elbow Method và Silhouette Score
    # So sánh giữa k theo Elbow và k theo Silhouette, lấy giá trị k cho thấy cả hai phương pháp
    # đều có kết quả gần giống hoặc tương tự nhất

    # So sánh Elbow và Silhouette, chọn k gần nhất
    if abs(elbow_k - silhouette_k) <= 1:
        optimal_k = elbow_k  # Nếu chúng gần nhau, chọn k của Elbow Method
        method = "elbow"
    else:
        # Nếu khác biệt quá lớn, chọn k theo Silhouette vì nó đánh giá chất lượng phân nhóm
        optimal_k = silhouette_k
        method = "silhouette"

    return optimal_k,method, elbow_k, silhouette_k, elbow_img_path, silhouette_img_path


def load_and_clean_data(file_path):
    # Đọc dữ liệu từ file CSV vào DataFrame retail
    retail = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)

    # Chuyển đổi CustomerID thành kiểu chuỗi và tạo cột Amount (Số tiền mua hàng)
    retail['CustomerID'] = retail['CustomerID'].astype(str)
    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']

    # Tính toán các chỉ số RFM
    # Tính Monetary (số tiền mua hàng của mỗi khách hàng)
    rfm_m = retail.groupby('CustomerID')['Amount'].sum().reset_index()

    # Tính Frequency (số lần mua hàng của mỗi khách hàng)
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']  # Đổi tên cột thành Frequency

    # Tính Recency (số ngày kể từ lần mua hàng cuối cùng của mỗi khách hàng)
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'],
                                           format='%d-%m-%Y %H:%M')  # Chuyển đổi ngày tháng về dạng datetime
    max_date = max(retail['InvoiceDate'])  # Lấy ngày mua cuối cùng trong dữ liệu
    retail['Diff'] = max_date - retail['InvoiceDate']  # Tính số ngày kể từ lần mua hàng cuối
    rfm_p = retail.groupby('CustomerID')['Diff'].min().reset_index()  # Tính Recency (ngày gần nhất)
    rfm_p['Diff'] = rfm_p['Diff'].dt.days  # Chuyển đổi ngày thành số ngày
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')  # Kết hợp các thông tin Monetary và Frequency
    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')  # Kết hợp thông tin Recency
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']  # Đổi tên cột cho dễ đọc

    # Loại bỏ các giá trị ngoại lệ (outliers)
    rfm = rfm.select_dtypes(include=[np.number])  # Chỉ chọn các cột số
    Q1 = rfm.quantile(0.05)  # Lấy giá trị tại phần trăm 5% (Q1)
    Q3 = rfm.quantile(0.95)  # Lấy giá trị tại phần trăm 95% (Q3)
    IQR = Q3 - Q1  # Tính khoảng giữa Q3 và Q1 (Interquartile Range)

    # Loại bỏ các dòng có giá trị ngoại lệ đối với các cột Amount, Recency và Frequency
    rfm = rfm[(rfm.Amount >= Q1[0] - 1.5 * IQR[0]) & (rfm.Amount <= Q3[0] + 1.5 * IQR[0])]
    rfm = rfm[(rfm.Recency >= Q1[2] - 1.5 * IQR[2]) & (rfm.Recency <= Q3[2] + 1.5 * IQR[2])]
    rfm = rfm[(rfm.Frequency >= Q1[1] - 1.5 * IQR[1]) & (rfm.Frequency <= Q3[1] + 1.5 * IQR[1])]

    return rfm  # Trả về dữ liệu đã được xử lý


def analyze_clusters(df_with_id):
    cluster_analysis = df_with_id.groupby('Cluster_Id').agg(
        {'Amount': ['mean', 'sum', 'std'],
         'Frequency': ['mean', 'sum', 'std'],
         'Recency': ['mean', 'min', 'max']}).reset_index()

    cluster_analysis.columns = ['Cluster_Id', 'Avg_Amount', 'Total_Amount', 'Std_Amount', 'Avg_Frequency',
                                'Total_Frequency', 'Std_Frequency', 'Avg_Recency', 'Min_Recency', 'Max_Recency']

    # Dự đoán các nhóm
    cluster_analysis['Group_Description'] = cluster_analysis.apply(lambda row:
                                                                   "Nhóm này có mức chi tiêu cao" if row[
                                                                                                         'Avg_Amount'] > 1000 else
                                                                   "Nhóm này có mức chi tiêu thấp", axis=1)

    return cluster_analysis


def preprocess_data(file_path):
    # Gọi hàm load_and_clean_data để tải và làm sạch dữ liệu
    rfm = load_and_clean_data(file_path)

    # Lựa chọn các cột 'Amount', 'Frequency', 'Recency' từ dữ liệu đã được làm sạch
    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]

    # Khởi tạo đối tượng StandardScaler để chuẩn hóa dữ liệu
    scaler = StandardScaler()

    # Sử dụng phương thức fit_transform để chuẩn hóa dữ liệu
    rfm_df_scaled = scaler.fit_transform(rfm_df)

    # Chuyển đổi dữ liệu đã chuẩn hóa thành DataFrame để dễ dàng xử lý và hiển thị
    rfm_df_scaled = pd.DataFrame(rfm_df_scaled)

    # Đổi tên cột của DataFrame đã chuẩn hóa thành 'Amount', 'Frequency', 'Recency'
    rfm_df_scaled.columns = ['Amount', 'Frequency', 'Recency']

    # Trả về dữ liệu ban đầu (rfm) và dữ liệu đã chuẩn hóa (rfm_df_scaled)
    return rfm, rfm_df_scaled


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']  # Nhận file người dùng tải lên
    method = request.form.get('method')  # Lấy phương pháp phân nhóm người dùng chọn
    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)

    # Tiền xử lý dữ liệu
    rfm, rfm_df_scaled = preprocess_data(file_path)

    if method == 'custom':
        # Lấy số nhóm người dùng nhập vào
        custom_clusters = request.form.get('custom-clusters')

        # Kiểm tra xem số nhóm có hợp lệ không (lớn hơn hoặc bằng 2)
        try:
            custom_clusters = int(custom_clusters)
            if custom_clusters < 2:
                return json.dumps({'error': 'Số nhóm phải lớn hơn hoặc bằng 2!'}, ensure_ascii=False)

            # Phân nhóm với số nhóm người dùng chọn
            kmeans = KMeans(n_clusters=custom_clusters, random_state=42)
            clusters = kmeans.fit_predict(rfm_df_scaled)

            # Mô tả kết quả phân nhóm
            cluster_description = f"Số nhóm được chọn là {custom_clusters}. Đã sử dụng {custom_clusters} nhóm để phân nhóm khách hàng."

            # Gắn nhãn cụm vào dữ liệu
            df_with_id = preprocess_data(file_path)[0]
            df_with_id['Cluster_Id'] = clusters
            # Tạo biểu đồ
            amount_img_path, freq_img_path, recency_img_path = generate_plots(df_with_id)

            # Phân tích nhóm
            df_with_id = preprocess_data(file_path)[0]
            df_with_id['Cluster_Id'] = clusters
            cluster_analysis = analyze_clusters(df_with_id)
            cluster_analysis_json = cluster_analysis.to_json(orient='records', force_ascii=False)

            response = {
                'method': method,
                'cluster_description': cluster_description,
                'amount_img': amount_img_path,
                'freq_img': freq_img_path,
                'recency_img': recency_img_path,
                'cluster_analysis': cluster_analysis_json,
            }

            return json.dumps(response, ensure_ascii=False)
        except ValueError:
            return json.dumps({'error': 'Số nhóm không hợp lệ!'}, ensure_ascii=False)

    # Nếu không sử dụng Elbow/Silhouette, sử dụng mô hình đã lưu
    if method == 'model1' or method == 'model2':
        init_model(method)
        results_df = model.predict(rfm_df_scaled)  # Sử dụng mô hình đã lưu để dự đoán
        df_with_id = preprocess_data(file_path)[0]
        df_with_id['Cluster_Id'] = results_df

        # Tạo biểu đồ
        amount_img_path, freq_img_path, recency_img_path = generate_plots(df_with_id)

        # Phân tích nhóm
        cluster_analysis = analyze_clusters(df_with_id)
        cluster_analysis_json = cluster_analysis.to_json(orient='records', force_ascii=False)

        response = {
            'method': method,
            # 'cluster_description': "Phân cụm sử dụng mô hình đã được huấn luyện.",
            'amount_img': amount_img_path,
            'freq_img': freq_img_path,
            'recency_img': recency_img_path,
            'cluster_analysis': cluster_analysis_json,
        }

        return json.dumps(response, ensure_ascii=False)

    # Phân nhóm theo phương pháp Elbow hoặc Silhouette
    elif method in ['elbow', 'silhouette','optimal']:
        # Xác định số cụm tối ưu
        optimal_k, m,elbow_k, silhouette_k, elbow_img, silhouette_img = determine_best_k(rfm_df_scaled)

        # Chọn số cụm dựa vào phương pháp người dùng chọn
        if method == 'elbow':
            kmeans = KMeans(n_clusters=elbow_k, random_state=42)
            clusters = kmeans.fit_predict(rfm_df_scaled)
            cluster_description = f"Số cụm tối ưu theo phương pháp Elbow là {elbow_k}. Đã sử dụng {elbow_k} cụm để phân nhóm khách hàng."
            img_path = elbow_img
        elif method == 'silhouette':
            kmeans = KMeans(n_clusters=silhouette_k, random_state=42)
            clusters = kmeans.fit_predict(rfm_df_scaled)
            cluster_description = f"Số cụm tối ưu theo phương pháp Silhouette là {silhouette_k}. Đã sử dụng {silhouette_k} cụm để phân nhóm khách hàng."
            img_path = silhouette_img
        elif method == 'optimal':
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(rfm_df_scaled)
            cluster_description = (f"Số cụm tối ưu theo phương pháp {m} tối ưu nhat là {silhouette_k}. Đã sử dụng"
                                   f" {optimal_k} cụm để phân nhóm khách hàng.")
            if m == 'elbow':
                img_path = elbow_img
            elif m == 'silhouette':
                img_path = silhouette_img
        # Gắn nhãn cụm vào dữ liệu
        df_with_id = preprocess_data(file_path)[0]
        df_with_id['Cluster_Id'] = clusters

        # Tạo biểu đồ
        amount_img_path, freq_img_path, recency_img_path = generate_plots(df_with_id)

        # Phân tích nhóm
        cluster_analysis = analyze_clusters(df_with_id)
        cluster_analysis_json = cluster_analysis.to_json(orient='records', force_ascii=False)

        response = {
            'method': method,
            'cluster_description': cluster_description,
            'cluster_img': img_path,
            'amount_img': amount_img_path,
            'freq_img': freq_img_path,
            'recency_img': recency_img_path,
            'cluster_analysis': cluster_analysis_json,
        }

        return json.dumps(response, ensure_ascii=False)

    else:
        return json.dumps({'error': 'Phương pháp không hợp lệ!'}, ensure_ascii=False)



if __name__ == "__main__":
    app.run(debug=False)

#  Cho hosting
# if __name__ == "__main__":
#     app.run(host = '0.0.0.0',port=8080)