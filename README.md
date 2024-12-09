# Dự án phân cụm khách hàng bằng phương pháp K-Means

## Môn: Trí tuệ nhân tạo
### Giảng viên: Trần Đình Toàn
### Trường: Đại học Công Thương TP.HCM (HUIT)
### Nhóm 7

---

## Mô tả dự án

Dự án này được thực hiện trong khuôn khổ môn học Trí tuệ nhân tạo tại trường Đại học Công Thương TP.HCM. Mục tiêu của dự án là áp dụng phương pháp phân cụm K-Means để phân tích hành vi mua sắm của khách hàng, từ đó hỗ trợ việc phân loại và tối ưu hóa chiến lược marketing.

Chúng tôi đã sử dụng bộ dữ liệu "OnlineRetail" bao gồm thông tin giao dịch của khách hàng, bao gồm các cột như mã khách hàng, số lần mua hàng, tổng số tiền chi tiêu, và thời gian mua sắm.

### Các bước thực hiện:
1. **Tiền xử lý dữ liệu**: Làm sạch dữ liệu, loại bỏ các giá trị ngoại lệ và chuẩn hóa dữ liệu.
2. **Tính toán các chỉ số RFM**:
   - **Recency (R)**: Số ngày từ lần mua cuối cùng.
   - **Frequency (F)**: Số lần mua hàng của khách hàng.
   - **Monetary (M)**: Tổng chi tiêu của khách hàng.
3. **Phân tích và phân cụm**: Áp dụng thuật toán K-Means để phân nhóm khách hàng dựa trên các chỉ số RFM đã tính toán.
4. **Đánh giá mô hình**: Sử dụng phương pháp Elbow, Silhoutte để xác định số cụm tối ưu và huấn luyện mô hình K-Means 
   với số cụm đó.
5. **Hiển thị kết quả**: Sử dụng biểu đồ để trực quan hóa kết quả phân cụm.

---

## Các yêu cầu

- Python 3.x
- Các thư viện cần thiết:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn
  - pickle

---

## Hướng dẫn sử dụng

1. **Cài đặt các thư viện cần thiết**:
   Bạn cần cài đặt các thư viện sử dụng trong dự án bằng cách sử dụng pip:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. **Tải dữ liệu**:
   Đảm bảo bạn đã tải bộ dữ liệu `OnlineRetail.csv` vào thư mục `data/` trong dự án.

3. **Chạy mã nguồn**:
   Sau khi đã cài đặt các thư viện và tải dữ liệu, bạn có thể chạy script chính `main.py` để thực hiện toàn bộ quá trình từ tiền xử lý dữ liệu đến phân cụm khách hàng.

4. **Kết quả**:
   Sau khi chạy xong, kết quả phân cụm sẽ được hiển thị dưới dạng biểu đồ phân bố số tiền mua sắm của các nhóm khách hàng. Ngoài ra, mô hình K-Means sẽ được lưu lại trong thư mục `model/` với tên `kmeans_model1.pkl`.

---

## Cấu trúc thư mục dự án

```
.
├── data/
│   └── OnlineRetail.csv  # Dữ liệu khách hàng
├── model/
│   └── kmeans_model1.pkl  # Mô hình KMeans đã huấn luyện
├── train/
│   └── train_kmeans.py  # Huấn luyện mô hình
├── static # Các hình ảnh kết quả phân tích
├── template # Các file giao diện web
├── app.py  # Script chính để chạy mô hình
└── README.md  
```

---

## Ghi chú

- Dữ liệu được sử dụng trong dự án là dữ liệu thực tế về giao dịch mua sắm của khách hàng từ một cửa hàng trực tuyến.
- Các chỉ số RFM là cơ sở để phân loại khách hàng, giúp các doanh nghiệp hiểu rõ hơn về hành vi của khách hàng và đưa ra chiến lược marketing phù hợp.

---

## Thành viên nhóm 7

- **Hà Huỳnh Ánh Ngân** - Trưởng nhóm
- **Nguyễn Ngọc Nghĩa** - Thành viên
- **Lê Hà Bảo Khanh** - Thành viên

---

## Liên hệ

Nếu có bất kỳ câu hỏi hoặc thắc mắc nào, vui lòng liên hệ với chúng tôi qua email:
- Email nhóm: ngocnghia2004nn@gmail.com
