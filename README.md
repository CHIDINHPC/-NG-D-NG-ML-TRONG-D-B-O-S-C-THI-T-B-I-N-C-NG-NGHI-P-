# 📘 ỨNG DỤNG MACHINE LEARNING TRONG DỰ BÁO SỰ CỐ THIẾT BỊ ĐIỆN CÔNG NGHIỆP
<p align="center"><img width="1915" height="959" alt="image" src="https://github.com/user-attachments/assets/9fd9029d-a3e0-4872-9bb7-9bfca5b2d911" /></p>

## 🚀 Giới thiệu
Dự án xây dựng mô hình **Machine Learning** nhằm dự đoán khả năng xảy ra sự cố của các thiết bị điện trong nhà máy. Giải pháp giúp doanh nghiệp chuyển đổi từ **bảo trì định kỳ (PM)** sang **bảo trì dự đoán (Predictive Maintenance – PdM)**, từ đó:
- Giảm chi phí bảo trì
- Hạn chế downtime sản xuất
- Nâng cao độ tin cậy và tuổi thọ thiết bị
- Tối ưu hóa kế hoạch vận hành

---

## 🎯 Mục tiêu dự án
- Thu thập và xử lý dữ liệu vận hành thiết bị điện.
- Xây dựng mô hình ML (Random Forest, XGBoost, SVM…).
- Dự đoán xác suất xảy ra sự cố trong tương lai.
- Tích hợp dashboard trực quan hóa kết quả.
- Hỗ trợ kỹ sư bảo trì ra quyết định nhanh và chính xác.

---

## 🛠️ Công nghệ sử dụng
- **Ngôn ngữ:** Python  
- **Thư viện:** NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn  
- **Mô hình ML:** Random Forest, Logistic Regression, XGBoost (tùy chọn)  
- **Công cụ trực quan:** Streamlit  
- **Triển khai:** GitHub, Colab/Jupyter Notebook

---

## 📂 Cấu trúc thư mục
<p align="center">
  <img src="https://github.com/user-attachments/assets/1e54b353-8228-4b36-89c2-a93e60450c75"
       alt="image"
       width="400"
       height="250">
</p>

---

## 📊 Quy trình thực hiện

### **1. Thu thập dữ liệu**
- Dữ liệu dòng điện, điện áp, nhiệt độ
- Thời gian hoạt động, chu kỳ đóng cắt
- Lịch sử bảo trì và hỏng hóc

### **2. Tiền xử lý dữ liệu**
- Xử lý giá trị thiếu  
- Loại bỏ nhiễu  
- Chuẩn hóa dữ liệu  
- Trích xuất đặc trưng (Feature Engineering)

### **3. Huấn luyện mô hình Machine Learning**
- So sánh nhiều thuật toán (RF, LR, SVM, XGBoost…)  
- Tối ưu tham số bằng GridSearchCV  
- Lưu mô hình (.pkl)  
- Đánh giá bằng Accuracy, Precision, Recall, F1-score

### **4. Đánh giá mô hình**
- Confusion Matrix  
- Classification Report  
- ROC Curve – AUC  
- Feature Importance  

### **5. Xây dựng và triển khai dashboard**
- Dashboard dựa trên Streamlit  
- Cho phép người dùng nhập dữ liệu mới  
- Hiển thị xác suất thiết bị sắp xảy ra sự cố  
- Biểu đồ phân tích trực quan

<p align="center"> <img width="1916" height="946" alt="image" src="https://github.com/user-attachments/assets/1b1607a8-6b11-4c7a-bac0-1ec0f57bb107" />
</p>
<p align="center"> <img width="1919" height="950" alt="image" src="https://github.com/user-attachments/assets/c84065d9-9009-4385-bf16-7a10702b1fde" />
 </p>
 <p align="center"> <img width="1913" height="943" alt="image" src="https://github.com/user-attachments/assets/f7fa111a-660d-44dd-8ee7-abf08b623ca8" />
  </p>
