import time
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import plotly.express as px
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------- CONFIG --------------------
st.set_page_config(page_title="Ứng dụng Dữ liệu lớn trong bảo trì dự đoán thiết bị công nghiệp", layout="wide")
# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    
    df = pd.read_csv("ai4i2020.csv")
    return df

data = load_data()

# Tách nhãn
# Xóa các cột ID / không dùng
drop_cols = ["Product ID", "UDI"]  # kiểm tra trong CSV xem tên chính xác
X = data.drop(columns=drop_cols + ["Machine failure"], errors="ignore")
y = data["Machine failure"]

# Encode nếu có cột Type
if "Type" in X.columns:
    X = pd.get_dummies(X, columns=["Type"])

y = data["Machine failure"]

# Encode cột 'Type' nếu có
if "Type" in X.columns:
    X = pd.get_dummies(X, columns=["Type"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- CSS custom ---
st.markdown("""
<style>
/* Background chính */
.reportview-container {
       background: linear-gradient(to right, #2193b0, #6dd5ed);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Sidebar */
.sidebar .sidebar-content {
    background: linear-gradient(to bottom, #3498db, #2ecc71);
    color: white;
    border-radius: 10px;
    padding: 20px;
    transition: all 0.3s ease;
}

/* Hiệu ứng hover sidebar */
.sidebar .sidebar-content:hover {
    box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    transform: scale(1.01);
}

/* Tiêu đề sidebar */
.sidebar h2 {
    color: white;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}

/* Cards thông số */
.cards-container {
    display: flex;
    justify-content: space-between; /* căn đều khoảng cách */
    margin-bottom: 20px;
}
.card {
    background: #ffffff;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    text-align: center;
    flex: 1; /* chiếm đều chiều ngang */
    margin: 0 10px; /* khoảng cách giữa các card */
    transition: all 0.3s ease-in-out;
}

/* Hover card */
.card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
}

/* Tiêu đề chính */
h1, h2, h3 {
    color: #2c3e50;
    font-weight: bold;
    transition: color 0.3s;
}

/* Hover đổi màu tiêu đề */
h1:hover, h2:hover, h3:hover {
    color: #3498db;
}

/* Button đẹp */
.stButton>button {
    background: linear-gradient(45deg, #3498db, #2ecc71);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
    transition: all 0.3s ease;
}

/* Hover button */
.stButton>button:hover {
    transform: scale(1.08);
    cursor: pointer;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
st.sidebar.title("Dashboard")
menu = st.sidebar.radio("Chọn mục", ["Home", "Analytics", "Settings",])

# --- Header ---
st.title("📊 Dashboard Dữ liệu Công nghiệp")

# --- Home Page ---
# --- Homepage ---
if menu == "Home":
    st.subheader("Tổng quan hệ thống thiết bị")

    # Tổng số thiết bị (theo UDI)
    total_devices = data['UDI'].nunique()

    # Hiệu suất trung bình 
    avg_torque = data['Torque [Nm]'].mean()

    # Số lỗi (machine failure = 1)
    total_failures = (data['Machine failure'] == 1).sum()

    # Hiển thị 3 KPI
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Tổng số thiết bị", total_devices)

    with col2:
        st.metric("Torque trung bình", f"{avg_torque:.2f} Nm")

    with col3:
        st.metric("Tổng số lỗi", total_failures)

# --- Analytics Page ---
elif menu == "Analytics":
    st.subheader("📊 Biểu đồ phân tích dữ liệu thực tế")

    # 1. Hiệu suất trung bình theo nhiệt độ quá trình
    fig1 = px.scatter(
        data, 
        x="Process temperature [K]", 
        y="Rotational speed [rpm]", 
        color="Machine failure",
        title="Mối quan hệ giữa Nhiệt độ quá trình & Tốc độ quay (theo tình trạng máy)",
        labels={"Process temperature [K]": "Nhiệt độ quá trình (K)",
                "Rotational speed [rpm]": "Tốc độ quay (rpm)"}
    )
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Phân bố tốc độ quay
    fig2 = px.histogram(
        data,
        x="Rotational speed [rpm]",
        nbins=50,
        title="Phân bố tốc độ quay của thiết bị"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Tỉ lệ máy hỏng vs bình thường
    failure_counts = data["Machine failure"].value_counts().reset_index()
    failure_counts.columns = ["Trạng thái", "Số lượng"]
    failure_counts["Trạng thái"] = failure_counts["Trạng thái"].map({0: "✅ Bình thường", 1: "⚠️ Hỏng"})

    fig3 = px.pie(
        failure_counts, 
        names="Trạng thái", 
        values="Số lượng",
        title="Tỉ lệ tình trạng máy"
    )
    st.plotly_chart(fig3, use_container_width=True)


# --- Settings Page ---
elif menu == "Settings":
    st.subheader("Cài đặt Dashboard")
    st.button("Cập nhật dữ liệu")
    st.button("Đặt lại")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ai4i2020.csv")
    return df

data = load_data()

# Tách nhãn
# Xóa các cột ID / không dùng
drop_cols = ["Product ID", "UDI"]  # kiểm tra trong CSV xem tên chính xác
X = data.drop(columns=drop_cols + ["Machine failure"], errors="ignore")
y = data["Machine failure"]

# Encode nếu có cột Type
if "Type" in X.columns:
    X = pd.get_dummies(X, columns=["Type"])

y = data["Machine failure"]

# Encode cột 'Type' nếu có
if "Type" in X.columns:
    X = pd.get_dummies(X, columns=["Type"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- TRAIN MODEL --------------------
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Lưu model ra file (chỉ 1 lần)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# -------------------- SIDEBAR --------------------
st.sidebar.title("🔧 Menu")
page = st.sidebar.radio("Chọn trang:", ["📊 Data", "🎯 Model", "🔮 Prediction", "📂 Upload CSV","⚡ Real-time Simulation", "🧠 Big Data Integration"])

# -------------------- DATA PAGE --------------------

if page == "📊 Data":
    st.title("📊 Data Overview")
    

   # --- Lọc danh sách máy hỏng ---
    st.subheader("⚠️ Danh sách thiết bị bị hỏng")

    failed_machines = data[data["Machine failure"] == 1]

    if failed_machines.empty:
        st.success("✅ Không có thiết bị nào hỏng trong dữ liệu.")
    else:
        st.write(f"📌 Tổng cộng có **{len(failed_machines)}** thiết bị bị hỏng.")
        st.dataframe(failed_machines)


    # ---- Bộ lọc dữ liệu ----
    st.subheader("🔍 Bộ lọc dữ liệu")
    types = st.multiselect("Chọn loại thiết bị (Type)", options=data["Type"].unique(), default=data["Type"].unique())
    failure_filter = st.selectbox("Chọn tình trạng máy", options=["Tất cả", "Hoạt động (0)", "Hỏng (1)"])
    
    filtered_data = data[data["Type"].isin(types)]
    if failure_filter == "Hoạt động (0)":
        filtered_data = filtered_data[filtered_data["Machine failure"] == 0]
    elif failure_filter == "Hỏng (1)":
        filtered_data = filtered_data[filtered_data["Machine failure"] == 1]

    st.dataframe(filtered_data.head())
    
    # ---- Biểu đồ trực quan ----
    st.subheader("📈 Visualization")
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.pie(filtered_data, names="Machine failure", 
                      title="Tỷ lệ tình trạng máy (0: OK, 1: Hỏng)")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(filtered_data, x="Rotational speed [rpm]", nbins=30, 
                            title="Phân bố tốc độ quay")
        st.plotly_chart(fig2, use_container_width=True)

    # Boxplot: Torque vs Tình trạng máy
    st.subheader("📊 Phân tích nâng cao")
    fig3 = px.box(filtered_data, x="Machine failure", y="Torque [Nm]", 
                  title="Phân bố Torque theo tình trạng máy")
    st.plotly_chart(fig3, use_container_width=True)

    # Scatter: Nhiệt độ quy trình vs Tốc độ quay
    fig4 = px.scatter(filtered_data, x="Process temperature [K]", y="Rotational speed [rpm]",
                      color="Machine failure", 
                      title="Quan hệ giữa Nhiệt độ quy trình và Tốc độ quay (theo tình trạng máy)")
    st.plotly_chart(fig4, use_container_width=True)

    # Ma trận tương quan
    corr = filtered_data.corr(numeric_only=True)
    fig5 = px.imshow(corr, text_auto=True, 
                     title="🔗 Ma trận tương quan giữa các đặc trưng")
    st.plotly_chart(fig5, use_container_width=True)
    
    # ---- Insight tự động ----
    st.subheader("💡 Nhận xét nhanh")
    failure_rate = filtered_data["Machine failure"].mean()
    avg_speed = filtered_data["Rotational speed [rpm]"].mean()
    st.info(f"Tỉ lệ máy hỏng trong dữ liệu đã lọc là **{failure_rate*100:.2f}%**. "
            f"Tốc độ quay trung bình khoảng **{avg_speed:.1f} rpm**.")
    # Biểu đồ so sánh nhiệt độ không khí giữa máy hỏng và bình thường
    fig_fail = px.box(data, x="Machine failure", y="Air temperature [K]",
                      title="So sánh nhiệt độ không khí: Máy hỏng (1) vs Bình thường (0)")
    st.plotly_chart(fig_fail, use_container_width=True)

# -------------------- MODEL PAGE --------------------
elif page == "🎯 Model":
    st.title("🎯 Model Performance")

    # Predict
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # --- Metrics hiển thị đẹp ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric("✅ Accuracy", f"{acc:.2f}")
    with col2:
        st.metric("📊 Sample Size", f"{len(y_test)}")

    # --- Classification Report dạng bảng ---
    st.subheader("📋 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.background_gradient(cmap='Blues').format("{:.2f}"))

    # --- Confusion Matrix ---
    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # --- Feature Importance (nếu model hỗ trợ) ---
    if hasattr(model, "feature_importances_"):
        st.subheader("📈 Feature Importance")
        fi = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.bar_chart(fi.set_index("Feature"))

# -------------------- PREDICTION PAGE --------------------
elif page == "🔮 Prediction":
    st.title("🔮 Dự đoán tình trạng máy (nhập tay dữ liệu)")

    st.write("""
    Nhập các thông số cảm biến bên dưới để hệ thống:
    1️⃣ Dự đoán tình trạng máy theo **mô hình Machine Learning**  
    2️⃣ Kiểm tra song song các **điều kiện kỹ thuật bất thường**
    """)

    # --- Chia giao diện 2 cột ---
    col1, col2 = st.columns(2)

    with col1:
        air_temp = st.number_input("🌡 Nhiệt độ không khí [K]", 280.0, 330.0, 300.0, 0.1)
        process_temp = st.number_input("🔥 Nhiệt độ quy trình [K]", 290.0, 350.0, 305.0, 0.1)
        rotational_speed = st.number_input("⚙️ Tốc độ quay [rpm]", 500, 4000, 1500)
    with col2:
        torque = st.number_input("🌀 Mô-men xoắn [Nm]", 0.0, 100.0, 40.0, 0.1)
        tool_wear = st.number_input("⏳ Độ mòn dụng cụ [min]", 0.0, 300.0, 100.0, 1.0)
        product_type = st.selectbox("🔩 Loại sản phẩm", ["L", "M", "H"])

    # --- Tạo DataFrame input ---
    df_input = pd.DataFrame([{
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type": product_type
    }])

    # --- Mã hóa cột Type (giống lúc huấn luyện) ---
    df_encoded = pd.get_dummies(df_input, columns=["Type"])
    df_encoded = df_encoded.reindex(columns=X.columns, fill_value=0)

    # --- Nút dự đoán ---
    if st.button("🚀 Dự đoán tình trạng máy"):
        st.divider()
        st.subheader("📊 Kết quả dự đoán:")

        # ====== 1️⃣ DỰ ĐOÁN THEO MÔ HÌNH ML ======
        prediction = model.predict(df_encoded)[0]
        prob = model.predict_proba(df_encoded)[0][1]

        if prediction == 0:
            st.success(f"✅ ML: Máy hoạt động bình thường\n\nXác suất hỏng: {prob:.2%}")
        else:
            st.error(f"⚠️ ML: Máy có nguy cơ hỏng!\n\nXác suất hỏng: {prob:.2%}")
       #----- Lưu model ra file (chỉ 1 lần) -----
            # ====== LƯU KẾT QUẢ VÀO MongoDB ======
        try:
            from pymongo import MongoClient
            client = MongoClient("mongodb://localhost:27017/")
            db = client["predictive_maintenance"]
            collection = db["predictions"]

            record = {
                "Type": product_type,
                "AirTemp": air_temp,
                "ProcessTemp": process_temp,
                "Speed": rotational_speed,
                "Torque": torque,
                "ToolWear": tool_wear,
                "Prediction": int(prediction),
                "Probability": float(prob)
            }

            collection.insert_one(record)
            st.info("✅ Kết quả đã được lưu vào MongoDB.")
        except Exception as e:
            st.warning(f"⚠️ Không thể kết nối MongoDB: {e}")

        # ====== 2️⃣ KIỂM TRA NGƯỠNG KỸ THUẬT ======
        alerts = []
        if process_temp > 315:
            alerts.append("🔥 Process temperature > 315 K → Quá nhiệt")
        if torque > 60:
            alerts.append("🌀 Torque > 60 Nm → Mô-men quá tải")
        if rotational_speed < 1200 or rotational_speed > 3000:
            alerts.append("⚙️ Rotational speed ngoài khoảng [1200, 3000] rpm → Quay không ổn định")
        if tool_wear > 200:
            alerts.append("⏳ Tool wear > 200 min → Dụng cụ mòn nhiều")
        if (process_temp - air_temp) > 15:
            alerts.append("🌡 Process temp - Air temp > 15 K → Nhiệt sinh ra quá lớn")

        st.divider()
        st.subheader("⚙️ Kiểm tra kỹ thuật:")

        if len(alerts) == 0:
            st.success("✅ Không phát hiện điều kiện bất thường – Thiết bị hoạt động ổn định.")
        else:
            st.error(f"⚠️ Phát hiện {len(alerts)} điều kiện bất thường:")
            for a in alerts:
                st.write(f"- {a}")
            if len(alerts) >= 2:
                st.warning("🚨 Kết luận: Thiết bị có **nguy cơ hỏng cao (≥2 điều kiện)**.")
            else:
                st.info("ℹ️ Cảnh báo nhẹ: Cần theo dõi thêm.")

        # ====== 3️⃣ HIỂN THỊ LẠI DỮ LIỆU NHẬP ======
        st.divider()
        st.subheader("🧾 Dữ liệu nhập:")
        st.dataframe(df_input)



# -------------------- UPLOAD CSV PAGE --------------------
elif page == "📂 Upload CSV":
    st.title("📂 Upload dữ liệu mới để dự đoán")

    uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"])
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)

        st.subheader("📊 Dữ liệu bạn vừa upload")
        st.dataframe(new_data.head())

        try:
            # Xử lý encode giống dữ liệu train
            if "Type" in new_data.columns:
                new_data = pd.get_dummies(new_data, columns=["Type"])

            # Căn chỉnh cột
            new_data = new_data.reindex(columns=X.columns, fill_value=0)

            preds = model.predict(new_data)
            probs = model.predict_proba(new_data)[:, 1]

            new_data["Prediction"] = preds
            new_data["Failure Probability"] = probs

            st.subheader("🔮 Kết quả dự đoán")
            st.dataframe(new_data)

            st.download_button(
                "📥 Tải kết quả về",
                new_data.to_csv(index=False).encode("utf-8"),
                file_name="prediction_results.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"⚠️ Lỗi khi dự đoán: {e}")

# -------------------- REAL-TIME SIMULATION PAGE --------------------
elif page == "⚡ Real-time Simulation":
    st.title("⚡ xử lý dữ liệu thời gian thực")

    st.write("Trang này mô phỏng dữ liệu cảm biến gửi về hệ thống theo thời gian thực, "
             "và mô hình Machine Learning sẽ dự đoán tình trạng thiết bị ngay khi dữ liệu được nhận.")

    # Cấu hình tốc độ mô phỏng
    delay = st.slider("⏱ Thời gian giữa các mẫu (giây):", 0.5, 5.0, 1.0)

    # Tạo vùng hiển thị realtime
    placeholder = st.empty()

    # Giới hạn demo (ví dụ 100 dòng đầu tiên để chạy mượt)
    subset = data.sample(100, random_state=42).reset_index(drop=True)

    for i in range(len(subset)):
        row = subset.iloc[i:i+1]
        # Chuẩn hóa input giống khi train
        input_row = row.drop(columns=["Machine failure", "Product ID", "UDI"], errors="ignore")

        # Encode nếu có cột Type
        if "Type" in input_row.columns:
            input_row = pd.get_dummies(input_row, columns=["Type"])
        input_row = input_row.reindex(columns=X.columns, fill_value=0)
        pred = model.predict(input_row)[0]
        prob = model.predict_proba(input_row)[0][1]

        # Hiển thị cập nhật realtime
        with placeholder.container():
            st.subheader(f"🕒 Dữ liệu cảm biến #{i+1}")
            st.dataframe(row)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Xác suất hỏng", f"{prob*100:.2f}%")
            with col2:
                if pred == 1:
                    st.error("⚠️ Cảnh báo: Thiết bị có nguy cơ hỏng!")
                else:
                    st.success("✅ Thiết bị hoạt động bình thường")

            # Biểu đồ dòng thời gian dự đoán
            if i > 1:
                subset_plot = subset.iloc[:i+1].copy()
                subset_plot["Predicted Failure"] = model.predict(
                    subset_plot.drop(columns=["Machine failure", "Product ID", "UDI"], errors="ignore")
                    .pipe(lambda d: pd.get_dummies(d, columns=["Type"]) if "Type" in d.columns else d)
                    .reindex(columns=X.columns, fill_value=0)
                )
                fig = px.line(subset_plot, 
                              y="Rotational speed [rpm]", 
                              color="Predicted Failure",
                              title="📈 Biểu đồ tốc độ quay (theo dự đoán realtime)")
                st.plotly_chart(fig, use_container_width=True)
        time.sleep(delay)
        
# -------------------- BIG DATA DEMO PAGE --------------------
elif page == "🧠 Big Data Integration":
    st.title("🧠 Tích hợp Big Data trong dự án dự báo sự cố thiết bị công nghiệp")

    st.divider()
    st.header("🧩 DEMO THỰC HÀNH BIG DATA (MÔ PHỎNG TRONG STREAMLIT)")

   # 1️⃣ DEMO HDFS - Lưu trữ dữ liệu cảm biến phân tán
    # ==========================
    st.divider()
    st.subheader("💾 Demo 1 – Lưu trữ dữ liệu cảm biến trên HDFS ")

    if st.button("▶️ Chạy mô phỏng HDFS"):
        df_demo = data.head(25)[["Air temperature [K]", "Process temperature [K]", "Torque [Nm]", 
                                 "Rotational speed [rpm]", "Tool wear [min]", "Machine failure", "Type"]]
        block_size = 5
        num_blocks = len(df_demo) // block_size
        st.write(f"📦 Giả lập chia dữ liệu cảm biến thành {num_blocks} khối (mỗi khối = {block_size} dòng).")
        for i in range(0, len(df_demo), block_size):
            block = df_demo.iloc[i:i+block_size]
            st.info(f"🗂️ Block {i//block_size + 1} – lưu tại Node-{i//block_size + 1}")
            st.dataframe(block)
            time.sleep(0.5)
        st.success("✅ Dữ liệu đã được lưu phân tán như HDFS thực tế.")

    # ==========================
    #  SPARK / MAPREDUCE - Xử lý song song dữ liệu công nghiệp
    # ==========================
    st.divider()
    st.subheader("⚙️ – Xử lý song song (Spark / MapReduce)")

    if st.button("🚀 Chạy Spark Job"):
        df_demo = data.sample(15, random_state=1)[["Type", "Torque [Nm]", "Rotational speed [rpm]", "Machine failure"]]
        st.write("📊 Dữ liệu đầu vào:")
        st.dataframe(df_demo)
        partitions = np.array_split(df_demo, 3)
        torque_avg_by_part = []
        for i, part in enumerate(partitions):
            avg_torque = part["Torque [Nm]"].mean()
            st.info(f"🧩 Partition {i+1}: Torque trung bình = {avg_torque:.2f} Nm")
            torque_avg_by_part.append(avg_torque)
            time.sleep(1)

        # Giai đoạn Reduce
        final_result = np.mean(torque_avg_by_part)
        st.success(f"🎯 Kết quả (Reduce): Torque trung bình toàn bộ hệ thống = {final_result:.2f} Nm")

        # ==========================
    # 3️⃣ DEMO MONGODB - Ghi dữ liệu từ file CSV vào MongoDB 
    st.divider()
    st.subheader("🧠 Demo 3 – Mô phỏng cảm biến gửi dữ liệu từ file CSV vào MongoDB")
    uploaded_csv = st.file_uploader("📂 Chọn file dữ liệu cảm biến (CSV)", type=["csv"])
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        cols = ["Air temperature [K]", "Process temperature [K]", "Torque [Nm]",
                "Rotational speed [rpm]", "Tool wear [min]"]
        df = df[cols].copy()
        max_rows = st.slider("🔢 Số dòng mô phỏng", 5, 200, 30)
        delay = st.slider("⏱ Thời gian giữa mỗi dòng (giây)", 0.5, 3.0, 1.0)
        df = df.head(max_rows).reset_index(drop=True)
        if "mongo_data" not in st.session_state:
            st.session_state.mongo_data = []
        if st.button("▶️ Bắt đầu ghi dữ liệu vào MongoDB (mô phỏng)"):
            st.info("📡 Đang gửi dữ liệu từ CSV lên MongoDB (giả lập)...")
            placeholder = st.empty()
            st.session_state.mongo_data = []  # reset dữ liệu cũ
            for i in range(len(df)):
                row = df.iloc[i]
                # Kiểm tra tình trạng thiết bị
                air = row["Air temperature [K]"]
                proc = row["Process temperature [K]"]
                torque = row["Torque [Nm]"]
                speed = row["Rotational speed [rpm]"]
                wear = row["Tool wear [min]"]
                status = "OK"
                if torque > 60 or (proc - air) > 15 or wear > 200:
                    status = "Warning"
                if torque > 70 or proc > 320:
                    status = "Failure"
                new_record = {
                    "AirTemp": air,
                    "ProcTemp": proc,
                    "Torque": torque,
                    "Speed": speed,
                    "Wear": wear,
                    "Status": status,
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                st.session_state.mongo_data.append(new_record)
                # Hiển thị dữ liệu mới
                with placeholder.container():
                    st.write(f"📨 Dòng {i+1}/{len(df)} gửi vào MongoDB ({status})")
                    st.dataframe(pd.DataFrame(st.session_state.mongo_data).tail(5))
                    st.progress((i+1) / len(df))
                time.sleep(delay)
            st.success(f"✅ Hoàn tất mô phỏng gửi {len(df)} dòng dữ liệu vào MongoDB!")
            st.balloons()

        # Hiển thị toàn bộ dữ liệu MongoDB mô phỏng
        st.write("📋 Dữ liệu cảm biến trong MongoDB (mô phỏng):")
        st.dataframe(pd.DataFrame(st.session_state.mongo_data))

    # ==========================
    # 4️⃣ DEMO CLOUD - Lưu trữ dữ liệu & hiển thị trên nền tảng đám mây
    # ==========================
    st.divider()
    st.subheader("☁️ Demo 4 – Upload và phân tích dữ liệu trên Cloud (giả lập)")

    uploaded_file = st.file_uploader("📂 Chọn file cảm biến CSV để upload (mô phỏng lưu lên S3 / Azure Blob)", type=["csv"])
    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.success("✅ Dữ liệu đã được upload lên Cloud (giả lập).")
        st.write("📊 5 dòng đầu dữ liệu:")
        st.dataframe(df_uploaded.head())

        # Tính toán mô phỏng xử lý trên Cloud
        avg_temp = df_uploaded["Process temperature [K]"].mean()
        avg_torque = df_uploaded["Torque [Nm]"].mean()
        fail_rate = df_uploaded["Machine failure"].mean() * 100

        st.info(f"🌡 Nhiệt độ trung bình (Process): {avg_temp:.2f} K")
        st.info(f"🌀 Torque trung bình: {avg_torque:.2f} Nm")
        st.warning(f"⚠️ Tỷ lệ thiết bị hỏng trong file: {fail_rate:.2f}%")
