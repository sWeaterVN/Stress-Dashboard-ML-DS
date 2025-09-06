import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from io import StringIO
from typing import List, Tuple, Optional

# ML (baseline)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Cấu hình trang
st.set_page_config(
    page_title="Phân tích Stress Học Sinh - Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Phân tích dữ liệu Stress của học sinh THPT")
st.caption("Dashboard EDA + Baseline ML — tải CSV của bạn vào để bắt đầu.")

# Load data
st.sidebar.header("1) Tải dữ liệu")
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

@st.cache_data(show_spinner=False)
def load_dataframe(file) -> pd.DataFrame:
    if file is not None:
        return pd.read_csv(file)
    try:
        return pd.read_csv("StressLevelDataset.csv")
    except Exception:
        return pd.DataFrame()

df = load_dataframe(uploaded_file)

if df.empty:
    st.warning("⚠️ No file .csv selected:(.")
    st.stop()

# Fix type error
df.columns = [c.strip() for c in df.columns]

# Main setting
st.sidebar.header("2) Cấu hình")
all_cols = df.columns.tolist()

guess_targets = [c for c in all_cols if c.lower() in ("stress_level", "stress_type", "target")]
target_col = st.sidebar.selectbox(
    "Chọn cột Target (mức độ stress)",
    guess_targets if guess_targets else all_cols,
    index=0 if guess_targets else len(all_cols) - 1
)


# Quick analysis bar (maybe)
psych_cols = [c for c in all_cols if c in ["anxiety_level","self_esteem","mental_health_history","depression"]]
phys_cols  = [c for c in all_cols if c in ["headache","blood_pressure","sleep_quality","breathing_problem"]]
env_cols   = [c for c in all_cols if c in ["noise_level","living_conditions","safety","basic_needs"]]
acad_cols  = [c for c in all_cols if c in ["academic_performance","study_load","teacher_student_relationship","future_career_concerns"]]
soc_cols   = [c for c in all_cols if c in ["social_support","peer_pressure","extracurricular_activities","bullying"]]

suggested_numeric = psych_cols + phys_cols + env_cols + acad_cols + soc_cols
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != target_col]

# Protect
def _safe_default_list(default_list, options, max_defaults=None):
    if not options:
        return []
    if not default_list:
        return []
    filtered = [c for c in default_list if c in options]
    if filtered:
        return filtered
    n = max_defaults if max_defaults is not None else min(len(options), len(default_list))
    return options[:n]

# Infomation
st.subheader("Tổng quan dữ liệu")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Số lượng đối tượng khảo sát", f"{len(df):,}")
c2.metric("Số lượng yếu tố gây ảnh hưởng", f"{len(df.columns):,}")
n_missing = int(df.isna().sum().sum())
c3.metric("Giá trị thiếu (tổng)", f"{n_missing:,}")
unique_target = df[target_col].nunique(dropna=True)
c4.metric("Số lớp Target", unique_target)

with st.expander("Xem nhanh dữ liệu (5 đối tượng đầu)"):
    st.dataframe(df.head())

with st.expander("Thông tin/kiểu dữ liệu"):
    buf = StringIO()
    df.info(buf=buf)
    st.text(buf.getvalue())

# Tabs (1-6)
tab_overview, tab_distribution, tab_corr, tab_compare, tab_model, tab_extra = st.tabs([
    "Tổng quan",
    "Phân phối",
    "Tương quan",
    "So sánh theo Stress",
    "Mô hình (Baseline)",
    "Nâng cao"
])

# TAB 1: Overview
with tab_overview:
    st.markdown("### Biểu đồ tổng quan")

    # Target
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Phân phối Target (Pie chart)**")
        target_counts = df[target_col].value_counts(dropna=False)
        fig, ax = plt.subplots()
        ax.pie(target_counts.values, labels=target_counts.index.astype(str), autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)

    with col2:
        st.markdown("**Phân phối Target (Bar chart)**")
        fig, ax = plt.subplots()
        target_counts.plot(kind="bar", ax=ax)
        ax.set_xlabel(target_col)
        ax.set_ylabel("Số lượng")
        st.pyplot(fig)
    
    st.markdown("---")
    st.markdown("### Thống kê: Trung bình, Độ lệch chuẩn, Min, Max, Median (numeric columns)")
    num_cols_for_stats = df.select_dtypes(include=[np.number]).columns.tolist()
    # Nếu muốn loại trừ target (nếu target là số), dùng: [c for c in num_cols_for_stats if c != target_col]
    if len(num_cols_for_stats) == 0:
        st.info("Không có cột numeric để tính thống kê.")
    else:
        means = df[num_cols_for_stats].mean(numeric_only=True).round(2)
        stds = df[num_cols_for_stats].std(numeric_only=True).round(2)
        mins = df[num_cols_for_stats].min(numeric_only=True).round(2)
        maxs = df[num_cols_for_stats].max(numeric_only=True).round(2)
        medians = df[num_cols_for_stats].median(numeric_only=True).round(2)
        stats_df = pd.DataFrame({
            "mean": means,
            "std": stds,
            "min": mins,
            "median": medians,
            "max": maxs
        })
        st.dataframe(stats_df.style.format("{:.2f}"))

    st.markdown("---")
    st.markdown("**Thiếu dữ liệu theo cột**")
    missing = df.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots()
    missing.plot(kind="bar", ax=ax)
    ax.set_ylabel("Số giá trị thiếu")
    st.pyplot(fig)

# TAB 2: Distribution
with tab_distribution:
    st.markdown("### Phân phối các biến")
    dist_options = numeric_cols if numeric_cols else all_cols
    dist_default = _safe_default_list((psych_cols[:2] + phys_cols[:2]) if (psych_cols or phys_cols) else (numeric_cols[:4] if len(numeric_cols)>=4 else numeric_cols), dist_options, max_defaults=4)
    selected_cols = st.multiselect("Chọn cột để vẽ Histogram / KDE", dist_options, default=dist_default)
    colA, colB = st.columns(2)

    for i, col in enumerate(selected_cols):
        if i % 2 == 0:
            with colA:
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Histogram + KDE: {col}")
                st.pyplot(fig)
        else:
            with colB:
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Histogram + KDE: {col}")
                st.pyplot(fig)

    st.markdown("---")
    # Biểu đồ (CDF) + Area chart
    st.markdown("### Biểu đồ đường & 'sóng' (Area)")
    line_col = st.selectbox("Chọn một biến số để vẽ CDF (line) & Area", numeric_cols if numeric_cols else all_cols, index=0)
    series = df[line_col].dropna().sort_values()
    if len(series) > 0:
        cdf = np.arange(1, len(series)+1) / len(series)
        fig, ax = plt.subplots()
        ax.plot(series.values, cdf)
        ax.set_xlabel(line_col)
        ax.set_ylabel("CDF")
        ax.set_title(f"CDF (Line chart) của {line_col}")
        st.pyplot(fig)

        st.markdown("**Area chart (dạng 'sóng') của phân phối giá trị**")
        vc = series.value_counts().sort_index()
        area_df = pd.DataFrame({line_col: vc.index, "count": vc.values}).set_index(line_col)
        st.area_chart(area_df)

# TAB 3: Tương quan
with tab_corr:
    st.markdown("### Ma trận tương quan (Pearson)")
    num_df = df.select_dtypes(include=[np.number])
    if target_col in num_df.columns:
        corr = num_df.corr()
    else:
        corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    if target_col in num_df.columns:
        st.markdown("---")
        st.markdown("### Mức tương quan từng biến với Target")
        tgt_corr = corr[target_col].drop(target_col).sort_values(ascending=False)
        fig, ax = plt.subplots()
        tgt_corr.plot(kind="bar", ax=ax)
        ax.set_ylabel("Hệ số tương quan với Target")
        st.pyplot(fig)

# TAB 4: Compare
with tab_compare:
    st.markdown("### So sánh phân phối theo Target")
    compare_options = numeric_cols if numeric_cols else all_cols
    compare_default = _safe_default_list(["sleep_quality","anxiety_level","academic_performance"] if "sleep_quality" in df.columns else (numeric_cols[:3] if len(numeric_cols)>=3 else numeric_cols), compare_options, max_defaults=3)
    compare_cols = st.multiselect("Chọn biến số để so sánh theo Target", compare_options, default=compare_default)
    plot_type = st.radio("Chọn loại biểu đồ", ["Boxplot", "Violin"], horizontal=True)

    for col in compare_cols:
        fig, ax = plt.subplots()
        if plot_type == "Boxplot":
            sns.boxplot(x=df[target_col].astype(str), y=df[col], ax=ax)
            ax.set_xlabel(target_col)
        else:
            sns.violinplot(x=df[target_col].astype(str), y=df[col], ax=ax)
            ax.set_xlabel(target_col)
        ax.set_title(f"{col} theo {target_col}")
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### Trung bình theo nhóm Target (Bar chart)")
    group_means = df.groupby(target_col)[compare_cols].mean(numeric_only=True)
    st.dataframe(group_means.style.format("{:.2f}"))
    fig, ax = plt.subplots(figsize=(min(10, 3*len(compare_cols)), 5))
    group_means.plot(kind="bar", ax=ax)
    ax.set_ylabel("Giá trị trung bình")
    ax.set_title("So sánh trung bình theo Target")
    st.pyplot(fig)

# TAB 5: Model (Baseline)
with tab_model:
    st.markdown("### Thiết lập & Huấn luyện + Xuất Submission")

    y_raw = df[target_col].copy()
    X_raw = df.drop(columns=[target_col])

    X = X_raw.select_dtypes(include=[np.number]).copy()

    if X.shape[1] == 0:
        st.error("Không có biến số (numeric) nào cho mô hình baseline. Hãy mã hóa dữ liệu trước.")
        st.stop()

    y = y_raw.copy()
    if y.dtype == object or y.nunique() <= 10:
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        class_names = list(le.classes_)
    else:
        class_names = sorted([str(c) for c in y_raw.dropna().unique()])[:10]

    test_size = st.slider("Tỷ lệ test size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("random_state", value=42, step=1)

    stratify_opt = y if pd.Series(y).nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_opt
    )

    model_type = st.selectbox("Chọn mô hình", ["RandomForest", "LogisticRegression"], index=0)

    if model_type == "RandomForest":
        n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
        max_depth = st.slider("max_depth (0 = None)", 0, 30, 0, 1)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=None if max_depth == 0 else max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        C = st.slider("C (Regularization strength)", 0.01, 5.0, 1.0, 0.01)
        clf = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("lr", LogisticRegression(max_iter=1000, multi_class="auto", C=C))
        ])

    train_btn = st.button("Huấn luyện mô hình")
    if train_btn:

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"Độ chính xác (accuracy) trên tập test: **{acc:.3f}**")

        # Confusion matrix
        st.markdown("#### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix")
        ax.set_xticklabels(class_names if len(class_names)==cm.shape[1] else np.arange(cm.shape[1]))
        ax.set_yticklabels(class_names if len(class_names)==cm.shape[0] else np.arange(cm.shape[0]))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
        st.pyplot(fig)

        # Báo cáo phân loại
        st.markdown("#### Báo cáo phân loại")
        try:
            report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
            rep_df = pd.DataFrame(report).T
            st.dataframe(rep_df.style.format("{:.3f}"))
        except Exception:
            st.text(classification_report(y_test, y_pred, zero_division=0))

        # Feature importance
        st.markdown("#### Tầm quan trọng đặc trưng (Feature Importance)")
        if model_type == "RandomForest":
            importances = clf.feature_importances_
            imp_df = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values("importance", ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(8, min(0.4*len(imp_df), 8)))
            ax.barh(imp_df["feature"], imp_df["importance"])
            ax.invert_yaxis()
            ax.set_xlabel("Importance")
            ax.set_title("Top đặc trưng quan trọng (RF)")
            st.pyplot(fig)
            st.dataframe(imp_df)
        else:
            try:
                lr = clf.named_steps["lr"]
                coefs = np.abs(lr.coef_)
                mean_coefs = coefs.mean(axis=0)
                imp_df = pd.DataFrame({"feature": X.columns, "coef_abs_mean": mean_coefs}).sort_values("coef_abs_mean", ascending=False).head(20)
                fig, ax = plt.subplots(figsize=(8, min(0.4*len(imp_df), 8)))
                ax.barh(imp_df["feature"], imp_df["coef_abs_mean"])
                ax.invert_yaxis()
                ax.set_xlabel("|coef| (mean over classes)")
                ax.set_title("Top đặc trưng quan trọng (LogReg)")
                st.pyplot(fig)
                st.dataframe(imp_df)
            except Exception as e:
                st.info("Không thể tính hệ số quan trọng cho Logistic Regression.")
                st.text(str(e))

        try:
            st.markdown("### Xuất file submission.csv")
            clf.fit(X, y)

            test_df = pd.read_csv("test.csv") 
            X_real_test = test_df.drop(columns=["id"])
            y_real_pred = clf.predict(X_real_test)

            submission = pd.DataFrame({
                "id": test_df["id"],
                "stress_level": y_real_pred
            })
            submission.to_csv("submission.csv", index=False)
    
            st.success("Tạo submit thành công!")
            st.dataframe(submission.head())

        except Exception as e:
            # Maybe cant find test.csv :D
            st.error("Error")
            st.text(str(e))

# TAB 6: Extra
with tab_extra:
    st.markdown("### Biểu đồ nâng cao")
    st.markdown("**Pairplot (nhìn nhanh mối liên hệ cặp)** — chọn vài biến để vẽ:")
    pair_options = numeric_cols if numeric_cols else all_cols
    pair_default = _safe_default_list((psych_cols[:2] + phys_cols[:2] + [target_col]) if len(psych_cols)>=2 else (numeric_cols[:5] if len(numeric_cols)>=5 else numeric_cols), pair_options, max_defaults=6)
    sel_for_pair = st.multiselect("Chọn 3–6 biến số", pair_options, default=pair_default)

    if 2 <= len(sel_for_pair) <= 8:
        try:
            plot_df = df[sel_for_pair].copy()
            hue_col = None
            if target_col in sel_for_pair:
                hue_col = target_col
            fig = sns.pairplot(plot_df, hue=hue_col, diag_kind="hist")
            st.pyplot(fig)
        except Exception as e:
            st.info("Không vẽ được pairplot.")
            st.text(str(e))

    st.markdown("---")
    st.markdown("**Bar chart nhóm (nhiều cột)** — chọn các biến số để so sánh trung bình theo Target.")
    multi_options = numeric_cols if numeric_cols else all_cols
    multi_default = _safe_default_list((psych_cols + phys_cols)[:6] if (psych_cols or phys_cols) else numeric_cols[:6], multi_options, max_defaults=6)
    multi_cols = st.multiselect("Chọn biến để so sánh (2–10 biến)", multi_options, default=multi_default)
    if len(multi_cols) >= 2:
        mean_df = df.groupby(target_col)[multi_cols].mean(numeric_only=True)
        fig, ax = plt.subplots(figsize=(min(12, 1.5*len(multi_cols)), 5))
        mean_df.plot(kind="bar", ax=ax)
        ax.set_title("So sánh trung bình theo Target - Nhiều biến")
        st.pyplot(fig)
st.markdown("---")




