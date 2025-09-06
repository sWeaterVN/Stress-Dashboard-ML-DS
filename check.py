import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Phân tích dữ liệu Stress của học sinh")

df = pd.read_csv("StressLevelDataset.csv")

st.subheader("Xem trước dữ liệu")
st.write(df.head())

st.subheader("Thông tin dữ liệu")
st.write("Kích thước dữ liệu:", df.shape)
st.write(df.describe())

st.subheader("Phân phối Stress Level")
fig1, ax1 = plt.subplots()
df['stress_level'].value_counts().plot(kind="bar", color="skyblue", ax=ax1)
ax1.set_title("Phân phối Stress Level")
st.pyplot(fig1)

st.subheader("Ma trận tương quan")
fig2, ax2 = plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".2f", ax=ax2)
st.pyplot(fig2)

st.subheader("Phân tích mối quan hệ")
x_axis = st.selectbox("Chọn biến X (categorical/target):", df.columns, index=len(df.columns)-1)
y_axis = st.selectbox("Chọn biến Y (numeric):", df.columns, index=6)

fig3, ax3 = plt.subplots()
sns.boxplot(x=x_axis, y=y_axis, data=df, ax=ax3, palette="Set2")
ax3.set_title(f"{y_axis} theo {x_axis}")
st.pyplot(fig3)
