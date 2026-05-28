import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Config halaman Streamlit
st.set_page_config(page_title="Sistem Clustering Mahasiswa", layout="wide")

st.title("📊 Sistem Clustering Kemampuan Akademik Mahasiswa")
st.write("Mengelompokkan mahasiswa berdasarkan Kehadiran, Tugas, UTS, dan UAS menggunakan algoritme K-Means.")

# --- 1. MEMUAT DATA ---
# Di tugas nyata, Anda bisa menggunakan st.file_uploader. 
# Untuk demo ini, kita masukkan data dummy yang Anda berikan langsung ke dalam kode.
@st.cache_data
def load_initial_data():
    data = {
        'No': list(range(1, 51)),
        'NIM': [221000 + i for i in range(1, 51)],
        'Nama': [
            "Andi Saputra", "Budi Santoso", "Citra Lestari", "Deni Pratama", "Eka Putri",
            "Fajar Nugroho", "Gina Maharani", "Hendra Wijaya", "Intan Permata", "Joko Susilo",
            "Kiki Amelia", "Luthfi Ramadhan", "Maya Sari", "Nanda Saputri", "Oki Prabowo",
            "Putra Aditya", "Qori Aisyah", "Rendi Kurniawan", "Siska Melati", "Tono Hartono",
            "Umi Kalsum", "Vina Oktavia", "Wahyu Firmansyah", "Xenia Putri", "Yusuf Hidayat",
            "Zahra Nabila", "Agus Salim", "Bella Safitri", "Cahyo Nugraha", "Dina Puspita",
            "Eko Riyanto", "Fitri Handayani", "Galih Prasetyo", "Hana Maharani", "Irfan Maulana",
            "Jihan Apriliya", "Kevin Alvaro", "Lina Marlina", "M Rizky", "Nia Ramadhani",
            "Ovan Saputra", "Prita Lestari", "Qomaruddin", "Rara Oktaviani", "Sandi Permana",
            "Tiara Anjani", "Umar Faruq", "Vicky Fernando", "Wulan Sari", "Yoga Pratama"
        ],
        'Kehadiran': [90,75,95,60,85,70,98,55,88,78,92,65,83,58,89,72,96,62,84,68,91,57,80,94,66,87,59,82,74,97,63,86,71,93,56,81,69,99,64,88,73,95,61,84,67,90,54,79,96,65],
        'Tugas': [88,70,92,65,80,72,95,60,86,75,90,68,81,55,87,74,94,64,82,66,89,59,78,93,67,85,61,84,73,96,62,88,69,91,58,80,71,97,63,89,75,94,60,85,68,91,56,77,95,66],
        'UTS': [85,72,90,58,82,68,94,57,84,73,91,64,79,54,86,70,95,60,83,67,88,56,77,92,65,86,60,81,72,95,61,87,70,90,55,79,68,98,62,87,74,93,59,83,66,89,55,78,94,64],
        'UAS': [87,74,94,62,84,69,97,58,85,76,93,66,82,57,88,71,96,61,85,69,90,58,79,95,64,88,62,83,75,98,64,89,72,92,57,82,70,99,65,90,76,96,63,86,69,92,57,80,97,67]
    }
    return pd.DataFrame(data)

df = load_initial_data()

# Menampilkan data awal
if st.checkbox("Tampilkan Data Awal Mahasiswa (50 Baris)"):
    st.dataframe(df, use_container_width=True)

# --- 2. PREPROCESSING ---
# Memilih fitur numerik yang akan digunakan untuk klustering
features = ['Kehadiran', 'Tugas', 'UTS', 'UAS']
X = df[features]

# Normalisasi data menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 3. PROSES CLUSTERING (K-MEANS) ---
# Menentukan jumlah K tetap yaitu 3 sesuai kebutuhan tugas
k_terpilih = 3
kmeans = KMeans(n_clusters=k_terpilih, random_state=42)
df['Cluster_Label'] = kmeans.fit_predict(X_scaled)

# --- 4. PEMETAAN NAMA KLUSTER ---
# Mencari tahu rata-rata total nilai per kluster untuk melabeli secara otomatis secara logis
cluster_means = df.groupby('Cluster_Label')[features].mean().mean(axis=1)
sorted_clusters = cluster_means.sort_values(ascending=False).index

# Membuat kamus pemetaan (Kluster nilai tertinggi = Berprestasi, dst)
cluster_mapping = {
    sorted_clusters[0]: 'Berprestasi',
    sorted_clusters[1]: 'Menengah',
    sorted_clusters[2]: 'Perlu Pembinaan'
}
df['Status Mahasiswa'] = df['Cluster_Label'].map(cluster_mapping)

# --- 5. TAMPILKAN HASIL ---
st.header("🏆 Hasil Pengelompokkan Mahasiswa")

# Statistik jumlah mahasiswa per kategori
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Mahasiswa Berprestasi", f"{len(df[df['Status Mahasiswa']=='Berprestasi'])} Orang")
with col2:
    st.metric("Mahasiswa Menengah", f"{len(df[df['Status Mahasiswa']=='Menengah'])} Orang")
with col3:
    st.metric("Perlu Pembinaan", f"{len(df[df['Status Mahasiswa']=='Perlu Pembinaan'])} Orang")

# Dropdown filter hasil kluster
status_filter = st.selectbox("Filter berdasarkan status:", ["Semua", "Berprestasi", "Menengah", "Perlu Pembinaan"])
if status_filter != "Semua":
    df_filtered = df[df['Status Mahasiswa'] == status_filter]
else:
    df_filtered = df

st.dataframe(df_filtered[['NIM', 'Nama', 'Kehadiran', 'Tugas', 'UTS', 'UAS', 'Status Mahasiswa']], use_container_width=True)

# --- 6. VISUALISASI ---
st.header("📈 Visualisasi Kluster")
fig, ax = plt.subplots(figsize=(10, 5))

# Plot sebaran menggunakan kombinasi Tugas vs UAS sebagai representasi visual
sns.scatterplot(
    data=df, 
    x='Tugas', 
    y='UAS', 
    hue='Status Mahasiswa', 
    palette={'Berprestasi': 'green', 'Menengah': 'blue', 'Perlu Pembinaan': 'red'},
    s=100,
    ax=ax
)
ax.set_title("Sebaran Kelompok Mahasiswa (Berdasarkan Nilai Tugas vs UAS)")
st.pyplot(fig)

# Menampilkan analisis rata-rata nilai per kelompok
st.subheader("📊 Rata-rata Nilai per Kelompok")
st.dataframe(df.groupby('Status Mahasiswa')[features].mean())