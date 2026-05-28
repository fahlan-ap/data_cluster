import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Sistem Clustering Mahasiswa", layout="wide")

st.title("📊 Sistem Clustering Kemampuan Akademik Mahasiswa")
st.write("Unggah file data mahasiswa Anda (CSV/Excel) untuk mengelompokkan mereka menggunakan algoritme K-Means.")

# --- 1. FITUR UPLOAD DATA INDEPENDEN ---
st.header("📂 Unggah Dataset Mahasiswa")
uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.success("🎉 Data berhasil diunggah!")
        
        # Menampilkan data awal
        if st.checkbox("Tampilkan Data Awal Mahasiswa"):
            st.dataframe(df, use_container_width=True)

        # --- 2. VALIDASI KOLOM & PREPROCESSING ---
        features = ['Kehadiran', 'Tugas', 'UTS', 'UAS']
        
        if all(col in df.columns for col in features):
            X = df[features]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # --- 3. PROSES CLUSTERING (K-MEANS) ---
            k_terpilih = 3
            kmeans = KMeans(n_clusters=k_terpilih, random_state=42)
            df['Cluster_Label'] = kmeans.fit_predict(X_scaled)

            # --- 4. PEMETAAN NAMA KLUSTER ---
            cluster_means = df.groupby('Cluster_Label')[features].mean().mean(axis=1)
            sorted_clusters = cluster_means.sort_values(ascending=False).index

            cluster_mapping = {
                sorted_clusters[0]: 'Berprestasi',
                sorted_clusters[1]: 'Menengah',
                sorted_clusters[2]: 'Perlu Pembinaan'
            }
            df['Status Mahasiswa'] = df['Cluster_Label'].map(cluster_mapping)

            # --- 5. TAMPILKAN HASIL ---
            st.header("🏆 Hasil Pengelompokkan Mahasiswa")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mahasiswa Berprestasi", f"{len(df[df['Status Mahasiswa']=='Berprestasi'])} Orang")
            with col2:
                st.metric("Mahasiswa Menengah", f"{len(df[df['Status Mahasiswa']=='Menengah'])} Orang")
            with col3:
                st.metric("Perlu Pembinaan", f"{len(df[df['Status Mahasiswa']=='Perlu Pembinaan'])} Orang")

            status_filter = st.selectbox("Filter berdasarkan status:", ["Semua", "Berprestasi", "Menengah", "Perlu Pembinaan"])
            if status_filter != "Semua":
                df_filtered = df[df['Status Mahasiswa'] == status_filter]
            else:
                df_filtered = df

            kolom_tampil = [col for col in ['NIM', 'Nama', 'Kehadiran', 'Tugas', 'UTS', 'UAS', 'Status Mahasiswa'] if col in df.columns]
            st.dataframe(df_filtered[kolom_tampil], use_container_width=True)

            # --- 6. VISUALISASI DIAGRAM BATANG PERSENTASE (COMPACT VERSION) ---
            st.header("📊 Visualisasi Persentase Kluster")
            
            persentase_data = df['Status Mahasiswa'].value_counts(normalize=True) * 100
            
            urutan_kategori = ['Berprestasi', 'Menengah', 'Perlu Pembinaan']
            persentase_data = persentase_data.reindex(urutan_kategori).fillna(0)

            persentase_data = persentase_data.iloc[::-1]

            fig, ax = plt.subplots(figsize=(5, 1.5))
            colors = ['#e74c3c', '#3498db', '#2ecc71']
            
            bars = ax.barh(persentase_data.index, persentase_data.values, color=colors, edgecolor='black', height=0.5)
            
            for bar in bars:
                width = bar.get_width()
                ax.annotate(f' {width:.1f}%',
                            xy=(width, bar.get_y() + bar.get_height() / 2),
                            xytext=(3, 0),
                            textcoords="offset points",
                            ha='left', va='center', fontsize=10, fontweight='bold')

            ax.set_title("Distribusi Kelompok Mahasiswa (%)", fontsize=11, fontweight='bold', pad=10)
            
            ax.get_xaxis().set_visible(False)
            
            sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
            
            ax.tick_params(axis='y', labelsize=10, length=0)
            
            st.pyplot(fig)

            st.subheader("📊 Rata-rata Nilai per Kelompok")
            st.dataframe(df.groupby('Status Mahasiswa')[features].mean())
            
        else:
            st.error(f"Format file salah! Pastikan file Anda memiliki kolom wajib: {', '.join(features)}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")

else:
    st.info("💡 Silakan unggah file Excel/CSV data mahasiswa terlebih dahulu di atas untuk memulai analisis.")