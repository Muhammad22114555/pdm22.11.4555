import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import io # Untuk membaca file yang diunggah
import tempfile # Untuk menyimpan file h5 sementara

# Judul aplikasi Streamlit
st.title("Aplikasi Prediksi Harga Properti Melbourne dengan LSTM (Per Bulan)")
st.subheader("Muhammad_22.11.4555")
st.write("Aplikasi ini memerlukan **dataset properti (.csv)** dan **model LSTM yang sudah dilatih (.h5)** untuk melakukan prediksi dan menampilkan visualisasi bulanan.")

# --- Bagian Unggah File ---
uploaded_data_file = st.file_uploader("1. Unggah file CSV Dataset Properti (misal: Property Sales of Melbourne City.csv)", type="csv")
uploaded_model_file = st.file_uploader("2. Unggah file model LSTM (.h5)", type="h5")

# --- Fungsi Preprocessing dan Pembuatan Sekuens ---
@st.cache_data # Cache data agar tidak diulang setiap kali ada interaksi UI
def load_and_preprocess_data(data_file):
    """Memuat dan melakukan preprocessing pada data."""
    if data_file is not None:
        df = pd.read_csv(data_file)

        columns_to_drop = [
            'Suburb', 'Address', 'Type', 'Method', 'SellerG', 'Distance', 'Postcode',
            'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'YearBuilt',
            'CouncilArea', 'Lattitude', 'Longtitude', 'Regionname', 'Propertycount',
            'Unnamed: 0'
        ]
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore') # Tambahkan errors='ignore'

        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df.sort_values('Date', inplace=True)
        df.dropna(subset=['Price'], inplace=True)
        df.fillna(method='ffill', inplace=True)

        original_dates = df['Date'].copy()

        scaler = MinMaxScaler()
        df_scaled = df[['Rooms', 'Price']].copy()
        df_scaled[['Rooms', 'Price']] = scaler.fit_transform(df_scaled[['Rooms', 'Price']])

        return df_scaled, original_dates, scaler
    return None, None, None

def create_sequences(data, seq_length):
    """Membuat sekuens untuk input LSTM."""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, :])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

# --- Proses Utama Aplikasi ---
if uploaded_data_file is None:
    st.info("Silakan unggah file CSV dataset properti.")
elif uploaded_model_file is None:
    st.info("Silakan unggah file model LSTM (.h5).")
else:
    st.write("File dataset dan model berhasil diunggah. Memproses data dan model...")

    # Memuat dan Preprocessing Data
    df_scaled, original_dates, scaler = load_and_preprocess_data(uploaded_data_file)

    if df_scaled is not None:
        seq_length = 30 # Sesuai dengan model yang dilatih
        data_for_sequences = df_scaled[['Rooms', 'Price']].values
        X, y = create_sequences(data_for_sequences, seq_length)

        # Pembagian data latih/uji (hanya untuk mendapatkan indeks test_dates)
        split = int(0.8 * len(X))
        X_test = X[split:]
        y_test = y[split:]

        # Memuat Model
        try:
            # Simpan file model ke file sementara dan load tanpa compile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp:
                tmp.write(uploaded_model_file.read())
                model_path = tmp.name
            model = load_model(model_path, compile=False)
            st.success("Model .h5 berhasil dimuat!")
        except Exception as e:
            st.error(f"Gagal memuat model .h5: {e}")
            st.stop() # Hentikan eksekusi jika model gagal dimuat

        # Melakukan Prediksi
        st.write("Melakukan prediksi dengan model...")
        y_pred = model.predict(X_test).flatten()

        # Mengembalikan transformasi skala prediksi dan nilai aktual ke skala harga asli
        dummy_rooms_test = np.zeros((len(y_test), 1))
        y_test_original = scaler.inverse_transform(np.concatenate((dummy_rooms_test, y_test.reshape(-1, 1)), axis=1))[:, 1]

        dummy_rooms_pred = np.zeros((len(y_pred), 1))
        y_pred_original = scaler.inverse_transform(np.concatenate((dummy_rooms_pred, y_pred.reshape(-1, 1)), axis=1))[:, 1]

        # Mendapatkan tanggal yang sesuai dengan set pengujian
        test_dates_index_start = split + seq_length
        test_dates = original_dates.iloc[test_dates_index_start : test_dates_index_start + len(y_test_original)]

        # Membuat DataFrame untuk plotting
        plot_df = pd.DataFrame({
            'Date': test_dates,
            'Actual Price': y_test_original,
            'Predicted Price': y_pred_original
        })

        # Filter data untuk tahun 2016 dan 2017
        plot_df_filtered = plot_df[(plot_df['Date'].dt.year >= 2016) & (plot_df['Date'].dt.year <= 2017)].copy()

        # --- Visualisasi Statis: Per Bulan ---
        processed_plot_df = pd.DataFrame()
        x_label = 'Bulan (1-24)' # Sesuai dengan kode pelatihan
        plot_title = 'Harga Properti Aktual vs. Prediksi (Bulan 1-24)' # Sesuai dengan kode pelatihan

        # Mengelompokkan per bulan untuk menghitung rata-rata
        plot_df_filtered['Month_Year'] = plot_df_filtered['Date'].dt.to_period('M') # Agregasi per bulan
        processed_plot_df = plot_df_filtered.groupby('Month_Year').agg({
            'Actual Price': 'mean',
            'Predicted Price': 'mean'
        }).reset_index()
        processed_plot_df = processed_plot_df.sort_values('Month_Year')
        processed_plot_df['Sequence'] = np.arange(1, len(processed_plot_df) + 1)
        st.write(f"Jumlah bulan historis yang tersedia untuk visualisasi (2016-2017): {len(processed_plot_df)}")


        # --- Prediksi Langkah Berikutnya ---
        # Prediksi ini akan tetap ditampilkan sebagai teks, tetapi TIDAK dimasukkan ke dalam plot
        if not df_scaled.empty:
            last_seq_data = data_for_sequences[-seq_length:]
            last_seq_data = last_seq_data.reshape(1, seq_length, data_for_sequences.shape[1])

            next_period_prediction_scaled = model.predict(last_seq_data).flatten()[0]

            dummy_rooms_next_pred = np.zeros((1, 1))
            next_period_prediction_original = scaler.inverse_transform(np.concatenate((dummy_rooms_next_pred, np.array([[next_period_prediction_scaled]])), axis=1))[:, 1][0]

            # Menentukan label periode berikutnya (ini hanya untuk informasi teks)
            if not processed_plot_df.empty and 'Month_Year' in processed_plot_df.columns:
                last_period = processed_plot_df['Month_Year'].iloc[-1]
                # Menambahkan satu bulan ke periode terakhir untuk label prediksi
                next_period_label = (last_period + 1).strftime('%Y-%m')
            else:
                next_period_label = pd.Timestamp.now().to_period('M') + 1
                next_period_label = next_period_label.strftime('%Y-%m')


            st.write(f"Prediksi harga untuk periode selanjutnya ({next_period_label}): **Rp{next_period_prediction_original:,.2f}**")
        else:
            st.warning("Data tidak cukup untuk melakukan prediksi periode berikutnya.")


        # --- Visualisasi ---
        st.subheader("Visualisasi Harga Properti Aktual vs. Prediksi")
        fig, ax = plt.subplots(figsize=(15, 7))
        sns.lineplot(x='Sequence', y='Actual Price', data=processed_plot_df, label='Harga Aktual Rata-rata Bulanan', color='blue', marker='o', ax=ax)
        sns.lineplot(x='Sequence', y='Predicted Price', data=processed_plot_df, label='Harga Prediksi Rata-rata Bulanan', color='red', linestyle='--', marker='x', ax=ax)

        ax.set_title(plot_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Harga')
        
        # Mengatur label sumbu X agar lebih informatif seperti kode pelatihan
        # Menampilkan setiap bulan pada sumbu x jika jumlahnya tidak terlalu banyak
        if len(processed_plot_df) <= 24: # Jika 24 bulan atau kurang, tampilkan setiap label
            tick_labels = [str(p) for p in processed_plot_df['Month_Year']]
            ax.set_xticks(processed_plot_df['Sequence'])
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        else: # Jika lebih banyak, biarkan matplotlib menentukan interval yang baik
            ax.set_xticks(np.arange(1, len(processed_plot_df) + 1, max(1, len(processed_plot_df) // 10))) # Contoh: tampilkan setiap 10 bulan
            ax.tick_params(axis='x', rotation=45, ha='right') # Rotasi untuk kerapian

        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        st.pyplot(fig) # Menampilkan plot di Streamlit

        st.write("Visualisasi berhasil ditampilkan.")
        st.write("Mohon maaf pak masih belum maksimal dan masih sangat overfitting")

    else:
        st.warning("Gagal memuat atau memproses dataset. Pastikan format file CSV benar.")
