import pandas as pd
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load scaler dan model
scaler = pickle.load(open('scaler.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Ambil file dari request
    file = request.files['file']
    data = pd.read_excel(file)

    # Kolom yang diperlukan
    selected_columns = [
        'Bayi usia kurang dari 6 bulan mendapat air susu ibu (ASI) eksklusif',
        'Anak usia 6-23 bulan yang mendapat Makanan Pendamping Air Susu Ibu (MP-ASI)',
        'Anak berusia di bawah lima tahun (balita) gizi buruk yang mendapat pelayanan tata laksana gizi buruk',
        'Anak berusia di bawah lima tahun (balita) yang dipantau pertumbuhan dan perkembangannya',
        'Anak berusia di bawah lima tahun (balita) gizi kurang yang mendapat tambahan asupan gizi',
        'Balita yang memperoleh imunisasi dasar lengkap',
        'Ibu hamil Kurang Energi Kronik (KEK) yang mendapatkan tambahan asupan gizi',
        'Ibu hamil yang mengonsumsi Tablet Tambah Darah (TTD) minimal 90 tablet selama masa kehamilan',
        'Kelompok Keluarga Penerima Manfaat (KPM) Program Keluarga Harapan (PKH) yang mengikuti Pertemuan Peningkatan Kemampuan Keluarga (P2K2) dengan modul kesehatan dan gizi',
        'Keluarga Penerima Manfaat (KPM) dengan ibu hamil, ibu menyusui, dan baduta yang menerima variasi bantuan pangan selain beras dan telur'
    ]

    # Pilih kolom yang diperlukan
    data = data[selected_columns]

    # Scaling data
    scaled_data = scaler.transform(data)

    # Prediksi
    predictions = model.predict(scaled_data)

    # Unscale hasil prediksi
    unscaled_predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

    # Kembalikan hasil sebagai JSON
    return jsonify({'predictions': unscaled_predictions.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
