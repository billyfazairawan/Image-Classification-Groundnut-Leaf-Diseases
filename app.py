import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Memuat model yang sudah dilatih 
model = load_model('best_model.keras')

# Nama kelas penyakit 
class_names = ['Early leaf spot', 'Healthy leaf', 'Late leaf spot', 'Rosette', 'Rust']

# Fungsi untuk memprediksi penyakit dari gambar
def predict_disease(img):
    img = img.resize((256, 256))  # Resize gambar sesuai input model 
    img = np.array(img.convert('RGB'))  # Pastikan gambar RGB
    img = img / 255.0  # Normalisasi gambar
    img = np.expand_dims(img, axis=0)  # Menambahkan dimensi batch
    
    pred = model.predict(img)
    return np.argmax(pred, axis=1)[0], np.max(pred)  # Mengembalikan label dan probabilitas

# Antarmuka pengguna Streamlit
st.title("Prediksi Penyakit Daun Kacang Tanah")
st.write("Unggah gambar daun untuk memprediksi penyakitnya.")

# Input gambar dari pengguna
uploaded_file = st.file_uploader("Pilih Gambar Daun Kacang Tanah", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)
    
    # Prediksi ketika tombol ditekan
    if st.button("Prediksi Penyakit"):
        # Prediksi kelas penyakit
        label_idx, confidence = predict_disease(img)
        label = class_names[label_idx]
        
        # Menampilkan hasil prediksi
        st.write(f"Penyakit yang terdeteksi: {label}")
        st.write(f"Kepercayaan: {confidence * 100:.2f}%")

       # Memberikan penjelasan lebih lanjut jika diperlukan
        if label == "Late leaf spot":
            st.write("Daun ini terdeteksi memiliki penyakit Late leaf spot. Segera lakukan penanganan!")
        elif label == "Early leaf spot":
            st.write("Daun ini terdeteksi memiliki penyakit Early leaf spot. Segera lakukan penanganan!")
        elif label == "Rosette":
            st.write("Daun ini terdeteksi memiliki penyakit Rosette. Segera lakukan penanganan!")
        elif label == "Rust":
            st.write("Daun ini terdeteksi memiliki penyakit Rust. Segera lakukan penanganan!")
        else:
            st.write("Daun ini sehat, tidak terdeteksi penyakit.")

            
