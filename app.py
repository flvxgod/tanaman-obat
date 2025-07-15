import streamlit as st
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
import pickle
from tensorflow.keras.applications.efficientnet import preprocess_input
import gdown
import os

# --- Download model dan label jika belum ada ---
MODEL_PATH = "model/model.h5"
LABEL_PATH = "model/class_labels.pkl"

if not os.path.exists("model"):
    os.makedirs("model")

if not os.path.exists(MODEL_PATH):
    gdown.download("https://drive.google.com/uc?id=1bKENSmi5HJZMj9W-PsNJS7boImbCNNAf", MODEL_PATH, quiet=False)

if not os.path.exists(LABEL_PATH):
    gdown.download("https://drive.google.com/uc?id=1-Q-qrQfl99rhtnW1AAw7S3SYytmxmRPa", LABEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model('model/model.h5')

# Load class labels
with open('model/class_labels.pkl', 'rb') as f:
    class_labels = pickle.load(f)

# Deskripsi tiap tanaman
descriptions = {
    "jambu": {
        "manfaat": "Mengandung flavonoid, tanin, minyak atsiri, dan alkaloid yang efektif sebagai anti-diare, antibakteri, serta membantu meredakan sakit perut.",
        "cara_olah": "Dapat ditumbuk dan dicampur gula merah, direbus lalu diminum, atau dikapsulkan untuk konsumsi harian saat sakit.",
        "sumber": "Yusnaini Rambe et al., Jurnal ADAM IPTS, Vol. 1 No. 2, 2022"
    },
    "jahe": {
        "manfaat": "Mengandung gingerol, flavonoid, dan minyak atsiri yang bermanfaat sebagai antiinflamasi, meredakan mual, meningkatkan daya tahan tubuh, dan mencegah berbagai penyakit seperti batuk, masuk angin, bahkan kanker.",
        "cara_olah": "Rimpang jahe dicuci bersih, dirajang kecil, lalu dikeringkan di bawah sinar matahari. Setelah kering dapat disimpan dalam bentuk rajangan atau dihaluskan menjadi serbuk simplisia.",
        "sumber": "Asni Amin & Risda Waris, Edukasi Penggunaan Dan Cara Pengolahan Rimpang Jahe Sebagai Bahan Baku Obat Tradisional Di Desa Gunung Silanu,  Kabupaten Jeneponto, Sulawesi Selatan, 2023"
    },
    "lidah buaya": {
        "manfaat": "Mengandung vitamin, enzim, dan senyawa aktif yang berfungsi sebagai antiinflamasi, antiseptik, antivirus, antibakteri, analgesik, mempercepat penyembuhan luka, membantu pencernaan, menurunkan kadar gula darah, mengontrol tekanan darah, dan meningkatkan sistem imun terhadap kanker.",
        "cara_olah": "Kupas kulit lidah buaya, potong dagingnya lalu rendam dalam air kapur sirih dan garam selama 30 menit untuk menghilangkan lendir. Cuci bersih, rebus dengan air, gula, dan daun pandan hingga mendidih sebentar. Minuman harus disimpan dalam kulkas karena tanpa bahan pengawet (tahan 3 hari di lemari es, 7 hari di freezer).",
        "sumber": "Mutia Lina Dewi, Pengolahan Aloe Vera (Lidah Buaya) sebagai Minuman Sehat , 2022"
    },
    "mentimun": {
        "manfaat": "Mengandung gizi tinggi, membantu menjaga kesehatan tubuh, memperlancar buang air kecil, mencegah dehidrasi, mengobati darah tinggi, mengatasi keracunan saat hamil, menjaga kesegaran tubuh, serta bermanfaat untuk kecantikan kulit.",
        "cara_olah": "Mentimun dapat diolah menjadi minuman dengan cara memeras sarinya, lalu dicampur madu dan jeruk nipis. Bisa juga dicampur yoghurt cair dan sedikit garam untuk menjaga kesegaran tubuh dan perlindungan dari sinar matahari.",
        "sumber": "Andi Rusdayani Amin, Mengenal Budidaya Mentimun Melalui Pemanfaatan Media Informasi, 2015"
    },
    "temulawak": {
        "manfaat": "Mengandung kurkuminoid, flavonoid, fenol, dan minyak atsiri yang berfungsi sebagai antibakteri, antiinflamasi, antispasmodik, antioksidan, membantu mengatasi gangguan pencernaan, penyakit hati, batu empedu, dan meningkatkan metabolisme tubuh.",
        "cara_olah": "Rimpang temulawak dikeringkan lalu diekstraksi menggunakan metode maserasi (perendaman dengan etanol) atau sokletasi (dengan pemanasan). Metode maserasi lebih disarankan karena menghasilkan rendemen dan kadar kurkumin yang tinggi tanpa merusak zat aktif.",
        "sumber": "Lidvina Niken Yasacaxena et al., Ekstraksi Rimpang Temulawak (Curcuma xanthorrhiza Roxb.) dan Aktivitas Sebagai Antibakteri, 2023"
    },
    "singkong": {
        "manfaat": "Mengandung karbohidrat, protein, dan pati resisten yang bermanfaat untuk pencernaan, mengatasi diare, diabetes, infeksi kulit, serta membantu mengatasi kerontokan rambut dan kemandulan. Daun singkong juga kaya vitamin C dan A, serta kalsium yang baik untuk kesehatan tubuh.",
        "cara_olah": "Singkong dapat diolah dengan cara direbus, digoreng, atau dijadikan produk olahan seperti gethuk krispi. Prosesnya dimulai dengan merebus singkong, ditumbuk, dibentuk, lalu digoreng hingga renyah. Olahan ini dapat divariasikan dengan berbagai rasa dan dipasarkan melalui media sosial atau marketplace.",
        "sumber": "Fatqu Rois et al., Pengoptimalan Pengolahan Singkong Menjadi Produk Pangan Dalam Meningkatkan Pendapatan Masyarakat Desa, 2023"
    },
    "lengkuas": {
        "manfaat": "Mengandung minyak atsiri, flavonoid, fenol, dan terpenoid yang berfungsi sebagai antijamur, antibakteri, antioksidan, antitumor, dan antikanker. Efektif menghambat pertumbuhan cendawan penyebab penyakit tanaman dan digunakan juga untuk pengobatan kulit seperti panu, kolera, dan eksem.",
        "cara_olah": "Rimpang lengkuas dipotong tipis, dikeringkan pada suhu 40°C, lalu dihaluskan dan diekstrak menggunakan pelarut (misalnya etanol). Ekstrak ini kemudian dicampurkan ke media seperti PDA untuk uji penghambatan jamur, atau digunakan dalam bentuk cair sebagai pestisida nabati.",
        "sumber": "Ismail Suaib et al., Efektifitas Ekstrak Rimpang Lengkuas Dalam Menghambat Aktifitas Cendawan Oncobasidium theobremae Secara In-vitro, 2016"
    },
    "kale": {
        "manfaat": "Mengandung kalium, mangan, zat besi, magnesium, serat, protein, dan kalsium. Bermanfaat untuk mencegah anemia, menjaga kesehatan mata, hati, otak, jantung, serta menstabilkan kolesterol, mengurangi sariawan dan radang usus.",
        "cara_olah": "Kangkung organik diolah menjadi makanan seperti keripik, kerupuk, dan jari-jari dengan berbagai rasa (original, balado, pedas, barbeque), serta minuman herbal bernama VIKS DRINK. Kangkung ditanam menggunakan media tanam ramah lingkungan dari sampah anorganik dan pupuk organik buatan sendiri.",
        "sumber": "Alif Yuanita Kartini & Shofa Robbani, Pemanfaatan Tanaman Kangkung dan Sampah Lingkungan Sebagai Upaya Peningkatan Ekonomi Masyarakat Desa Ngumpakdalem di Masa Pandemi Covid 19, 2022"
    },
    "kacang panjang": {
        "manfaat": "Mengandung thiamin (vitamin B1) dan serat yang berfungsi membantu menurunkan kadar glukosa darah, memperbaiki kerja reseptor insulin, dan memperlambat penyerapan glukosa. Efektif digunakan sebagai terapi tambahan untuk pasien Diabetes Mellitus tipe 2.",
        "cara_olah": "Kacang panjang dijus dengan dosis 100 gram per 50 kg berat badan, dikonsumsi dua kali sehari (pagi dan sore) setelah makan selama 14 hari. Jus dikonsumsi langsung tanpa bahan tambahan.",
        "sumber": "Harmayetty et al., Jus Kacang Panjang (Vigna Sinensis L.) Menurunkan Kadar Glukosa Darah Pasien Diabetes Mellitus, 2009"
    },
    "bayam": {
        "manfaat": "Mengandung zat besi, vitamin A, B, C, K, E, serta mineral seperti kalsium, fosfor, kalium, magnesium, dan serat. Bermanfaat untuk mencegah anemia, meningkatkan daya tahan tubuh, menjaga kesehatan mata, memperbaiki pencernaan, mencegah sembelit, osteoporosis, hipertensi, dan kanker saluran pencernaan.",
        "cara_olah": "Bayam dapat diolah menjadi keripik bayam dengan cara mencuci daun bayam, mencampurkannya ke dalam adonan tepung beras dan tapioka yang telah dibumbui, lalu digoreng hingga kecokelatan. Pengolahan ini membuat bayam lebih disukai anak-anak sebagai camilan bergizi tinggi.",
        "sumber": "Astrini Padapi et al., Pengolahan Daun Bayam Hijau (Amarhantus tricolor ) guna Meningkatkan Tingkat Konsumsi Masyarakat, 2022"
    },
    "bawang merah": {
        "manfaat": "Mengandung flavonoid, quercetin, allicin, dan senyawa antioksidan yang berfungsi sebagai antimikroba, antiinflamasi, serta membantu meningkatkan imunitas tubuh, melawan infeksi, meredakan peradangan, dan membantu detoksifikasi.",
        "cara_olah": "Bawang merah diiris tipis dan direbus bersama bahan herbal seperti jahe, sereh, kayu manis, cengkeh, dan lemon. Rebus selama 5–10 menit hingga aroma dan zat aktifnya keluar. Saring dan tambahkan madu atau gula batu untuk rasa. Disajikan hangat sebagai minuman herbal peningkat daya tahan tubuh.",
        "sumber": "Nova Elok Mardliyana et al., Pelatihan Pemanfaatan Bawang Merah (Allium Cepa L) Menjadi Minuman Herbal Untuk Peningkatan Imunitas Tubuh, 2023"
    },
}

# --- UI Streamlit ---
st.title("Klasifikasi Tanaman Obat")

uploaded_file = st.file_uploader("Upload gambar daun", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Gambar yang diunggah", use_column_width=True)

    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    pred_idx = np.argmax(prediction[0])
    confidence = prediction[0][pred_idx] * 100
    label = [k for k, v in class_labels.items() if v == pred_idx][0]

    st.subheader(f"Prediksi: {label} ({confidence:.2f}%)")

    if label in descriptions:
        st.markdown("**Manfaat:** " + descriptions[label]["manfaat"])
        st.markdown("**Cara Olah:** " + descriptions[label]["cara_olah"])
        st.markdown("*Sumber:* " + descriptions[label]["sumber"])