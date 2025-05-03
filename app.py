import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from transformers import pipeline
import torch # Diperlukan oleh transformers
import timm # Pastikan terinstal
import os
import time # Untuk simulasi loading atau jeda kecil

# --- Konfigurasi Dasar & Pemuatan Model ---

st.set_page_config(page_title="Deteksi Objek", layout="wide")

# Cache model agar tidak dimuat ulang setiap interaksi
@st.cache_resource
def load_object_detector():
    """Memuat pipeline deteksi objek."""
    try:
        print("Mencoba memuat model di GPU (jika tersedia)...")
        detector = pipeline("object-detection", model="facebook/detr-resnet-50")
        print("Model dimuat di device:", detector.device)
        return detector
    except Exception as e_gpu:
        print(f"Gagal memuat di GPU: {e_gpu}")
        try:
            print("Mencoba memuat model di CPU...")
            detector = pipeline("object-detection", model="facebook/detr-resnet-50", device=-1)
            print("Model berhasil dimuat di CPU.")
            return detector
        except Exception as e_cpu:
            print(f"Gagal memuat model di CPU juga: {e_cpu}")
            st.error(f"Gagal memuat model deteksi objek: {e_cpu}", icon="🚨")
            return None

object_detector = load_object_detector()

# --- Fungsi Helper ---
# (Fungsi load_image_from_url, find_font, draw_detections tetap sama seperti sebelumnya)
# ... (salin fungsi helper dari versi sebelumnya di sini) ...
def load_image_from_url(url):
    """Mengunduh dan membuka gambar dari URL."""
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except requests.exceptions.RequestException as e:
        st.error(f"Error mengunduh gambar: {e}", icon=" L")
        return None
    except Exception as e:
        st.error(f"Error memproses gambar dari URL: {e}", icon="️")
        return None

def find_font(font_size=15):
    """Mencari font yang tersedia di sistem."""
    font_paths = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", # Umum di Linux/Colab
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",              # Umum di Linux/Colab
        "arial.ttf"                                                    # Windows/Fallback
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except IOError:
                continue
    print("Font spesifik tidak ditemukan, menggunakan font default.")
    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
        return ImageFont.load_default()


def draw_detections(image, detections, confidence_threshold=0.8):
    """Menggambar kotak pembatas dan label pada gambar."""
    if image is None:
        return None, []

    draw = ImageDraw.Draw(image)
    font = find_font(font_size=15)
    detected_objects_info = []

    for detection in detections:
        score = detection['score']
        if score >= confidence_threshold:
            label = detection['label']
            box = detection['box']
            xmin = int(box['xmin'])
            ymin = int(box['ymin'])
            xmax = int(box['xmax'])
            ymax = int(box['ymax'])

            detected_objects_info.append({
                "Label": label,
                "Confidence": round(score, 3),
                "Box (xmin, ymin, xmax, ymax)": (xmin, ymin, xmax, ymax)
            })

            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red", width=3)
            text = f"{label}: {score:.2f}"

            try:
                text_bbox = draw.textbbox((xmin, ymin), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                try:
                    text_width, text_height = font.getsize(text)
                except AttributeError:
                     text_width = len(text) * 8
                     text_height = 12

            text_y = ymin - text_height - 5
            if text_y < 5:
                text_y = ymin + 5
            text_x = xmin + 2

            bg_rect = (text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2)
            draw.rectangle(bg_rect, fill="red")
            draw.text((text_x, text_y), text, fill="white", font=font)

    return image, detected_objects_info

# --- Antarmuka Streamlit ---

st.title("️ Deteksi Objek pada Gambar")
st.write("Gunakan model FB DETR (via Transformers) untuk menemukan objek dalam gambar.")
st.write("---")

input_method = st.radio(
    "Pilih metode input gambar:",
    ("Masukkan URL", "Unggah File"),
    horizontal=True
)

input_image = None
image_source_info = ""

if input_method == "Masukkan URL":
    image_url = st.text_input("URL Gambar:", "https://storage.googleapis.com/petbacker/images/blog/2017/dog-and-cat-cover.jpg")
    if image_url:
        input_image = load_image_from_url(image_url)
        if input_image:
            image_source_info = f"dari URL: {image_url[:50]}..."

else: # Unggah File
    uploaded_file = st.file_uploader("Pilih file gambar:", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        try:
            input_image = Image.open(uploaded_file).convert("RGB")
            image_source_info = f"dari file: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Gagal membuka file gambar: {e}", icon="️")

st.write("---")

confidence_thresh = st.slider(
    "Threshold Kepercayaan Minimum:",
    min_value=0.1,
    max_value=1.0,
    value=0.8,
    step=0.05
)
st.write(f"Hanya objek dengan confidence >= **{confidence_thresh:.2f}** yang akan ditampilkan.")
st.write("---")

if object_detector is None:
    st.warning("Model tidak dapat dimuat. Aplikasi tidak dapat berjalan.", icon="")
elif input_image is not None:
    st.subheader(f"Hasil Deteksi {image_source_info}")

    col1, col2 = st.columns(2)

    with col1:
        # === PERUBAHAN DI SINI ===
        st.image(input_image, caption="Gambar Asli", use_container_width=True)

    with st.spinner('⏳ Menganalisis gambar...'):
        start_time = time.time()
        detections = object_detector(input_image.convert("RGB"))
        end_time = time.time()

    image_with_boxes, detected_details = draw_detections(
        input_image.copy(),
        detections,
        confidence_threshold=confidence_thresh
    )

    with col2:
        # === PERUBAHAN DI SINI ===
        st.image(image_with_boxes, caption=f"Gambar dengan Deteksi (Threshold: {confidence_thresh:.2f})", use_container_width=True)

    st.write(f"Waktu proses deteksi: {end_time - start_time:.2f} detik")

    st.subheader("Detail Objek Terdeteksi")
    if detected_details:
        st.dataframe(detected_details)
    else:
        st.info(f"Tidak ada objek yang terdeteksi dengan confidence >= {confidence_thresh:.2f}")

elif not image_source_info:
    st.info("Silakan masukkan URL gambar atau unggah file untuk memulai.")

st.write("---")
st.caption("Dibuat dengan Streamlit & Hugging Face Transformers")
print("Aplikasi Streamlit siap.")
