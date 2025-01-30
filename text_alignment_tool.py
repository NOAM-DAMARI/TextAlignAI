import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import streaml
it as st
import os

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def deskew_image(image_path):
    image = preprocess_image(image_path)
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def enhance_text(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    enhanced = cv2.equalizeHist(image)
    return enhanced

def convert_images_to_pdf(image_paths, output_pdf):
    images = [Image.open(img).convert('RGB') for img in image_paths]
    images[0].save(output_pdf, save_all=True, append_images=images[1:])

def convert_pdf_to_images(pdf_path, output_folder):
    images = convert_from_path(pdf_path)
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths

# ן¿½ן¿½ן¿½ן¿½ ן¿½ן¿½ן¿½ן¿½ן¿½ ן¿½ן¿½ Streamlit
st.title("ן¿½ן¿½ן¿½ן¿½ן¿½ ן¿½ן¿½ן¿½ן¿½ן¿½ן¿½ ן¿½ן¿½ן¿½ן¿½ ן¿½ן¿½ן¿½ן¿½ן¿½ן¿½ן¿½ ן¿½-PDF")
uploaded_file = st.file_uploader("ן¿½ן¿½ן¿½ן¿½ ן¿½ן¿½ן¿½ן¿½ן¿½ ן¿½ן¿½ PDF", type=["jpg", "png", "pdf"])

if uploaded_file:
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    if uploaded_file.name.endswith(".pdf"):
        st.write("ן¿½ן¿½ן¿½ן¿½ PDF ן¿½ן¿½ן¿½ן¿½ן¿½ן¿½ן¿½...")
        image_paths = convert_pdf_to_images(file_path, "temp")
    else:
        image_paths = [file_path]
    
    for img_path in image_paths:
        st.image(img_path, caption="ן¿½ן¿½ן¿½ן¿½ן¿½ ן¿½ן¿½ן¿½ן¿½ן¿½ן¿½", use_column_width=True)
        aligned = deskew_image(img_path)
        enhanced = enhance_text(img_path)
        
        aligned_path = img_path.replace(".png", "_aligned.png").replace(".jpg", "_aligned.jpg")
        enhanced_path = img_path.replace(".png", "_enhanced.png").replace(".jpg", "_enhanced.jpg")
        
        cv2.imwrite(aligned_path, aligned)
        cv2.imwrite(enhanced_path, enhanced)
        
        st.image(aligned_path, caption="ן¿½ן¿½ן¿½ן¿½ן¿½ ן¿½ן¿½ן¿½ן¿½ן¿½ן¿½", use_column_width=True)
        st.image(enhanced_path, caption="ן¿½ן¿½ן¿½ן¿½ן¿½ ן¿½ן¿½ן¿½ן¿½ן¿½ן¿½", use_column_width=True)
        
        st.download_button("ן¿½ן¿½ן¿½ן¿½ ן¿½ן¿½ן¿½ן¿½ן¿½ ן¿½ן¿½ן¿½ן¿½ן¿½ן¿½", data=open(aligned_path, "rb").read(), file_name="aligned.png")
        st.download_button("ן¿½ן¿½ן¿½ן¿½ ן¿½ן¿½ן¿½ן¿½ן¿½ ן¿½ן¿½ן¿½ן¿½ן¿½ן¿½", data=open(enhanced_path, "rb").read(), file_name="enhanced.png")
