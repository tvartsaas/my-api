from flask import Flask, request, jsonify
from flask_cors import CORS  # Importação para CORS
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Permite requisições de todas as origens

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Função de extração de cores
def extract_colors(image_path, n_colors=10):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((100, 100))  # Reduz a imagem
    pixels = np.array(image).reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors, random_state=0)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_

    label_counts = np.bincount(labels)
    total_count = len(labels)
    percentages = (label_counts / total_count) * 100

    # Converte para HEX e calcula porcentagens
    color_data = [
        {"hex": "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2])),
         "percentage": round(percent, 2)}
        for color, percent in zip(colors, percentages)
    ]

    return color_data

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400

    n_colors = request.form.get("n_colors", default=10, type=int)

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        try:
            color_data = extract_colors(file_path, n_colors=n_colors)
            return jsonify({"colors": color_data}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Something went wrong."}), 500

@app.route("/", methods=["GET"])
def home():
    return "Color Extraction API is running!", 200

if __name__ == "__main__":
    app.run(debug=True)
