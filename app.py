from flask import Flask, render_template, request, jsonify
import json
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
from keras.models import load_model
import base64
import webbrowser

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = load_model("mobilenet_model.h5")

# Initialize face detector using Haar Cascade
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

emotion_counts = {'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 'surprise': 0, 'neutral': 0}

# Default YouTube links
youtube_links = {
    'angry': "https://www.youtube.com/watch?v=i90EvEZ3axw&list=PL97kH0xIAu5n8JTmEVhxCSp8EhIrAXO8R&pp=gAQB",
    'fear' : "https://www.youtube.com/watch?v=d5gf9dXbPi0&list=PL97kH0xIAu5k2e4Uynb1mS6E0gdSyy8Gx&pp=gAQB",
    'happy': "https://www.youtube.com/watch?v=PEM0Vs8jf1w&list=PL97kH0xIAu5lRrHRIr-OlINoAGC3yNjs6&pp=gAQB",
    'sad'  : "https://www.youtube.com/watch?v=_m6l5nKEGIA&list=PL97kH0xIAu5kcX185p6-u8nr6boWXLZNH&pp=gAQB",
    'neutral': "https://www.youtube.com/watch?v=dBFp0Ext0y8&list=PL97kH0xIAu5llG7M9KG6AANgjHE5vpHv2&pp=gAQB"
}

# ----------------------------

# Load links from JSON file
def load_links():
    global youtube_links
    try:
        with open('links.json', 'r') as file:
            youtube_links = json.load(file)
    except FileNotFoundError:
        pass  # File belum ada, gunakan default

@app.before_request
def reload_links():
    load_links()  # Pastikan setiap request menggunakan data terbaru

@app.route('/')
def index():
    return render_template('check_page.html')

@app.route('/coba_kamera')
def coba_kamera():
    return render_template('coba_kamera.html')

@app.route('/offline_camera')
def offline_camera():
    return render_template('offline_camera.html')

@app.route('/get_link/<emotion>', methods=['GET'])
def get_link(emotion):
    try:
        # Dapatkan link untuk emosi yang diminta
        if emotion in youtube_links:
            return jsonify({'success': True, 'link': youtube_links[emotion]})
        else:
            return jsonify({'success': False, 'message': 'Emotion not found'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/setting')
def setting():
    return render_template('setting.html')

@app.route('/update_links', methods=['POST'])
def update_links():
    try:
        data = request.get_json()

        # Perbarui link berdasarkan data input
        for emotion, default_link in youtube_links.items():
            youtube_links[emotion] = data.get(emotion, default_link)

        # Simpan ke file JSON
        with open('links.json', 'w') as file:
            json.dump(youtube_links, file)

        return jsonify({"success": True, "message": "Links updated successfully!"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


# ----------------------------

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = base64.b64decode(data['image'].split(',')[1])
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces_detected) == 0:
        return jsonify({'emotion': 'No face detected'})
    
    x, y, w, h = faces_detected[0]
    roi_gray = gray_img[y:y + h, x:x + w]  # Ensure correct ROI dimensions
    roi_gray = cv2.resize(roi_gray, (224, 224))
    img_pixels = image.img_to_array(roi_gray)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    
    predictions = model.predict(img_pixels)
    max_index = np.argmax(predictions[0])
    predicted_emotion = emotions[max_index]
    
    # Update detected emotion counts
    emotion_counts[predicted_emotion] += 1

    # Check if 20 facial expressions have been detected
    if sum(emotion_counts.values()) == 15:
        highest_emotion = max(emotion_counts, key=emotion_counts.get)
        if highest_emotion in youtube_links:
            webbrowser.open(youtube_links[highest_emotion])
        return jsonify({'emotion': predicted_emotion, 'done': True})
    
    return jsonify({'emotion': predicted_emotion, 'done': False})

@app.route('/reset', methods=['POST'])
def reset():
    global emotion_counts
    emotion_counts = {emotion: 0 for emotion in emotions}
    return jsonify({'reset': True})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

