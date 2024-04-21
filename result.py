from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import numpy as np
from flask_cors import CORS
from PIL import Image  # Import Image from Pillow

app = Flask(__name__)
app.debug = True
CORS(app)

model = load_model('model1.h5')

def preprocess_image(image_file):
    # Open the image using PIL (Pillow)
    img = Image.open(image_file)
    # Resize the image to the desired target size
    img = img.resize((224, 224))
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/')
def my_index():
    return render_template("index.html", flask_token="Hello world")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        categories = ['Adenocarcinoma', 'Neuroendocrine Tumors', 'Squamous cell cancer']
        definitions = ['Adenocarcinoma is a type of cancer. It develops in the glands that line your organs. Common forms of adenocarcinoma include breast, stomach, prostate, lung, pancreatic and colorectal cancers.',
                       'Neuroendocrine tumors are cancers that begin in specialized cells called neuroendocrine cells.',
                       'Squamous cell carcinoma of the lungs is a type of non-small-cell cancer that originates in the thin, flat cells lining the airways.']
        conditions = ['Curable (depending on various factors such as the stage at diagnosis, tumor size, spread to lymph nodes or other organs, and the presence of specific genetic mutations.)',
                      'Not Critical (but also depending on type)',
                      'Not Critical (the squamous cell carcinoma survival rate is very high(around 99%))']
        medicines = ['1. Chemotherapy that include drugs such as cisplatin, carboplatin, pemetrexed, and gemcitabine  2. Targeted Therapy like erlotinib, gefitinib, afatinib, and osimertinib  3. Immunotherapy drugs such as pembrolizumab and atezolizumab',
                     '1. Chemotherapy for neuroendocrine tumors may include drugs such as cisplatin, etoposide, and temozolomide  2. Somatostatin Analogs such drugs as octreotide and lanreotide  3. Targeted Therapy like Everolimus',
                     '1. Chemotherapy for squamous cell carcinoma may include drugs such as cisplatin, carboplatin, paclitaxel, docetaxel, and gemcitabine  2. Immunotherapy drugs like pembrolizumab and nivolumab  3. Targeted Therapy (less common for squamous cell cancer, drugs like cetuximab may be used in certain cases)']

        image_file = request.files['image']
        processed_image = preprocess_image(image_file)
        predictions = model.predict(processed_image)
        final_prediction = int(np.argmax(predictions))
        confidence = np.max(predictions) * 100

        final_output = {
            'category': categories[final_prediction],
            'confidence': confidence,
            'definition': definitions[final_prediction],
            'condition': conditions[final_prediction],
            'medicine': medicines[final_prediction],
        }

        return jsonify(final_output)
    except KeyError as e:
        return jsonify({'error': 'Invalid category'}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred during prediction'}), 500

app.run(debug=True)
