import os
from flask import Flask, request, render_template
from pdfminer.high_level import extract_text
import pickle

app = Flask(__name__)
UPLOAD_FOLDER = 'resumes'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def extract_resume_text(path):
    return extract_text(path)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        resumes = request.files.getlist('resumes')
        for resume in resumes:
            path = os.path.join(UPLOAD_FOLDER, resume.filename)
            resume.save(path)
            text = extract_resume_text(path)
            vec = vectorizer.transform([text])
            prediction = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0][1] * 100
            results.append((resume.filename, prediction, f"{prob:.2f}%"))
    return render_template('index.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
