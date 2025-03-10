import io

from PIL import Image
from flask import Flask, render_template, request

from network import image_classify

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])

def upload():
    if 'image' not in request.file:
        return 'No file passed'
    
    file = request.file['image']

    if file:
        img = Image.open(io.BytesIO(file.read()))

        return f'Classification: {image_classify(img)}'
    

if __name__ == '__main__':
    app.run(debug=True)