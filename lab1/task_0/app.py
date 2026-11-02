import io
import base64
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    img_data = None

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                try:
                    from PIL import Image
                    image_bytes = file.read()
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    buffered = io.BytesIO()
                    image.save(buffered, format=image.format or "JPEG")
                    img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    
                except Exception as e:
                    error = f"Could not open image: {e}"

    return render_template('index.html', img_data=img_data, error=error)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
