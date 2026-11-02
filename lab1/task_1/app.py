import base64
import io
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, render_template, request

app = Flask(__name__)

app.config['max'] = 16 * 1024 * 1024

def process_image(image_bytes, grid_rows=5, grid_cols=5, threshold=128):
    orig_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = orig_img.size

    gray_img = orig_img.convert("L")
    img_np = np.array(gray_img)

    bw_np = np.where(img_np < threshold, 0, 255).astype(np.uint8)
    
    binary_mask = (bw_np == 0)

    row_edges = np.linspace(0, height, grid_rows + 1, dtype=int)
    col_edges = np.linspace(0, width, grid_cols + 1, dtype=int)

    abs_vector = []

    for i in range(grid_rows):
        for j in range(grid_cols):
            r_start, r_end = row_edges[i], row_edges[i + 1]
            c_start, c_end = col_edges[j], col_edges[j + 1]
            
            block = binary_mask[r_start:r_end, c_start:c_end]
            pixel_count = int(np.sum(block))
            abs_vector.append(pixel_count)

    total_pixels = sum(abs_vector)
    if total_pixels > 0:
        norm_vector = [round(count / total_pixels, 4) for count in abs_vector]
    else:
        norm_vector = [0.0] * len(abs_vector)

    bw_image = Image.fromarray(bw_np).convert("RGB")

    draw = ImageDraw.Draw(bw_image)

    for col_x in col_edges:
        draw.line([(col_x, 0), (col_x, height)], fill="red", width=2)

    for row_y in row_edges:
        draw.line([(0, row_y), (width, row_y)], fill="red", width=2)

    orig_buffer = io.BytesIO()
    orig_img.save(orig_buffer, format="PNG")
    original_base64 = base64.b64encode(orig_buffer.getvalue()).decode('utf-8')

    grid_buffer = io.BytesIO()
    bw_image.save(grid_buffer, format="PNG")
    grid_base64 = base64.b64encode(grid_buffer.getvalue()).decode('utf-8')

    return original_base64, grid_base64, abs_vector, norm_vector


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="Файл не знайдено у запиті.")
            
        file = request.files['file']
        
        if file.filename == '':
            return render_template('index.html', error="Файл не обрано.")

        try:
            image_bytes = file.read()
            original_data, grid_data, abs_vector, norm_vector = process_image(image_bytes)

            return render_template(
                'index.html',
                original_data=original_data,
                grid_data=grid_data,
                abs_vector=abs_vector,
                norm_vector=norm_vector
            )
        except Exception as e:
            return render_template('index.html', error=f"Помилка при обробці зображення: {str(e)}")

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)