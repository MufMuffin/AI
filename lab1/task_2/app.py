import base64
import io
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CLASSES = ["Клас A", "Клас B", "Клас C", "Клас D"]


def process_image(image, grid_rows=5, grid_cols=5):
    img_gray = image.convert("L")
    img_bw_single = img_gray.point(lambda p: 255 if p > 200 else 0, mode="1")

    width, height = img_bw_single.size
    cell_w = width / grid_cols
    cell_h = height / grid_rows

    abs_vector = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            box = (
                int(c * cell_w),
                int(r * cell_h),
                int((c + 1) * cell_w),
                int((r + 1) * cell_h),
            )
            cell_crop = img_bw_single.crop(box)
            black_pixels = sum(1 for p in cell_crop.getdata() if p == 0)
            abs_vector.append(black_pixels)

    total_black = sum(abs_vector)
    if total_black > 0:
        norm_vector = [x / total_black for x in abs_vector]
    else:
        norm_vector = [0.0] * len(abs_vector)

    img_bw_rgb = img_bw_single.convert("RGB")
    draw = ImageDraw.Draw(img_bw_rgb)

    for c in range(1, grid_cols):
        x = int(c * cell_w)
        draw.line([(x, 0), (x, height)], fill="red", width=1)

    for r in range(1, grid_rows):
        y = int(r * cell_h)
        draw.line([(0, y), (width, y)], fill="red", width=1)

    buf = io.BytesIO()
    img_bw_rgb.save(buf, format="PNG")
    bw_grid_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    norm_vector_rounded = [round(x, 4) for x in norm_vector]

    return {
        "b64": bw_grid_b64,
        "abs_vector": np.array(abs_vector, dtype=np.float64),
        "norm_vector": np.array(norm_vector, dtype=np.float64),
        "abs_full": abs_vector,                 
        "norm_full": norm_vector_rounded        
    }


def chebyshev_distance(vec1, vec2):
    return float(np.max(np.abs(vec1 - vec2)))


@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        processed_templates = {}

        for cls in CLASSES:
            file_key = f"template_{cls}"
            if file_key in request.files:
                file = request.files[file_key]
                if file and file.filename != '':
                    img = Image.open(file.stream)
                    processed_templates[cls] = process_image(img, grid_rows=5, grid_cols=5)

        unknown_data = None
        if "unknown" in request.files:
            file = request.files["unknown"]
            if file and file.filename != '':
                img = Image.open(file.stream)
                unknown_data = process_image(img, grid_rows=5, grid_cols=5)

        if len(processed_templates) == len(CLASSES) and unknown_data:
            distances = {}

            for cls, t_data in processed_templates.items():
                dist = chebyshev_distance(unknown_data["norm_vector"], t_data["norm_vector"])
                distances[cls] = round(dist, 6)

            predicted_class = min(distances, key=distances.get)
            min_dist = distances[predicted_class]

            results = {
                "templates": processed_templates,
                "unknown": unknown_data,
                "distances": distances,
                "predicted_class": predicted_class,
                "min_dist": min_dist
            }

    return render_template("index.html", classes=CLASSES, results=results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)