import base64
import io
import numpy as np
import os
from PIL import Image, ImageDraw
from flask import Flask, render_template, request

import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
app.config['max'] = 16 * 1024 * 1024

CLASSES = ["Клас 1", "Клас 2", "Клас 3"]


def process_image(image, grid_rows=5, grid_cols=5):

    img_gray = image.convert("L")
    img_bw = img_gray.point(lambda p: 255 if p > 200 else 0, mode="1")

    width, height = img_bw.size
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
            cell_crop = img_bw.crop(box)
            
            pixels = np.array(cell_crop)
            black_pixels = int(np.sum(pixels == 0))
            
            abs_vector.append(black_pixels)

    total_black = sum(abs_vector)
    if total_black > 0:
        norm_vector = [x / total_black for x in abs_vector]
    else:
        norm_vector = [0.0] * len(abs_vector)

    img_with_grid = img_bw.convert("RGB")
    draw = ImageDraw.Draw(img_with_grid)

    for c in range(1, grid_cols):
        x = int(c * cell_w)
        draw.line([(x, 0), (x, height)], fill="red", width=2)

    for r in range(1, grid_rows):
        y = int(r * cell_h)
        draw.line([(0, y), (width, y)], fill="red", width=2)

    buf_grid = io.BytesIO()
    img_with_grid.save(buf_grid, format="PNG")
    grid_b64 = "data:image/png;base64," + base64.b64encode(buf_grid.getvalue()).decode("utf-8")

    norm_vector_rounded = [round(x, 4) for x in norm_vector]

    return {
        "grid_b64": grid_b64,
        "abs_vector": np.array(abs_vector, dtype=np.float64),
        "norm_vector": np.array(norm_vector, dtype=np.float64),
        "abs_full": abs_vector,
        "norm_full": norm_vector_rounded
    }


def euclidean_distance(vec1, vec2):
    return float(np.sqrt(np.sum((vec1 - vec2) ** 2)))


@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        training_data = {cls: [] for cls in CLASSES}

        for clas in CLASSES:
            files = request.files.getlist(f"train_{clas}")
            for file in files:
                if file and file.filename != '':
                    img = Image.open(file.stream)
                    data = process_image(img)
                    training_data[clas].append(data)

        geometric_centers = {}
        cluster_stats = {}

        for clas in CLASSES:
            if training_data[clas]:
                vectors = [item["norm_vector"] for item in training_data[clas]]
                mean_center = np.mean(np.array(vectors), axis=0)
                geometric_centers[clas] = mean_center

                cluster_stats[clas] = {
                    "count": len(training_data[clas]),
                    "center_full": [round(float(x), 4) for x in mean_center],
                    "items": training_data[clas]
                }

        unknown_data = None
        if "unknown" in request.files:
            file = request.files["unknown"]
            if file and file.filename != '':
                img = Image.open(file.stream)
                unknown_data = process_image(img)

        if unknown_data and all(len(training_data[clas]) > 0 for clas in CLASSES):
            u_vec = unknown_data["norm_vector"]
            distances = {}

            for clas in CLASSES:
                dist = euclidean_distance(u_vec, geometric_centers[clas])
                distances[clas] = round(dist, 6)

            predicted_class = min(distances, key=distances.get)
            min_dist = distances[predicted_class]

            results = {
                "unknown": unknown_data,
                "cluster_stats": cluster_stats,
                "distances": distances,
                "predicted_class": predicted_class,
                "min_dist": min_dist
            }

    return render_template("index3.html", classes=CLASSES, results=results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)