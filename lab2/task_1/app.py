import io
import base64
import hashlib

import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

stan = {
    'vahy': None,
    'porih': 0.0,
    'krok_navchannya': 0.1,
    'istoriya': [],                
    'vybirka': {},                 
    'potochnyi_id': None,
    'potochnyi_norm_vektor': None,
    'potochnyi_abs_vektor': None,
    'ostannye_pribadannya': None
}

def vytyagty_vektor_oznak(baitu_zobrazhennya, rozmir_sitky=(6, 6)):
    img = Image.open(io.BytesIO(baitu_zobrazhennya)).convert('RGBA')

    fon = Image.new('RGBA', img.size, (255, 255, 255, 255))
    fon.alpha_composite(img)
    img = fon.convert('L')
    img = img.resize((60, 60))
    img_arr = np.array(img)

    binary = (img_arr < 128).astype(float)

    h, w = binary.shape
    bh, bw = h // rozmir_sitky[0], w // rozmir_sitky[1]

    abs_oznaky = []
    for i in range(rozmir_sitky[0]):
        for j in range(rozmir_sitky[1]):
            blok = binary[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            abs_oznaky.append(np.sum(blok))

    abs_oznaky = np.array(abs_oznaky, dtype=float)

    norm_oznaky = abs_oznaky / (bh * bw)

    return abs_oznaky, norm_oznaky


def klas_za_vektorom(x):
    s = np.dot(np.array(stan['vahy']), x) - stan['porih']
    y = 1 if s > 0 else -1
    return s, y


def pomylky_na_vybirtsi():
    kilkist = 0
    for obraz in stan['vybirka'].values():
        s, y = klas_za_vektorom(np.array(obraz['x']))
        if y != obraz['klas']:
            kilkist += 1
    return kilkist


@app.route('/')
def holovna():
    return render_template('index.html')


@app.route('/zavantazhyty', methods=['POST'])
def zavantazhyty_kartynku():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не обрано'}), 400

    fail = request.files['file']
    if fail.filename == '':
        return jsonify({'error': 'Порожнє ім\'я файлу'}), 400

    baitu = fail.read()
    try:
        abs_vec, norm_vec = vytyagty_vektor_oznak(baitu)
    except Exception:
        return jsonify({'error': 'Не вдалося прочитати зображення'}), 400

    stan['potochnyi_abs_vektor'] = abs_vec.tolist()
    stan['potochnyi_norm_vektor'] = norm_vec.tolist()
    stan['potochnyi_id'] = hashlib.md5(baitu).hexdigest()
    stan['ostannye_pribadannya'] = None

    if stan['vahy'] is None or len(stan['vahy']) != len(norm_vec):
        stan['vahy'] = np.random.uniform(0, 1, size=len(norm_vec)).tolist()
        stan['porih'] = float(np.random.uniform(0, 1))

    base64_img = base64.b64encode(baitu).decode('utf-8')
    img_url = f"data:{fail.mimetype};base64,{base64_img}"

    return jsonify({
        'image_url': img_url,
        'abs_vector': [round(x, 2) for x in abs_vec.tolist()],
        'norm_vector': [round(x, 4) for x in norm_vec.tolist()],
        'vahy': [round(w, 4) for w in stan['vahy']],
        'porih': round(stan['porih'], 4),
        'krok': stan['krok_navchannya']
    })


@app.route('/iniciliazatsiya', methods=['POST'])
def initsializatsiya_perseptrona():
    data = request.get_json() or {}
    try:
        krok = float(data.get('krok'))
        porih = data.get('porih')
        porih = float(porih) if str(porih).strip() != '' else float(np.random.uniform(0, 1))
    except (TypeError, ValueError):
        return jsonify({'error': 'Крок і поріг мають бути числами'}), 400

    if not 0 < krok < 1:
        return jsonify({'error': 'Крок навчання має бути меншим за 1 (0 < r < 1)'}), 400

    stan['krok_navchannya'] = krok
    stan['porih'] = porih

    dovzhyna_vektora = len(stan['potochnyi_norm_vektor']) if stan['potochnyi_norm_vektor'] else 36
    vlasni_vahy = data.get('vlasni_vahy')

    if vlasni_vahy is not None and str(vlasni_vahy).strip() != '':
        try:
            znachennya = float(vlasni_vahy)
            stan['vahy'] = [znachennya] * dovzhyna_vektora
        except ValueError:
            try:
                spysok_vah = [float(x.strip()) for x in str(vlasni_vahy).split(',') if x.strip()]
            except ValueError:
                return jsonify({'error': 'Ваги мають бути числами'}), 400
            if len(spysok_vah) == dovzhyna_vektora:
                stan['vahy'] = spysok_vah
            else:
                return jsonify({'error': f'Потрібно вказати 1 число або {dovzhyna_vektora} чисел через кому'}), 400
    else:
        stan['vahy'] = np.random.uniform(0, 1, size=dovzhyna_vektora).tolist()

    stan['istoriya'] = []
    stan['vybirka'] = {}

    return jsonify({
        'status': 'success',
        'vahy': [round(w, 4) for w in stan['vahy']],
        'porih': round(stan['porih'], 4),
        'krok': stan['krok_navchannya']
    })


@app.route('/klasyfikatsiya', methods=['POST'])
def klasyfikatsiya():
    if stan['potochnyi_norm_vektor'] is None:
        return jsonify({'error': 'Спочатку завантажте зображення'}), 400

    x = np.array(stan['potochnyi_norm_vektor'])
    s_suma, y_vyhid = klas_za_vektorom(x)
    stan['ostannye_pribadannya'] = y_vyhid

    return jsonify({
        's_suma': round(float(s_suma), 4),
        'vyhid': y_vyhid,
        'klas_rezultat': 'Клас A (+1)' if y_vyhid == 1 else 'Клас B (-1)'
    })


@app.route('/vchytsel_korrektsiya', methods=['POST'])
def vchytsel_korrektsiya():
    if stan['potochnyi_norm_vektor'] is None:
        return jsonify({'error': 'Спочатку завантажте зображення'}), 400

    data = request.get_json() or {}
    tsilovyi_klas = int(data.get('tsilovyi_klas', 1))

    x = np.array(stan['potochnyi_norm_vektor'])
    w = np.array(stan['vahy'])
    b = stan['porih']
    r = stan['krok_navchannya']

    s_suma, y_vyhid = klas_za_vektorom(x)

    pomylka = tsilovyi_klas - y_vyhid

    if pomylka != 0:
        w = w + r * pomylka * x
        b = b - r * pomylka
        stan['vahy'] = w.tolist()
        stan['porih'] = float(b)
        stan['istoriya'].append(1)
        povidomlennya = f"Результат спростовано. Ваги та поріг скориговано (помилка e = {pomylka})."
    else:
        stan['istoriya'].append(0)
        povidomlennya = "Результат підтверджено. Корекція не потрібна."

    stan['vybirka'][stan['potochnyi_id']] = {'x': x.tolist(), 'klas': tsilovyi_klas}

    nova_s_suma, novyi_y = klas_za_vektorom(x)

    klasy = set(o['klas'] for o in stan['vybirka'].values())
    pomylky = pomylky_na_vybirtsi()
    zbizhnist = (len(klasy) == 2) and (pomylky == 0)

    return jsonify({
        'vahy': [round(val, 4) for val in stan['vahy']],
        'porih': round(stan['porih'], 4),
        'pomylka': pomylka,
        'nova_s_suma': round(float(nova_s_suma), 4),
        'novyi_klas': 'Клас A (+1)' if novyi_y == 1 else 'Клас B (-1)',
        'vsego_krokiv': len(stan['istoriya']),
        'kilkist_korektsii': sum(stan['istoriya']),
        'rozmir_vybirky': len(stan['vybirka']),
        'pomylky_vybirky': pomylky,
        'zbizhnist': zbizhnist,
        'povidomlennya': povidomlennya
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
