import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MAX_ITERATIONS = 100
PORIG_ZBIHU = 0.9   

stan = {
    'net_type': 'hamming', 
    'nazvy': ['Клас 1', 'Клас 2', 'Клас 3'],
    'etalony_abs': None,
    'etalony_norm': None,
    'vahy': None,
    'porih': None,
    'epsilon': None
}


def vytyagty_vektor_oznak(baitu_zobrazhennya, rozmir_sitky=(6, 6)):
    img = Image.open(io.BytesIO(baitu_zobrazhennya)).convert('RGBA')

    fon = Image.new('RGBA', img.size, (255, 255, 255, 255))
    fon.alpha_composite(img)
    arr = np.array(fon.convert('L'), dtype=float)

    krai = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    kolir_fonu = np.median(krai)
    maska = np.abs(arr - kolir_fonu) > 60

    abs_oznaky_nul = np.zeros(rozmir_sitky[0] * rozmir_sitky[1], dtype=float)
    ys, xs = np.where(maska)
    if len(ys) == 0:
        return abs_oznaky_nul

    obrizok = maska[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = obrizok.shape
    storona = max(h, w)
    kvadrat = np.zeros((storona, storona), dtype=np.uint8)
    dy, dx = (storona - h) // 2, (storona - w) // 2
    kvadrat[dy:dy + h, dx:dx + w] = obrizok.astype(np.uint8) * 255

    img60 = Image.fromarray(kvadrat).resize((60, 60), Image.BILINEAR)
    binary = (np.array(img60) >= 128).astype(float)

    h, w = binary.shape
    bh, bw = h // rozmir_sitky[0], w // rozmir_sitky[1]

    abs_oznaky = []
    for i in range(rozmir_sitky[0]):
        for j in range(rozmir_sitky[1]):
            blok = binary[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            abs_oznaky.append(np.sum(blok))

    return np.array(abs_oznaky, dtype=float)


def binarizatsiya(abs_vec):
    return np.where(abs_vec >= 50, 1, -1)


def zobrazhennya_v_url(baitu, mimetype):
    if not mimetype or not mimetype.startswith('image/'):
        mimetype = 'image/png'
    return f"data:{mimetype};base64," + base64.b64encode(baitu).decode('utf-8')


def create_hopfield_weights(patterns):
    patterns = np.asarray(patterns, dtype=float)
    n = patterns.shape[1]
    weights = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(len(patterns)):
                    weights[i][j] += patterns[k][i] * patterns[k][j]
                weights[i][j] /= n

    return weights


def hopfield_recognize(input_vector, weights):
    y = np.array(input_vector).astype(int).copy()
    historiya = [y.tolist()]

    for iteration in range(MAX_ITERATIONS):
        previous = y.copy()

        for j in range(len(y)):
            activation = np.dot(weights[j], y)
            if activation > 0:
                y[j] = 1
            elif activation < 0:
                y[j] = -1

        historiya.append(y.tolist())

        if np.array_equal(y, previous):
            return y, historiya, True

    return y, historiya, False


@app.route('/')
def holovna():
    return render_template('index.html')


@app.route('/vstanovyty_typ_merezhi', methods=['POST'])
def vstanovyty_typ_merezhi():
    data = request.get_json()
    typ = data.get('net_type')
    if typ in ['hamming', 'hopfield']:
        stan['net_type'] = typ
        stan['vahy'] = None
        stan['porih'] = None
        stan['epsilon'] = None
        return jsonify({'success': True, 'net_type': stan['net_type']})
    return jsonify({'error': 'Невірний тип мережі'}), 400


@app.route('/zavantazhyty_etalony', methods=['POST'])
def zavantazhyty_etalony():
    faily = [request.files.get(f'file{i}') for i in range(3)]

    for f in faily:
        if f is None or f.filename == '':
            return jsonify({'error': 'Оберіть зображення для всіх трьох класів'}), 400

    abs_spysok = []
    norm_spysok = []
    kartynky = []
    nazvy = []

    for i in range(3):
        fail = faily[i]
        baitu = fail.read()
        try:
            abs_vec = vytyagty_vektor_oznak(baitu)
        except Exception:
            return jsonify({'error': f'Не вдалося прочитати зображення для класу {i + 1}'}), 400

        abs_spysok.append(abs_vec)
        norm_spysok.append(binarizatsiya(abs_vec))
        kartynky.append(zobrazhennya_v_url(baitu, fail.mimetype))

        nazva = request.form.get(f'nazva{i}', '').strip()
        if nazva == '':
            nazvy.append(f'Клас {i + 1}')
        else:
            nazvy.append(nazva)

    stan['nazvy'] = nazvy
    stan['etalony_abs'] = abs_spysok
    stan['etalony_norm'] = norm_spysok
    stan['vahy'] = None
    stan['porih'] = None
    stan['epsilon'] = None

    return jsonify({
        'nazvy': nazvy,
        'kartynky': kartynky,
        'abs_vectors': [v.astype(int).tolist() for v in abs_spysok],
        'norm_vectors': [v.tolist() for v in norm_spysok]
    })


@app.route('/obchyslyty_vahy', methods=['POST'])
def obchyslyty_vahy():
    if stan['etalony_norm'] is None:
        return jsonify({'error': 'Спочатку завантажте еталони'}), 400

    X = np.array(stan['etalony_norm'])
    m, n = X.shape

    if stan['net_type'] == 'hamming':
        stan['vahy'] = X / 2
        stan['porih'] = np.full(m, n / 2)
        stan['epsilon'] = 1 / (m + 1)

        return jsonify({
            'net_type': 'hamming',
            'vahy': stan['vahy'].tolist(),
            'porih': stan['porih'].tolist(),
            'epsilon': round(stan['epsilon'], 4),
            'n': n,
            'm': m
        })

    elif stan['net_type'] == 'hopfield':
        W = create_hopfield_weights(X)
        stan['vahy'] = W
        stan['porih'] = np.zeros(n)

        return jsonify({
            'net_type': 'hopfield',
            'vahy': np.round(W, 3).tolist(),
            'n': n,
            'm': m
        })


@app.route('/klasyfikatsiya', methods=['POST'])
def klasyfikatsiya():
    if stan['vahy'] is None:
        return jsonify({'error': 'Спочатку обчисліть вагові коефіцієнти'}), 400

    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'Файл не обрано'}), 400

    fail = request.files['file']
    baitu = fail.read()

    try:
        abs_vec = vytyagty_vektor_oznak(baitu)
    except Exception:
        return jsonify({'error': 'Не вдалося прочитати зображення'}), 400

    x = binarizatsiya(abs_vec)
    n = len(x)

    if stan['net_type'] == 'hamming':
        # Логіка мережі Хемінга
        y = x @ stan['vahy'].T + stan['porih']
        pochatkovi_y = y.copy()

        eps = stan['epsilon']
        iteratsii = [y.tolist()]

        for k in range(MAX_ITERATIONS):
            s = y - eps * (np.sum(y) - y)
            y_novyi = np.maximum(s, 0)
            iteratsii.append(y_novyi.tolist())

            if np.allclose(y_novyi, y):
                y = y_novyi
                break
            y = y_novyi

        peremozhets = None
        max_znach = np.max(y)

        kilkist_maksimumiv = 0
        for val in y:
            if val == max_znach:
                kilkist_maksimumiv += 1

        if max_znach > 0 and kilkist_maksimumiv == 1:
            for i in range(len(y)):
                if y[i] == max_znach:
                    peremozhets = i
                    break

        klas_rezultat = stan['nazvy'][peremozhets] if peremozhets is not None else 'Образ не розпізнано'

        return jsonify({
            'net_type': 'hamming',
            'image_url': zobrazhennya_v_url(baitu, fail.mimetype),
            'abs_vector': abs_vec.astype(int).tolist(),
            'norm_vector': x.tolist(),
            'pershyi_shar': [round(v, 2) for v in pochatkovi_y.tolist()],
            'vidstani': [round(n - v, 2) for v in pochatkovi_y.tolist()],
            'iteratsii': [[round(v, 3) for v in ryadok] for ryadok in iteratsii],
            'kilkist_iteratsii': len(iteratsii) - 1,
            'nazvy': stan['nazvy'],
            'peremozhets': peremozhets,
            'klas_rezultat': klas_rezultat
        })

    elif stan['net_type'] == 'hopfield':
        W = stan['vahy']
        y, iteratsii, zbiglasya = hopfield_recognize(x, W)

        X_etalony = np.array(stan['etalony_norm'])
        zbigi_y = [int(np.sum(y == e)) for e in X_etalony]   
        zbigi_x = [int(np.sum(x == e)) for e in X_etalony]  
        max_zbigiv = max(zbigi_y)
        kandydaty = [i for i, z in enumerate(zbigi_y) if z == max_zbigiv]

        if len(kandydaty) > 1:
            najkrashe = max(zbigi_x[i] for i in kandydaty)
            kandydaty = [i for i in kandydaty if zbigi_x[i] == najkrashe]

        if zbiglasya and len(kandydaty) == 1 and max_zbigiv >= n * PORIG_ZBIHU:
            peremozhets = kandydaty[0]
            klas_rezultat = stan['nazvy'][peremozhets]
        else:
            peremozhets = None
            klas_rezultat = 'Образ не розпізнано'

        return jsonify({
            'net_type': 'hopfield',
            'image_url': zobrazhennya_v_url(baitu, fail.mimetype),
            'abs_vector': abs_vec.astype(int).tolist(),
            'norm_vector': x.tolist(),
            'stabilny_stan': y.astype(int).tolist(),
            'iteratsii': iteratsii,
            'kilkist_iteratsii': len(iteratsii) - 1,
            'zbiglasya': zbiglasya,
            'zbigi_vhid': zbigi_x,
            'zbigi_stan': zbigi_y,
            'n': n,
            'nazvy': stan['nazvy'],
            'peremozhets': peremozhets,
            'klas_rezultat': klas_rezultat
        })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
