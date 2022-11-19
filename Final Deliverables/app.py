import bcrypt
from flask_mysqldb import MySQL, MySQLdb
import numpy as np
import os
from PIL import Image
from flask import Flask, request, render_template, jsonify
import matplotlib.pyplot as plt
from train import MnistModel
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib
import config
import base64
import torch
from flask import Flask, render_template, redirect, session, request, url_for
from flask_restful import Api
from werkzeug.utils import secure_filename, redirect
from gevent.pywsgi import WSGIServer
from keras.models import load_model
from keras.preprocessing import image
from flask import send_from_directory


app = Flask(__name__)
api = Api(app)

UPLOAD_FOLDER = 'uploads'


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("testmodel.h5")

matplotlib.use('Agg')


MODEL = None
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


app.secret_key = "secret key"

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'flaskdb'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == 'GET':
        return render_template("register.html")
    else:
        name = request.form['name']
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
        hash_password = bcrypt.hashpw(password, bcrypt.gensalt())
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (name, email, password) VALUES (%s,%s,%s)",
                    (name, email, hash_password,))
        mysql.connection.commit()
        session['name'] = request.form['name']
        session['email'] = request.form['email']
        session['loggedin'] = True
        return redirect(url_for('home'))


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password'].encode('utf-8')

        curl = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        curl.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = curl.fetchone()
        curl.close()

        if len(user) > 0:
            if bcrypt.hashpw(password, user["password"].encode('utf-8')) == user["password"].encode('utf-8'):
                session['name'] = user['name']
                session['email'] = user['email']
                session['loggedin'] = True
                return render_template("index.html")
            else:
                return "Error password and email not match"
        else:
            return "Error user not found"
    else:
        return render_template("login.html")


@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.clear()
    return render_template("index.html")


@app.route("/recognize_page", methods=["GET", "POST"])
def recognize_page():
    return render_template("Recognize.html")


@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    return render_template("upload.html")


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        f = request.files["image"]
        filepath = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filepath))

        upload_img = os.path.join(UPLOAD_FOLDER, filepath)
        img = Image.open(upload_img).convert("L")
        img = img.resize((28, 28))

        im2arr = np.array(img)

        im2arr = im2arr.reshape(1, 28, 28, 1)

        pred = model.predict(im2arr)

        num = np.argmax(pred, axis=1)

        return render_template('upload.html', num=str(num[0]))


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def register_hook():
    save_output = SaveOutput()
    hook_handles = []

    for layer in MODEL.modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)
    return save_output


def module_output_to_numpy(tensor):
    return tensor.detach().to('cpu').numpy()


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{0:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')


def prob_img(probs):
    fig, ax = plt.subplots()
    rects = ax.bar(range(len(probs)), probs)
    ax.set_xticks(range(len(probs)), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    ax.set_ylim(0, 110)
    ax.set_title('Probability % of Digit by Model')
    autolabel(rects, ax)
    probimg = BytesIO()
    fig.savefig(probimg, format='png')
    probencoded = base64.b64encode(probimg.getvalue()).decode('utf-8')
    return probencoded


def interpretability_img(save_output):
    images = module_output_to_numpy(save_output.outputs[0])
    with plt.style.context("seaborn-white"):
        fig, _ = plt.subplots(figsize=(20, 20))
        plt.suptitle("Interpretability by Model", fontsize=50)
        for idx in range(16):
            plt.subplot(4, 4, idx+1)
            plt.imshow(images[0, idx])
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    interpretimg = BytesIO()
    fig.savefig(interpretimg, format='png')
    interpretencoded = base64.b64encode(
        interpretimg.getvalue()).decode('utf-8')
    return interpretencoded


def mnist_prediction(img):
    save_output = register_hook()
    img = img.to(DEVICE, dtype=torch.float)
    outputs = MODEL(x=img)

    probs = torch.exp(outputs.data)[0] * 100
    probencoded = prob_img(probs)
    interpretencoded = interpretability_img(save_output)

    _, output = torch.max(outputs.data, 1)
    pred = module_output_to_numpy(output)
    return pred[0], probencoded, interpretencoded


@app.route("/process", methods=["GET", "POST"])
def process():
    data_url = str(request.get_data())
    offset = data_url.index(',')+1
    img_bytes = base64.b64decode(data_url[offset:])
    img = Image.open(BytesIO(img_bytes))
    img = img.convert('L')
    img = img.resize((28, 28))

    img = np.array(img)
    img = img.reshape((1, 28, 28))
    img = torch.tensor(img, dtype=torch.float).unsqueeze(0)

    data, probencoded, interpretencoded = mnist_prediction(img)

    response = {
        'data': str(data),
        'probencoded': str(probencoded),
        'interpretencoded': str(interpretencoded),
    }
    return jsonify(response)


if __name__ == "__main__":
    app.debug = True
    app.run()
