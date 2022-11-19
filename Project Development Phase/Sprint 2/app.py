
from flask import Flask, request, render_template, jsonify

from PIL import Image
import numpy as np
import matplotlib
from flask import Flask, render_template, redirect, session, request, url_for
from flask_restful import Api


from flask_mysqldb import MySQL, MySQLdb
import bcrypt

app = Flask(__name__)
api = Api(app)


app = Flask(__name__)


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


@app.route("/recongize", methods=["GET", "POST"])
def recog():
    return render_template("default.html")


if __name__ == "__main__":
    
    
    app.debug = True
    app.run()
