from flask import Flask
from flask import session
import logging
from logging.handlers import RotatingFileHandler
import os
from flask import render_template
from flask import request
from scripts.StyleTransferRunner import StyleTransferRunner
from flask import send_from_directory
import uuid
from PIL import Image

app = Flask(__name__, static_folder='./templates/static')
model_runner = StyleTransferRunner()

log_folder = 'logs'
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

app.secret_key = open("./config/secret").read()
app.config['UPLOAD_FOLDER'] = './temp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB

# Create a file handler for the log file
file_handler = RotatingFileHandler(os.path.join(log_folder, 'app.log'), maxBytes=1024 * 1024 * 10, backupCount=10)
file_handler.setLevel(logging.INFO)

# Create a formatter for the log messages
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Get the app's logger and add the file handler
app.logger.addHandler(file_handler)

@app.route('/uploads/<dir>/<filename>')
def uploaded_file(dir, filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], f'{dir}/{filename}')

@app.route("/", methods = ["GET", "POST"])
def hello_world():
    app.logger.info('Home page accessed')
    if request.method == "GET":
        return render_template('home.html')
    else:
        file1 = request.files['content']
        file2 = request.files['style']
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        
        user_id = session['user_id']
        app.logger.info(f'New session started {user_id}')
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], user_id)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        if file1:
            file1.seek(0)
            file1.save(os.path.join(app.config['UPLOAD_FOLDER'], f'{user_id}/content.png'))
        if file2:
            file2.seek(0)
            file2.save(os.path.join(app.config['UPLOAD_FOLDER'], f'{user_id}/style.png'))
        
        query = {
            'content' : f'/uploads/{user_id}/content.png',
            'style' : f'/uploads/{user_id}/style.png',
            'content_weight' : int(request.form['contentSlider']),
            'style_weight' : int(request.form['styleSlider']),
            'learning_rate' : int(request.form['learningRateSlider']),
            'samples' : int(request.form['examplesSlider'])
        }

        content = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], f'{user_id}/content.png')).resize((224, 224))
        style = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], f'{user_id}/style.png')).resize((224, 224))
        app.logger.info(f'Running Model {user_id}')
        images = model_runner.transferStyle(
            content,
            style,
            (query['content_weight'], query['style_weight']),
            (query['learning_rate'] / 1000),
            query['samples'],
            os.path.join(app.config['UPLOAD_FOLDER'], f'{user_id}')
        )
        images = [dirname.replace("temp", "uploads") for dirname in images]
        app.logger.info(f'Finised Model Run {user_id}')

        query['images'] = images

        return render_template('home.html', **query)

