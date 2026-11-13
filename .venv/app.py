from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, get_flashed_messages

from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import base64
import json
import os
from datetime import datetime
from pathlib import Path

app = Flask(__name__)

allowed_origins = [
    origin.strip()
    for origin in os.getenv('CORS_ORIGINS', '').split(',')
    if origin.strip()
]
CORS(app, origins=allowed_origins or '*', supports_credentials=bool(allowed_origins))

secret_key = os.getenv('SECRET_KEY')
if not secret_key:
    if os.getenv('APP_ENV') == 'production':
        raise RuntimeError('SECRET_KEY must be configured in production')
    secret_key = 'local-development-only-secret'

database_url = os.getenv('DATABASE_URL')
if not database_url:
    database_url = 'sqlite:///' + str(Path(__file__).resolve().parent / 'dr_analyzer.db')
if database_url.startswith('mysql://'):
    database_url = 'mysql+pymysql://' + database_url[len('mysql://'):]

app.config['SECRET_KEY'] = secret_key
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SESSION_COOKIE_SECURE'] = os.getenv('APP_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


# --- Database Models (Tables) ---

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    # Store hashed passwords, not plain text!
    password = db.Column(db.String(60), nullable=False)
    # Link to the predictions
    predictions = db.relationship('Prediction', backref='author', lazy=True)

    def __repr__(self):
        return f"User('{self.username}')"

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    result_text = db.Column(db.String(100), nullable=False)
    result_level = db.Column(db.Integer, nullable=False)
    is_detected = db.Column(db.Boolean, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.now)
    # Foreign key to link to the User table
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Prediction('{self.result_text}' at {self.timestamp})"


# Map for severity classes
SEVERITY_CLASSES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

def severity_from_scores(scores):
    if isinstance(scores, dict):
        scores = [{'label': label, 'score': score} for label, score in scores.items()]
    if not isinstance(scores, list) or not scores:
        raise RuntimeError('Inference API returned no classifications')

    best = max(scores, key=lambda item: float(item.get('score', 0)))
    label = str(best.get('label', '')).strip().lower()
    for level, severity in SEVERITY_CLASSES.items():
        if (
            label in {str(level), severity.lower()}
            or severity.lower() in label
            or label.endswith(f'_{level}')
            or label.endswith(f' {level}')
        ):
            return level, severity
    raise RuntimeError(f'Inference API returned unknown class label: {label}')

def predict_from_gradio(image_bytes, content_type, space_url):
    data_url = (
        f'data:{content_type or "application/octet-stream"};base64,'
        f'{base64.b64encode(image_bytes).decode("ascii")}'
    )
    submit_request = Request(
        f'{space_url.rstrip("/")}/call/predict',
        data=json.dumps({'data': [data_url]}).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urlopen(submit_request, timeout=30) as response:
            event_id = json.loads(response.read().decode('utf-8')).get('event_id')
        if not event_id:
            raise RuntimeError('Gradio API did not return an event ID')

        with urlopen(
            Request(f'{space_url.rstrip("/")}/call/predict/{event_id}'),
            timeout=60
        ) as response:
            for line in response.read().decode('utf-8').splitlines():
                if line.startswith('data: '):
                    result = json.loads(line[6:])
                    output = result[0] if isinstance(result, list) else result
                    return severity_from_scores(output)
    except HTTPError as error:
        details = error.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'Gradio API returned HTTP {error.code}: {details}') from error
    except URLError as error:
        raise RuntimeError(f'Gradio API request failed: {error.reason}') from error
    raise RuntimeError('Gradio API returned no prediction')

def predict_severity(image_bytes, content_type):
    gradio_space_url = os.getenv('GRADIO_SPACE_URL')
    if gradio_space_url:
        return predict_from_gradio(image_bytes, content_type, gradio_space_url)

    inference_url = os.getenv('INFERENCE_API_URL')
    if not inference_url:
        raise RuntimeError('GRADIO_SPACE_URL is not configured')

    headers = {'Content-Type': content_type or 'application/octet-stream'}
    inference_token = os.getenv('INFERENCE_API_TOKEN') or os.getenv('HF_TOKEN')
    if inference_token:
        headers['Authorization'] = f'Bearer {inference_token}'

    request = Request(inference_url, data=image_bytes, headers=headers, method='POST')
    try:
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode('utf-8'))
    except HTTPError as error:
        details = error.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'Inference API returned HTTP {error.code}: {details}') from error
    except URLError as error:
        raise RuntimeError(f'Inference API request failed: {error.reason}') from error

    if isinstance(result, dict) and result.get('error'):
        raise RuntimeError(f'Inference API error: {result["error"]}')
    scores = result[0] if isinstance(result, list) and result and isinstance(result[0], list) else result
    return severity_from_scores(scores)

def initialize_database():
    with app.app_context():
        db.create_all()

# --- Page Routes (URLs) ---

@app.route("/")
def home():
    session.clear()
    return render_template(
        'login.html',
        forgot_password_url=os.getenv('FORGOT_PASSWORD_URL', '#')
    )

@app.route('/healthz')
def healthz():
    return jsonify({'status': 'ok'})

@app.route("/register", methods=['GET','POST'])
def register():
    if request.method == 'POST':
        initialize_database()
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please log in.', 'danger')
            return redirect(url_for('register'))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Account created! You can now log in.', 'success')
        return redirect(url_for('home'))
    
    return render_template('register.html')


@app.route("/login", methods=['POST'])
def login():
    initialize_database()
    username = request.form['username']
    password = request.form['password']
    
    user = User.query.filter_by(username=username).first()
    
    # Check if user exists and password is correct
    if user and bcrypt.check_password_hash(user.password, password):
        # Log the user in by saving their ID in the session
        session['user_id'] = user.id
        session['username'] = user.username
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash('Login unsuccessful. Check username and password.', 'danger')
        return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.clear() 
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    # Protect this page: user must be logged in
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'danger')
        return redirect(url_for('home'))
    
    # Get the user's past predictions from the database
    initialize_database()
    user_id = session['user_id']
    predictions = Prediction.query.filter_by(user_id=user_id).order_by(Prediction.timestamp.desc()).all()
    
    # Show the dashboard page
    return render_template('dashboard.html', username=session['username'], predictions=predictions)


# --- API Route (For the JavaScript) ---

@app.route("/predict", methods=['POST'])
def predict():
    # Protect this API: user must be logged in
    if 'user_id' not in session:
        return jsonify({'error': 'User not logged in'}), 401 # 401 is "Unauthorized"

    initialize_database()
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        image_bytes = file.read()
        try:
            predicted_class_index, severity_name = predict_severity(
                image_bytes,
                file.mimetype
            )
        except RuntimeError as error:
            return jsonify({'error': str(error)}), 502
        is_dr_detected = (predicted_class_index != 0)
        
        # --- SAVE TO DATABASE ---
        user_id = session['user_id']
        new_prediction = Prediction(
            result_text=severity_name,
            result_level=int(predicted_class_index),
            is_detected=bool(is_dr_detected),
            author=User.query.get(user_id)
        )
        db.session.add(new_prediction)
        db.session.commit()
        # ---
        
        return jsonify({
            'dr_detected': bool(is_dr_detected),
            'severity': severity_name,
            'severity_level': int(predicted_class_index)
        })

# --- Run the App ---
if __name__ == '__main__':
    app.run(
        debug=os.getenv('APP_ENV') != 'production',
        host='0.0.0.0',
        port=int(os.getenv('PORT', '5000'))
    )