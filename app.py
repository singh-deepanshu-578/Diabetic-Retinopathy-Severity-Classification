import tensorflow as tf
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, get_flashed_messages

from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import cv2
import numpy as np
import io
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Secret key is needed for user sessions
app.config['SECRET_KEY'] = 'your_super_secret_key_change_this' 
# Configure the SQLite database
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'site.db')
# Format: 'mysql://USERNAME:PASSWORD@HOST/DATABASE_NAME'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:password@root/database_name'

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


# --- Load The AI Model (Do this once) ---
model = tf.keras.models.load_model('cnn_model_1.h5')
print("Model loaded successfully.")

# Map for severity classes
SEVERITY_CLASSES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}

# --- Image Preprocessing Function ---
IMG_SIZE = 224
def preprocess_image(image_bytes):
    img = tf.image.decode_image(image_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0  # Normalize
    return tf.expand_dims(img, axis=0) # Add batch dimension

def is_valid_retina(image_bytes):
    # Bytes ko OpenCV image mein convert karein
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return False

    # 1. Darkness Check (Certificate white hota hai, Retina dark)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    # Agar image bahut bright hai (mean > 180), toh wo retina nahi hai
    if mean_brightness > 180:
        return False

    # 2. Red Channel Dominance (Retina mein Red color sabse zyada hota hai)
    # BGR format: 0=Blue, 1=Green, 2=Red
    avg_color = np.mean(img, axis=(0, 1))
    if avg_color[2] < avg_color[0] or avg_color[2] < avg_color[1]:
        return False

    return True

# --- Page Routes (URLs) ---

@app.route("/")
def home():
    session.clear()
    return render_template('login.html')

@app.route("/register", methods=['GET','POST'])
def register():
    if request.method == 'POST':
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

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        image_bytes = file.read()
        if not is_valid_retina(image_bytes):
            return jsonify({
                'error': 'Invalid Image! Please upload a proper eye fundus scan.',
                'dr_detected': False
            }), 400
        processed_image = preprocess_image(image_bytes)
        prediction_array = model.predict(processed_image)
        
        predicted_class_index = np.argmax(prediction_array[0])
        severity_name = SEVERITY_CLASSES[predicted_class_index]
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
    # Create the database tables if they don't exist
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0')