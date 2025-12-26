import tensorflow as tf
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, get_flashed_messages

from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import numpy as np
import io
import os
from datetime import datetime

# --- App Configuration ---

# Get the absolute path of the directory where the script is located
#basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
CORS(app)

# Secret key is needed for user sessions
app.config['SECRET_KEY'] = 'your_super_secret_key_change_this' 
# Configure the SQLite database
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'site.db')
# Format: 'mysql://USERNAME:PASSWORD@HOST/DATABASE_NAME'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:634237@localhost/dr_analyzer'

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
model = tf.keras.models.load_model('D:\Diabetic-Retinopathy\.venv\cnn_model_1.h5')
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


# --- Page Routes (URLs) ---

@app.route("/")
def home():
    # If user is already logged in, send them to the dashboard
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    # Otherwise, show the login page
    return render_template('login.html')

@app.route("/register", methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    
    # Check if user already exists
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        flash('Username already exists. Please log in.', 'danger')
        return redirect(url_for('home'))

    # Hash the password
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    # Create new user
    user = User(username=username, password=hashed_password)
    db.session.add(user)
    db.session.commit()
    
    flash('Account created! You can now log in.', 'success')
    return redirect(url_for('home'))


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
    # Consume/clear any leftover flash messages so old ones don't reappear later
    _ = get_flashed_messages(with_categories=True)

    # Remove logged-in user
    session.pop("user", None)

    # Flash the logout message you want shown now
    flash("You have been logged out.", "info")

    return redirect(url_for("login"))


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