import numpy as np
import gradio as gr
import tensorflow as tf

MODEL_PATH = 'cnn_model_1.h5'
IMG_SIZE = 224
SEVERITY_CLASSES = {
    0: 'No DR',
    1: 'Mild DR',
    2: 'Moderate DR',
    3: 'Severe DR',
    4: 'Proliferative DR'
}

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def classify(image):
    if image is None:
        raise gr.Error('Please upload a retinal image.')

    pixels = np.asarray(image.convert('RGB'), dtype=np.float32)
    pixels = tf.image.resize(pixels, [IMG_SIZE, IMG_SIZE]) / 255.0
    prediction = model.predict(tf.expand_dims(pixels, axis=0), verbose=0)[0]
    return {
        SEVERITY_CLASSES[index]: float(score)
        for index, score in enumerate(prediction)
    }

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type='pil'),
    outputs=gr.JSON(),
    api_name='predict',
    title='Diabetic Retinopathy Classifier'
)

demo.launch()
