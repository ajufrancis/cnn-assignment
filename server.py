from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from threading import Thread

app = Flask(__name__)

# Global variables to store training progress
training_data = {
    'epoch': 0,
    'train_loss': [],
    'train_accuracy': [],
    'test_accuracy': []
}
examples_received = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update', methods=['POST'])
def update():
    global training_data
    epoch = int(request.form.get('epoch'))
    train_loss = float(request.form.get('train_loss'))
    train_accuracy = float(request.form.get('train_accuracy'))
    test_accuracy = float(request.form.get('test_accuracy'))

    training_data['epoch'] = epoch
    training_data['train_loss'].append(train_loss)
    training_data['train_accuracy'].append(train_accuracy)
    training_data['test_accuracy'].append(test_accuracy)

    # Save the plot
    plot = request.files['plot']
    plot.save(os.path.join('static', 'img', 'plot.png'))

    return 'Update received', 200

@app.route('/update/examples', methods=['POST'])
def update_examples():
    global examples_received
    examples = request.files['examples']
    examples.save(os.path.join('static', 'img', 'examples.png'))
    examples_received = True
    return 'Examples received', 200

@app.route('/data')
def data():
    return jsonify(training_data)

@app.route('/examples_received')
def check_examples():
    return jsonify({'examples_received': examples_received})

def run_app():
    app.run(host='0.0.0.0', port=5500)

if __name__ == '__main__':
    # Ensure the 'static/img' directory exists
    os.makedirs(os.path.join('static', 'img'), exist_ok=True)

    # Run the Flask app in a separate thread if needed
    run_app()

