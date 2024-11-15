# How to Run the MNIST CNN with Live Training Visualization

This guide will walk you through the steps to set up and run the convolutional neural network (CNN) training on the MNIST dataset with live training visualization using Flask.
The web interface will display loss curves and accuracy metrics in real-time and show the results of 10 random images after training.

# Prerequisites
- Python 3.x installed on your system.
- CUDA-compatible GPU (optional): If you have a GPU and want to use CUDA for faster training.
- Python Packages:
- torch
- torchvision
- matplotlib
- tqdm
- flask
- requests


Setup Instructions
1. Clone or Download the Project Files

Create a new directory for the project and ensure it has the following structure:
```csharp
your_project/
├── train.py
├── server.py
├── HowTo.md
├── templates/
│   └── index.html
└── static/
    ├── img/
    │   ├── plot.png      # Will be generated during training
    │   └── examples.png  # Will be generated after training
```


- train.py: The training script.
- server.py: The Flask server script.
- templates/index.html: The HTML template for the web interface.
- static/img/: Directory for images (plots and examples).

# Running the Application
Step 1: Start the Flask Server

In the terminal, run:
```
python server.py
```
- The server will start on http://localhost:5000.
- Note: Keep this terminal open as the server needs to run continuously.

Step 2: Run the Training Script

Open another terminal in the same project directory and run:
```
python train.py
```
- The script will start training the CNN on the MNIST dataset.
- It will send updates to the Flask server after each epoch.
- After training, it will send the results of 10 random images.
