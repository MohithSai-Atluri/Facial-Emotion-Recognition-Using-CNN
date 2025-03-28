import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from os.path import isfile, join
# prevents appearance of tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class EMR:
  def __init__(self):
    self.target_classes = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

  def build_network(self):
      """
      Build the convnet.
      Input is 48x48
      3072 nodes in fully connected layer
      """ 
      self.network = input_data(shape = [None, 48, 48, 1])
      print("Input data     ",self.network.shape[1:])
      self.network = conv_2d(self.network, 64, 5, activation = 'relu')
      print("Conv1          ",self.network.shape[1:])
      self.network = max_pool_2d(self.network, 3, strides = 2)
      print("Maxpool1       ",self.network.shape[1:])
      self.network = conv_2d(self.network, 64, 5, activation = 'relu')
      print("Conv2          ",self.network.shape[1:])
      self.network = max_pool_2d(self.network, 3, strides = 2)
      print("Maxpool2       ",self.network.shape[1:])
      self.network = conv_2d(self.network, 128, 4, activation = 'relu')
      print("Conv3          ",self.network.shape[1:])
      self.network = dropout(self.network, 0.3)
      print("Dropout        ",self.network.shape[1:])
      self.network = fully_connected(self.network, 3072, activation = 'relu')
      print("Fully connected",self.network.shape[1:])
      self.network = fully_connected(self.network, len(self.target_classes), activation = 'softmax')
      print("Output         ",self.network.shape[1:])
      print("\n")
      # Generates a TrainOp which contains the information about optimization process - optimizer, loss function, etc
      self.network = regression(self.network,optimizer = 'momentum',metric = 'accuracy',loss = 'categorical_crossentropy')
      # Creates a model instance.
      self.model = tflearn.DNN(self.network,checkpoint_path = 'CNN_MODEL',max_checkpoints = 1,tensorboard_verbose = 2)
      # Loads the model weights from the checkpoint
  
  def load_trained_model(self, model_path = "CNN_MODEL"):
     # Ensure that you have built the network before loading
        self.model.load(model_path)
        print("Model loaded successfully.")
  
  # The CSV file usually has a column 'pixels' where each row is a space-separated string of pixel values.
  # Convert these strings into 48x48 grayscale images.
  def process_pixels(self, pixel_sequence):
    pixels = np.array([int(pixel) for pixel in pixel_sequence.split()], dtype="float32")
    # Normalize pixel values to [0, 1]
    pixels /= 255.0
    return pixels.reshape(48, 48, 1)

  def train_model(self):
    # Assume 'fer2013.csv' is in the same directory.
    data = pd.read_csv("fer2013.csv")
    # Filter rows where Usage column is "Training"
    train_data = data[data["Usage"] == "Training"]
    # Obtain all remaing colums for testing
    test_data = data[data["Usage"] != "Training"] 
    # Process all training images
    faces = np.array([self.process_pixels(pixels) for pixels in train_data["pixels"].tolist()])
    # One-hot encode the emotion labels (assumed to be in a column named 'emotion')
    from tflearn.data_utils import to_categorical
    emotions = to_categorical(train_data["emotion"], nb_classes=7)
    # Process test images and labels
    test_faces = np.array([self.process_pixels(pixels) for pixels in test_data["pixels"].tolist()])
    test_emotions = to_categorical(test_data["emotion"], nb_classes=7)
    self.model.fit(faces, emotions, n_epoch=40, validation_set=0.1, shuffle=True, show_metric=True, batch_size=64, snapshot_epoch=True)
    # Save the trained model weights
    self.model.save("CNN_MODEL")
    print("Model saved.") 
    # Evaluate returns a list of metrics defined in regression (here, accuracy).
    test_accuracy = self.model.evaluate(test_faces, test_emotions)
    print("Test Accuracy: {:.2f}%".format(test_accuracy[0] * 100))
    # Optionally, you can also get predictions and compare:
    predictions = self.model.predict(test_faces)
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(test_emotions, axis=1)
    manual_accuracy = np.mean(predicted_classes == actual_classes)
    print("Manual Computed Accuracy: {:.2f}%".format(manual_accuracy * 100))
    
  def predict(self, image):
    """
    Image is resized to 48x48, and predictions are returned.
    """
    if image is None:
      return None
    image = image.reshape([-1, 48, 48, 1])
    return self.model.predict(image)

  def load_model(self):
    """
    Loads pre-trained model.
    """
    if isfile("CNN_MODEL.tflearn.meta"):
      self.model.load("CNN_MODEL.tflearn")
    else:
        self.train_model()
        self.load_trained_model()