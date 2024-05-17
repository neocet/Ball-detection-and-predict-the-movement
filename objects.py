import cv2
import supervision as sv
import numpy as np
import requests
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression


class RingBuffer:

  """
  A circular buffer to store a fixed number of (x, y) coordinate pairs.

  Attributes:
    size (int): The maximum number of pairs to store in the buffer.
    buffer (list): A list to store the (x, y) coordinate pairs.
    index (int): The current index in the buffer to add the next pair.
  """
  
  def __init__(self, size):
    self.size = size
    self.buffer = [(None, None)] * size
    self.index = 0

  def add(self, x, y):
    """
    Add a new (x, y) coordinate pair to the buffer.

    Args:
      x (float): The x-coordinate.
      y (float): The y-coordinate.
    """
    self.buffer[self.index] = (x, y)
    self.index = (self.index + 1) % self.size

  def get(self):
    """
      Get the current buffer as a list of (x, y) coordinate pairs.

      Returns:
        list: A list of (x, y) coordinate pairs.
    """
    return self.buffer

  def get_linear_regression(self):
    """
      Perform linear regression on the (x, y) coordinate pairs in the buffer.

      Returns:
        LinearRegression: A fitted linear regression model.
    """
    x_values = [x for x, _ in self.buffer]
    y_values = [y for _, y in self.buffer]
    x_values = np.array(x_values).reshape(-1, 1)
    y_values = np.array(y_values)
    model = LinearRegression()
    model.fit(x_values, y_values)
    return model
  
class RangeCalculator:

  """
  Calculate the expected range based on the ball's position and goal parameters.

  Attributes:
    goalSize (float): The size of the goal.
    nearPost (float): The position of the near goal post.
    farPost (float): The position of the far goal post.
    goalPosition (float): The desired position of the ball relative to the goal.
  """
  
  def __init__(self, goalSize, nearPost, farPost, goalPosition):
    self.goalSize = goalSize
    self.nearPost = nearPost
    self.farPost = farPost
    self.goalPosition = goalPosition

  def moveRange(self, x):
    """
    Calculate the expected range based on the ball's x-coordinate.

    Args:
      x (float): The x-coordinate of the ball.

    Returns:
      float: The expected range.
    """
    coef = self.goalSize / (self.farPost - self.nearPost)
    if x < self.nearPost:
      expectedRange = 0
    elif x > self.farPost:
      expectedRange = 30
    else:
      expectedRange = (x - self.nearPost) * coef
    return expectedRange

class WebcamCapturer:

  """
    Open and manage the webcam for capturing frames.

    Attributes:
      resolution (tuple): The desired resolution of the webcam frames.
      fps (int): The desired frames per second of the webcam.
      cap (cv2.VideoCapture): The VideoCapture object for the webcam.
  """

  def __init__(self, resolution=(1280, 720), fps=60): # Change resolution and fps to use the desired settings
    self.resolution = resolution
    self.fps = fps

  def open_webcam(self, cam_index=1): # Change cam_index to use the desired webcam
    """
    Open the webcam and set the desired resolution and FPS.

    Args:
      cam_index (int, optional): The index of the webcam to open. Default is 0.
    """
    self.cap = cv2.VideoCapture(cam_index)
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
    self.cap.set(cv2.CAP_PROP_FPS, self.fps)

  def get_frame(self):
    """
    Capture a single frame from the webcam.

    Returns:
      numpy.ndarray: The captured frame as a NumPy array.
    """
    ret, frame = self.cap.read()
    return frame

  def release(self):
    """
    Release the webcam resources.
    """
    self.cap.release()

class HTTPClient:

  """
  Send HTTP GET requests to a specified URL.

  Attributes:
    url (str): The URL to send the requests to.
  """

  def __init__(self, url):
    self.url = url

  def send_request(self, data):
    """
    Send an HTTP GET request with the provided data.

    Args:
      data (any): The data to send in the request.
    """
    params = {"data": data} # Change the parameter name to match the server's expected parameter name
    try:
      response = requests.get(self.url, params=params)
      response.raise_for_status()
    except requests.exceptions.RequestException as e:
      pass

class BallDetector:

  """
  Detect the ball in a frame using the YOLO model and annotate the frame.

  Attributes:
    model (YOLO): The YOLO model for object detection.
    box_annotator (sv.BoundingBoxAnnotator): The annotator for drawing bounding boxes.
    label_annotator (sv.LabelAnnotator): The annotator for drawing labels.
    dot_annotator (sv.DotAnnotator): The annotator for drawing dots.
    circle_annotator (sv.CircleAnnotator): The annotator for drawing circles.
  """

  def __init__(self, model_path):
    self.model = YOLO(model_path)
    self.box_annotator = sv.BoundingBoxAnnotator()
    self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
    self.dot_annotator = sv.DotAnnotator()
    self.circle_annotator = sv.CircleAnnotator()

  def detect_ball(self, frame):

    """
    Detect the ball in the given frame and annotate the frame.

    Args:
      frame (numpy.ndarray): The frame to detect the ball in.

    Returns:
      tuple: A tuple containing the annotated frame and the detection results.
    """

    results = self.model.predict(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    frame = self.label_annotator.annotate(frame, detections)
    frame = self.dot_annotator.annotate(frame, detections)
    frame = self.circle_annotator.annotate(frame, detections)

    return frame, results
