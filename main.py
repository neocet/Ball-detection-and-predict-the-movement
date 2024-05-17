'''
This program was developed for the purpose of a Laboratiom Teknik Fisika II project at the Institut Teknologi Bandung.
This program essentially detects and predicts the direction of a ball heading towards a goal. 
The program's output provides prediction data to the client via a Wi-Fi network.
'''

from objects import *


def main():

  """
  The main function to run the ball detection and range calculation.
  """

  # Change the buffer size to store more or fewer coordinate pairs
  buffer = RingBuffer(3)

  # The goal parameters is (length of goal, near post in normalize x coordinate, far post in normalize x coordinate, goal position in normalize y coordinate)
  range_calculator = RangeCalculator(30, 0.2, 0.8, 0.0)

  # Open the webcam
  webcam_capturer = WebcamCapturer()
  webcam_capturer.open_webcam()

  # Initialize the HTTP client to send requests to the server (change the URL to match the server's URL)
  http_client = HTTPClient("http://192.168.4.1")

  # Initialize the ball detector with the desired YOLO model
  ball_detector = BallDetector('ball_v1.pt')

  while True:
    '''
    The main loop to capture frames from the webcam, detect the ball, calculate the range, and send the prediction to the server.
    Press 'q' to exit the loop and close the program.
    '''

    # Get the frame from the webcam and detect the ball
    frame = webcam_capturer.get_frame()
    annotated_frame, results = ball_detector.detect_ball(frame)

    # Annotate the frame with the normalized coordinates
    for r in results:
      boxes = r.boxes.xyxy
      boxes = boxes.cpu().numpy()

      # Get the coordinates of the ball
      for box in boxes:
        x1, y1, x2, y2 = box

        # Normalize coordinates
        frame_width, frame_height = webcam_capturer.resolution
        x = round((x1 + x2) / 2 / frame_width, 3)
        y = round((y1 + y2) / 2 / frame_height, 3)

        # Annotate the frame with the normalized coordinates
        annotated_frame = cv2.putText(annotated_frame, f"({x}, {y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # uncomment the following line to see the normalized coordinates
        # print(f"Normalized coordinates: ({x}, {y})")

        # Add the coordinates to update the buffer
        buffer.add(x, y)

        # Get the buffer values 
        buffer_values = buffer.get()
        x_values = [x for x, _ in buffer_values]
        y_values = [y for _, y in buffer_values]
        # uncomment the following line to see the buffer values
        # print(x_values, y_values)

        # Perform linear regression if there are enough values in the buffer
        if None not in x_values:
          linear_regression_model = buffer.get_linear_regression()
          predicted_x = (range_calculator.goalPosition - linear_regression_model.intercept_) / linear_regression_model.coef_
          range_value = range_calculator.moveRange(predicted_x[0])

          # Send the predicted range to the server
          http_client.send_request(round(range_value, 2))

    # Annotate the frame with a message if no ball is detected
    if len(results) == 0:
      annotated_frame = cv2.putText(annotated_frame, "No ball detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Detection Ball', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  # Release the webcam resources and close the OpenCV windows
  webcam_capturer.release()
  cv2.destroyAllWindows()



# Run the main function if the script is executed
if __name__ == '__main__':
  main()