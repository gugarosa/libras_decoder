import cv2
import imutils

from utils.video_stream import VideoStream

# Starts a thread from the `VideoStream` class
v = VideoStream().start_thread()

# While the loop is True
while True:
    # Reads a new frame
    frame = v.read()

    # Resizes the frame
    frame = imutils.resize(frame, width=400)

    # Shows the frame using `open-cv`
    cv2.imshow("Frame", frame)

    # If the `q` key is inputted, breaks the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Destroy all windows for cleaning up memory
cv2.destroyAllWindows()

# Stop the thread
v.stop_thread()
