from threading import Thread

import cv2
import imutils


class Stream:
    """A Stream class enables an user from loading and streaming its own camera device.

    """

    def __init__(self, height, width, device=0):
        """Initialization method.

        Args:
            height (int): Height for the frames from the input device.
            width (int): Width for the frames from the input device.
            device (int): From which device the webcam should be loaded.

        """

        # Creating a tuple to hold the frame size
        self.size = (height, width)

        # Creating a property for the VideoCapture from `device`
        self.video = cv2.VideoCapture(device)

        # Initializing the first frame
        self.ret, self.frame = self.video.read()

        # Whenever the thread is running, maintain this property as True
        self.thread_running = True

    def start_thread(self):
        """Starts a video streamming thread.

        Returns:
            The class object itself.

        """

        # Initializes a thread with frame-by-frame capture
        Thread(target=self.capture).start()

        # Returns the whole object
        return self

    def read(self):
        """Reads a new frame.

        Returns:
            Whether frame is valid and the frame that has been read itself.

        """

        # Resizes the frame
        frame = imutils.resize(self.frame, height=self.size[0], width=self.size[1])

        # Flips the frame
        frame = cv2.flip(frame, 1)

        return self.ret, frame

    def capture(self):
        """Captures a new frame from the streamming video.

        Returns:
            Exits the capturing loop is the thread is not running anymore.

        """

        # While the loop is True
        while True:
            # Checks if the thread is still running
            if not self.thread_running:
                # If not, returns the control to the process
                return

            # Otherwise, reads the next frame
            (self.ret, self.frame) = self.video.read()

    def stop_thread(self):
        """Stops a video streamming thread.

        """

        # Defines the thread running property as False
        self.thread_running = False
