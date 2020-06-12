# LIBRAS Decoder: Tracking and Recognizing Gesture Alphabet

Most of gesture alphabet detectors establish a fixed bounding box to capture the hand gesture and further classify it into the corresponding sign. Nevertheless, such an approach might be inefficient when dealing with real-world scenarios as users should freely express their gestures without being contained in a marking box. Therefore, our approach relies on a hand-tracking system that can freely detect a human hand, establish its bounding box, and perform the classification over the expressed gesture.

*This repository is a work in progress, and we intend to keep fostering accessibility-based research.*

## Structure
  * `classifiers`
    * `small.py`: A small-sized Convolutional Neural Network, used for easy tasks;
  * `core`
    * `classifier.py`: Defines the classifier class, used to recognize the gestures;
    * `detector.py`: Defines the detector class, used to track the hands;
    * `stream.py`: Defines the stream class, used to stream a video from an input webcam;
  * `data`: A place that gathers datasets used in the classification task;
  * `models`: A place that holds pre-trained detection and classification models;
  * `utils`
    * `compressor.py`: Helper that eases the compression and de-compression of .tar.gz files;
    * `constants.py`: Constants definitions;
    * `dictionary.py`: Dictionary that maps between classes and labels;
    * `loader.py`: Loads classification-based datasets into generators;
    * `processor.py`: Processing utilities, such as detecting bound boxes, drawing labels and creating binary masks.

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```pip install -r requirements.txt```

### Environment configuration

Check the `utils/constants.py` file in order to correct set where data and pre-trained models should be saved and loaded.

## Usage

### Download Pre-Trained Hand Detection Models

### Train and Evaluate Classification Model

### Detect and Classify a Video Stream

## Acknowledgements

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.