# LIBRAS Decoder: Tracking and Recognizing Gesture Alphabet

Most of gesture alphabet detectors establish a fixed bounding box to capture the hand gesture and further classify it into the corresponding sign. Nevertheless, such an approach might be inefficient when dealing with real-world scenarios as users should freely express their gestures without being contained in a marking box. Therefore, our approach relies on a hand-tracking system that can freely detect a human hand, establish its bounding box, and perform the classification over the expressed gesture.

*This repository is a work in progress, and we intend to keep fostering accessibility-based research.*

---

## Structure
  * `classifiers`
    * `small.py`: A small-sized Convolutional Neural Network, used for easy tasks;
    * `medium.py`: A medium-sized Convolutional Neural Network, used for more complex tasks;
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

---

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```pip install -r requirements.txt```

### Environment configuration

Check the `utils/constants.py` file in order to correct set where data and pre-trained models should be saved and loaded.

---

## Usage

### Pre-Train Hand Detection Models

The first step to recognize the gesture alphabet lies in identifying and tracking hand's movement. To accomplish such a procedure, one needs to [pre-train](https://github.com/jkjung-avt/hand-detection-tutorial) its hand detection model (usually performed with the Egohands dataset).

### Train and Evaluate Classification Model

After detecting and tracking the hand's movement, it is possible to snapshot the hand and pass it through a classification network, such as a Convolutional Neural Network. To accomplish such a step, one needs to download the LIBRAS dataset provided by [LASIC/UEFS](http://sites.ecomp.uefs.br/lasic/projetos/libras-dataset).

With the dataset in hands, one can pre-train its classification model using the following script:

```python train_classifier.py -h```

Additionally, as an optional procedure, one can evaluate its classification model, as follows:

```python evaluate_classifier.py -h```

*Note that `-h` invokes the script helper, which assists users in employing the appropriate parameters.*

### Detect and Classify a Video Stream

Finally, it is now possible to stream a video from the webcam and perform real-time detection and classification. Use the following script to accomplish such a procedure:

```python detect_classify_stream.py -h```

---

## Acknowledgements

We are glad to acknowledge two important sources for conducting our research, as follows:

* TF Object Detection API Wrapper: https://github.com/jkjung-avt/hand-detection-tutorial

* LIBRAS Dataset: I. L. O. Bastos, M. F. Angelo and A. C. Loula, Recognition of Static Gestures Applied to Brazilian Sign Language (Libras). 28th SIBGRAPI Conference on Graphics, Patterns and Images, Salvador, 2015, pp. 305-312.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---
