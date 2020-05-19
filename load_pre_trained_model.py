import utils.loader as l

# Defining a constant for the base URL and the model's name
BASE_URL = 'http://recogna.tech/files'
MODEL_NAME = 'ssd_mobilenet_v1_egohands'

# Defining a constant for the base URL and the model's name (Tensorflow Zoo)
# BASE_URL = 'http://download.tensorflow.org/models/object_detection'
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

# Loading the model from web
model = l.load_from_web(MODEL_NAME, BASE_URL)
