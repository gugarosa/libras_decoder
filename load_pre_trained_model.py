import utils.loader as l

# Defining a constant for the model's name
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

# Loading the model from Tensorflow's Zoo
model = l.load_from_zoo(MODEL_NAME)
