import utils.loader as l

# Defines the URL
URL = 'http://recogna.tech/files/hand_detection'

# Defines the file's name
FILE_NAME = 'ssd_mobilenet_v1_egohands.tar.gz'

# Downloads the file
l.download_model(URL, FILE_NAME)
