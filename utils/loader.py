import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_generator(data_path, height, width, batch_size=1, color='grayscale', label='sparse', shuffle=True, rescale=True):
    """
    """
    
    #
    if rescale:
        #
        generator = ImageDataGenerator(rescale=1./255)
    
    #
    else:
        #
        generator = ImageDataGenerator()

    #
    data = generator.flow_from_directory(directory=data_path, target_size=(height, width),
                                         batch_size=batch_size, color_mode=color, class_mode=label, shuffle=shuffle)

    return data