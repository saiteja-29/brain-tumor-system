import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from .config import IMG_SIZE

def get_data_generators(train_dir, val_dir):
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical'
    )

    return train_gen, val_gen

def get_test_generator(test_dir):
    from keras.preprocessing.image import ImageDataGenerator

    test_datagen = ImageDataGenerator(rescale=1./255)

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        shuffle=False   # VERY IMPORTANT
    )

    return test_gen