from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


def get_autoencoder_model(img_width=64, img_height=64):
    autoencoder = Sequential()
    # Encoder
    autoencoder.add(Input(shape=(img_width, img_height, 3)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    # Decoder
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))
    autoencoder.compile(
        optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    autoencoder.summary()
    plot_model(autoencoder, to_file='model.png',
               show_shapes=True, show_layer_names=True)
    return autoencoder


def get_autoencoder_model128(img_width=128, img_height=128):  # Built for 128x128
    autoencoder = Sequential()
    autoencoder.add(Input(shape=(img_width, img_height, 3)))
    autoencoder.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    # Decoder
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(3, (5, 5), activation='sigmoid', padding='same'))
    autoencoder.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    autoencoder.summary()
    return autoencoder
