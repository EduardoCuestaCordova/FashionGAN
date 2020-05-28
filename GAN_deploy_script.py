import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

generator2 = keras.models.load_model('gen2')
num_images = 10
# Give the model a random seed
random_latent_vectors = tf.random.normal(shape=(num_images, 128))
# Then generate the images
generated_images = generator2(random_latent_vectors)
# Finally, save the images
for i in range(0, num_images):
  img = keras.preprocessing.image.array_to_img(generated_images[i])
  img.save("generated_img_{i}.png".format(i=i))