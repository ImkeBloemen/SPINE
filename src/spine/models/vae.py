# vae.py

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VariationalAutoencoder(tf.keras.Model):
    def __init__(self, input_dim, encoding_dim=4, **kwargs):
        super(VariationalAutoencoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        # Encoder layers
        self.encoder_dense = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.encoder_dense2 = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.z_mean_dense = layers.Dense(encoding_dim, name='z_mean')
        self.z_log_var_dense = layers.Dense(encoding_dim, name='z_log_var')
        self.sampling = Sampling()

        # Decoder layers
        self.decoder_dense1 = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.decoder_dense2 = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.decoder_dense3 = layers.Dense(input_dim, activation='sigmoid')

    def encode(self, x):
        x = self.encoder_dense(x)
        x = self.encoder_dense2(x)
        z_mean = self.z_mean_dense(x)
        z_log_var = self.z_log_var_dense(x)
        z = self.sampling([z_mean, z_log_var])
        return z_mean, z_log_var, z

    def decode(self, z):
        x = self.decoder_dense1(z)
        x = self.decoder_dense2(x)
        x = self.decoder_dense3(x)
        return x

    def call(self, inputs):
        z_mean, z_log_var, z = self.encode(inputs)
        reconstructed = self.decode(z)

        # Add KL divergence regularization loss
        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=1
        )
        self.add_loss(tf.reduce_mean(kl_loss))

        return reconstructed

def train_variational_autoencoder(X_scaled, input_dim, encoding_dim=6, epochs=100, batch_size=64):
    """Train the Variational Autoencoder (VAE) and return the trained model and encoder."""
    # Instantiate the VAE model
    vae = VariationalAutoencoder(input_dim, encoding_dim)

    # Compile the VAE
    vae.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    # Train the VAE
    vae.fit(
        X_scaled,
        X_scaled, 
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.1,
        verbose=1
    )

    # Extract the encoder part of the VAE
    # Create an encoder model that outputs z_mean
    inputs = tf.keras.Input(shape=(input_dim,))
    z_mean, _, _ = vae.encode(inputs)
    encoder = tf.keras.Model(inputs, z_mean)

    return vae, encoder
