import json
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import np_utils

from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.layers import Dense, Flatten, Dropout, LeakyReLU, BatchNormalization, Reshape, Input, LSTM, Lambda


class NameGAN:

    def __init__(self, filename):
        self.filename = filename
        self.names_data = []
        self.g_output_shape = 10
        self.noise_shape = (1, )

        self.do_one_hot = False
        if self.do_one_hot:
            self.g_output_shape = 30

        self.do_scaling = not self.do_one_hot

        self.load_names()

        optimizer = Adam(0.00001, decay=0.000001)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='categorical_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated imgs
        z = Input(shape=self.noise_shape)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='categorical_crossentropy', optimizer=optimizer)

        self.combined.summary()
        self.tmp_noise = None

    def load_names(self):

        with open("./data/{0}.json".format(self.filename), "r", encoding="utf8") as f:
            names = f.read()

        names = json.loads(names)
        names = [x.lower() for x in names if len(x) <= 10]

        chars = sorted(list(set(" ".join(names))))
        char_to_num = dict((c, i) for i, c in enumerate(chars))

        out = []
        for n in names:
            n = [char_to_num[n[i]] if len(n) > i else char_to_num[" "] for i in range(10)]
            out.append(n)

        self.chars = chars
        self.chars_to_num = char_to_num
        self.nums_to_char = {v: k for k, v in self.chars_to_num.items()}
        self.names_data = np.array(out)

        if self.do_one_hot:
            self.names_data = np_utils.to_categorical(self.names_data, self.g_output_shape)

        if self.do_scaling:
            self.names_data = self.names_data / max(self.nums_to_char.keys())

    def build_generator(self):

        model = Sequential()

        model.add(Dense(512, input_shape=self.noise_shape, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        #model.add(Dense(64, activation="relu"))
        #model.add(Dense(128, activation="relu"))
        #model.add(Dense(256, activation="relu"))
        #model.add(Dense(512, activation="relu"))
        #model.add(Dense(self.g_output_shape))

        if self.do_one_hot:
            model.add(Dense(500, activation="relu"))
            model.add(Reshape((10, 50)))
            model.add(Dense(self.g_output_shape, activation="softmax"))
        else:
            model.add(Dense(self.g_output_shape))

        model.summary()

        noise = Input(shape=self.noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_shape=self.names_data.shape[1:], activation="relu"))
        #model.add(Dense(512, input_shape=self.names_data.shape[1:], activation="relu"))
        if self.do_one_hot:
            model.add(Flatten())
        model.add(Dense(1024, activation="relu"))
        #model.add(Dense(512, activation="relu"))
        #model.add(Dense(128, activation="relu"))
        #model.add(Dense(256, activation="relu"))
        #model.add(Dense(512, activation="relu"))

        #model.add(Dense(1, activation="sigmoid"))
        model.add(Dense(2, activation="softmax"))
        model.summary()

        img = Input(shape=self.names_data.shape[1:])
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs=1000, batch_size=128):

        half_batch = int(batch_size / 2)
        x = np.array(self.names_data)

        for epoch in range(epochs):

            # Select a random half batch of data
            nms = x[np.random.randint(0, x.shape[0], half_batch)]

            #noise = np.random.normal(0, 1, (half_batch, 100))
            noise = self.generate_noise(half_batch)

            # Generate a half batch of new data
            gen_nms = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(nms, np_utils.to_categorical(np.ones((half_batch, 1)), 2))
            d_loss_fake = self.discriminator.train_on_batch(gen_nms, np_utils.to_categorical(np.zeros((half_batch, 1)), 2))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            #noise = np.random.normal(0, 1, (batch_size, 100))
            noise = self.generate_noise(batch_size)

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)
            valid_y = np_utils.to_categorical(valid_y, 2)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % 1000 == 0:
                print("Epoch:", epoch)
                print(self.generate_random())

    def generate_noise(self, num):

        if self.tmp_noise is None:
            self.tmp_noise = np.random.normal(0, 1, (len(self.names_data), self.noise_shape[0]))


        noise = self.tmp_noise[np.random.randint(0, len(self.tmp_noise), num)]

        #noise = np.random.uniform(0, 1, (num, self.noise_shape[0]))
        #noise = np.random.normal(0, 1, (num, self.noise_shape[0]))

        #noise = n_n
        """
        n_u = np.random.uniform(0, 1, (num, self.noise_shape[0]))
        n_n = np.random.normal(0, 1, (num, self.noise_shape[0]))
        n_u = n_u[np.random.choice(n_u.shape[0], size=num//2, replace=False)]
        n_n = n_n[np.random.choice(n_n.shape[0], size=num//2, replace=False)]
        noise = np.concatenate((n_u, n_n))
        np.random.shuffle(noise)
        """
        #noise = np.random.normal(4, 1, (num, 100))

        #print(min(noise.flatten()), max(noise.flatten()))

        return noise

    def generate_random(self, num=10):

        noise = self.generate_noise(num)

        gen_nms = self.generator.predict(noise)

        if self.do_scaling:
            gen_nms *= max(self.nums_to_char.keys())

        if self.do_one_hot:
            gen_nms = np.argmax(gen_nms, axis=2)

        out = []
        for gn in gen_nms:

            gn = np.rint(gn).astype(int).clip(0, max(self.nums_to_char.keys()))

            n = "".join([self.nums_to_char[gn[i]] for i in range(10)])

            out.append(n)
            #print(n)

        return out


if __name__ == "__main__":

    #gan = NameGAN(filename="cro_names")
    #gan = NameGAN(filename="eng_names")
    gan = NameGAN(filename="univ_names")

    gan.train(epochs=30000, batch_size=32)
    gan.generate_random()
