import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Activation, Conv2D, Conv2DTranspose, Flatten, Reshape, \
    BatchNormalization, Embedding, multiply, Dropout, Concatenate, ZeroPadding2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from utils.data import DataLoader
from utils import setpath as path
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import BinaryCrossentropy
import os


class Pix2pix:
    def __init__(self, model_name=None, config=None):
        if model_name == None:
            self.modelname = self.__class__.__name__
        else:
            self.modelname = model_name
        path.train_folder_process()
        config.save_to_json(path=f'{path.LOG_CONFIGPATH}/config.json')
        self.gen_lr = config.gen_lr
        self.dis_lr = config.dis_lr
        self.epochs = config.epochs
        self.batchsize = config.batchsize

        self.binary_crossentropy_loss = BinaryCrossentropy(from_logits=True)
        self.lambda_ = config.lambda_

        self.genModel = self.build_generator()
        self.disModel = self.build_discriminator()
        self.advModel = self.build_adversialmodel()
        self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        self.disc_patch = (int(256 / 2 ** 4), int(256 / 2 ** 4), 1)
        self.sketchdata, self.realdata = DataLoader().load_data(metrics='[0,1]')

        self.gentotalloss = []
        self.genganloss = []
        self.genl1loss = []
        self.distotalloss = []
        self.summary_writer = tf.summary.create_file_writer(path.DATEPATH+'/fit')

    def encoder_block(self, input_, u, k, s=2, p='same', bn=True, act='leakyrelu'):
        x = Conv2D(u, kernel_size=k, strides=s, padding=p, use_bias=False, kernel_initializer='he_uniform')(input_)
        if bn:
            x = BatchNormalization(momentum=0.8)(x)
        if act == 'leakyrelu':
            x = LeakyReLU(alpha=0.2)(x)
        return x

    def decoder_block(self, input_, u, k, s=2, p='same', bn=True, dropout=True, act='relu'):
        x = Conv2DTranspose(u, kernel_size=k, strides=s, padding=p, use_bias=False, kernel_initializer='he_uniform')(
            input_)
        if bn:
            x = BatchNormalization(momentum=0.8)(x)
        if dropout:
            x = Dropout(0.5)(x)
        if act == 'relu':
            x = Activation('relu')(x)
        elif act == 'leakyrelu':
            x = LeakyReLU(alpha=0.2)(x)
        elif act == 'sigmoid':
            x = Activation('sigmoid')(x)
        return x

    # @property
    def build_generator(self):
        input_layer = Input(shape=(256, 256, 3))
        en1 = self.encoder_block(input_layer, u=64, k=4, bn=False)
        en2 = self.encoder_block(en1, u=128, k=4)
        en3 = self.encoder_block(en2, u=256, k=4)
        en4 = self.encoder_block(en3, u=512, k=4)
        en5 = self.encoder_block(en4, u=512, k=4)
        en6 = self.encoder_block(en5, u=512, k=4)
        en7 = self.encoder_block(en6, u=512, k=4)
        latent = self.encoder_block(en7, u=512, k=4)
        de1 = self.decoder_block(latent, u=512, k=4)
        de1 = Concatenate(axis=-1)([de1, en7])
        de2 = self.decoder_block(de1, u=512, k=4)
        de2 = Concatenate(axis=-1)([de2, en6])
        de3 = self.decoder_block(de2, u=512, k=4)
        de3 = Concatenate(axis=-1)([de3, en5])
        de4 = self.decoder_block(de3, u=512, k=4, dropout=False)
        de4 = Concatenate(axis=-1)([de4, en4])
        de5 = self.decoder_block(de4, u=256, k=4, dropout=False)
        de5 = Concatenate(axis=-1)([de5, en3])
        de6 = self.decoder_block(de5, u=128, k=4, dropout=False)
        de6 = Concatenate(axis=-1)([de6, en2])
        de7 = self.decoder_block(de6, u=64, k=4, dropout=False)
        de7 = Concatenate(axis=-1)([de7, en1])
        out = self.decoder_block(de7, u=3, k=4, dropout=False, bn=False, act='sigmoid')

        model = Model(inputs=input_layer, outputs=out)
        # model.summary()
        plot_model(model, to_file=f'{path.LOG_MODELPATH}/{self.modelname}_generator.png', show_shapes=True)
        # model.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=Adam(0.0002, 0.5))
        return model

    def generator_loss(self, d_loss_fake, gen_output, target):
        gen_loss = self.binary_crossentropy_loss(tf.ones_like(d_loss_fake), d_loss_fake)
        l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
        total_gen_loss = gen_loss + (self.lambda_ * l1_loss)
        return total_gen_loss, gen_loss, l1_loss

    def discriminator_loss(self, d_loss_real, d_loss_fake):
        real_loss = self.binary_crossentropy_loss(tf.ones_like(d_loss_real), d_loss_real)
        gen_loss = self.binary_crossentropy_loss(tf.zeros_like(d_loss_fake), d_loss_fake)
        total_disc_loss = real_loss + gen_loss
        return total_disc_loss

    def build_discriminator(self):
        sketch_input_layer = Input(shape=(256, 256, 3))
        real_input_layer = Input(shape=(256, 256, 3))
        input_layer = Concatenate(axis=-1)([real_input_layer, sketch_input_layer])
        x = self.encoder_block(input_layer, u=64, k=4, bn=False)
        x = self.encoder_block(x, u=128, k=4)
        x = self.encoder_block(x, u=256, k=4)
        x = ZeroPadding2D()(x)
        x = self.encoder_block(x, u=512, k=4, s=1, p='valid')
        x = ZeroPadding2D()(x)
        out = Conv2D(1, kernel_size=4, strides=1, padding='valid', kernel_initializer='he_uniform')(x)

        model = Model(inputs=[real_input_layer, sketch_input_layer], outputs=out)
        # model.summary()
        plot_model(model, to_file=f'{path.LOG_MODELPATH}/{self.modelname}_discriminator.png', show_shapes=True)
        # model.compile(loss='mse', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
        return model

    def build_adversialmodel(self):
        sketch_input_layer = Input(shape=(256, 256, 3), name='sketch')
        real_input_layer = Input(shape=(256, 256, 3), name='real')
        sketch2real = self.genModel(sketch_input_layer)
        self.disModel.trainable = False
        valid = self.disModel([sketch2real, sketch_input_layer])
        model = Model(inputs=[real_input_layer, sketch_input_layer], outputs=[valid, sketch2real])
        # model.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=Adam(0.0002, 0.5))
        # model.summary()
        plot_model(model, to_file=f'{path.LOG_MODELPATH}/{self.modelname}_adversial.png', show_shapes=True,
                   show_layer_names=True)
        # Calculate output shape of D (PatchGAN)

        return model

    @tf.function
    def train_step(self, input_image, target, step):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.genModel(input_image, training=True)

            disc_real_output = self.disModel([input_image, target], training=True)
            disc_generated_output = self.disModel([input_image, gen_output], training=True)
            gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
            disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                self.genModel.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     self.disModel.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.genModel.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.disModel.trainable_variables))

        if step%500==0:
            pass
        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_total_loss, step=step // 1000)
            tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step // 1000)
            tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step // 1000)
            tf.summary.scalar('disc_loss', disc_loss, step=step // 1000)


    def finish_training(self):
        self.genModel.save(f'{path.RESULT_H5PATH}/{self.modelname}_generator.h5')
        self.disModel.save(f'{path.RESULT_H5PATH}/{self.modelname}_discriminator.h5')
        self.advModel.save(f'{path.RESULT_H5PATH}/{self.modelname}_adversarial.h5')
        # plt.clf()
        # plt.title('loss')
        # plt.plot(self.gentotalloss, label='g')
        # plt.plot(self.genganloss, label='g_gan')
        # plt.plot(self.genl1loss, label='g_l1')
        # plt.plot(self.distotalloss, label='d')
        # np.save(f'{path.LOG_LOSSPATH}/gen_total_loss.npy', arr=self.gentotalloss)
        # np.save(f'{path.LOG_LOSSPATH}/gen_gan_loss.npy', arr=self.genganloss)#self.gen_gan_loss, self.gen_l1_loss
        # np.save(f'{path.LOG_LOSSPATH}/gen_l1_loss.npy', arr=self.genl1loss)
        # np.save(f'{path.LOG_LOSSPATH}/dloss.npy', arr=self.distotalloss)

        # plt.legend(loc='best')
        # plt.savefig(f"{path.LOG_LOSSPATH}/loss.png")

    def predict(self, num_images=3):
        test_sketchdata, test_realdata = DataLoader().load_testing_data(metrics='[0,1]')
        idx = np.random.randint(0, test_sketchdata.shape[0], num_images)
        model = load_model(f'./result/h5/{self.modelname}_generator.h5')
        generated_images = model(test_sketchdata[idx])
        plt.figure(figsize=(8, 8))
        plt.title(f'predict')
        plt.suptitle(f'The Images Of {self.modelname} Generate', fontsize=17)

        for j in range(num_images):
            plt.subplot(num_images, num_images, num_images * j + 1)
            plt.title('generated', fontsize=12)
            plt.imshow(generated_images[j])
            plt.axis("off")
            plt.subplot(num_images, num_images, num_images * j + 2)
            plt.title('sketch', fontsize=12)
            plt.imshow(test_sketchdata[j])
            plt.axis("off")
            plt.subplot(num_images, num_images, num_images * j + 3)
            plt.title('true', fontsize=12)
            plt.imshow(test_realdata[j])
            plt.axis("off")
        plt.savefig(f'{path.LOG_PREDICTPATH}/predict.png')
        # plt.show()

    def showinput(self, num_images=16, idx=[], data=None):
        # TODO 可以做新增資料夾&儲存輸入data 資料夾：trainpred,sketchinput,realinput (不要用epoch建立資料夾，影片會不好用)
        img = data[idx]
        plt.figure(figsize=(8, 8))
        plt.title(f'predict')
        for j in range(num_images):
            plt.subplot(4, 4, j + 1)
            plt.imshow(img[j, :, :, :])
            plt.axis("off")
        plt.show()

if __name__ == '__main__':
    import time
    from utils import config

    config = config.TrainConfig(epochs=10000,
                                batchsize=16,
                                gen_lr=0.0002,
                                dis_lr=0.0002)
    gan = Pix2pix(model_name='pix2pix', config=config)
    s = time.time()
    for i in range(config.epochs):
        print(f'Epochs: {i+1}')
        idx = np.random.randint(0, gan.sketchdata.shape[0], gan.batchsize)
        gan.train_step(gan.sketchdata[idx], gan.realdata[idx], i) # 10000 epochs->used 25826.565 seconds7.17hour
    e = time.time()
    gan.finish_training()
    print(f'used {e-s:.3f} seconds')
    gan.predict()

