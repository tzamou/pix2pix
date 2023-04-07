import time

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Activation, Conv2D, Conv2DTranspose, Flatten, Reshape, \
    BatchNormalization, Embedding, multiply, Dropout, Concatenate, ZeroPadding2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import utils
from utils.data import DataLoader
from utils import setpath as path
from tensorflow.keras.utils import plot_model


class Pix2pix:
    def __init__(self,model_name=None,config=None):
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

        self.genModel = self.build_generator()
        self.disModel = self.build_discriminator()
        self.advModel = self.build_adversialmodel()
        self.disc_patch = (int(256 / 2 ** 4), int(256 / 2 ** 4), 1)
        self.sketchdata, self.realdata =DataLoader().load_data(metrics='[0,1]')

    def encoder_block(self, input_, u, k, s=2, p='same', bn=True, act='leakyrelu'):
        x = Conv2D(u, kernel_size=k, strides=s, padding=p, use_bias=False, kernel_initializer='he_uniform')(input_)
        if bn:
            x = BatchNormalization(momentum=0.8)(x)
        if act == 'leakyrelu':
            x = LeakyReLU(alpha=0.2)(x)
        return x

    def decoder_block(self, input_, u, k, s=2, p='same', bn=True, dropout=True, act='relu'):
        x = Conv2DTranspose(u, kernel_size=k, strides=s, padding=p, use_bias=False, kernel_initializer='he_uniform')(input_)
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
        # plot_model(model, to_file=f'{utils.RESULT_MODELPATH}/{self.modelname}_generator.png', show_shapes=True)
        # model.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=Adam(0.0002, 0.5))
        return model

    def build_discriminator(self):
        sketch_input_layer = Input(shape=(256, 256, 3))
        real_input_layer = Input(shape=(256, 256, 3))
        input_layer = Concatenate(axis=-1)([real_input_layer, sketch_input_layer])
        x = self.encoder_block(input_layer, u=64, k=4, bn=False)
        x = self.encoder_block(x, u=128, k=4)
        x = self.encoder_block(x, u=128, k=4)
        x = self.encoder_block(x,u=512, k=4)
        out = Conv2D(1, kernel_size=4, strides=1, padding='same', kernel_initializer='he_uniform')(x)

        model = Model(inputs=[real_input_layer, sketch_input_layer],outputs=out)
        # model.summary()
        # plot_model(model,to_file=f'{utils.RESULT_MODELPATH}/{self.modelname}_discriminator.png', show_shapes=True)
        model.compile(loss='mse', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
        return model

    def build_adversialmodel(self):
        sketch_input_layer = Input(shape=(256, 256, 3),name='sketch')
        real_input_layer = Input(shape=(256, 256, 3),name='real')
        sketch2real = self.genModel(sketch_input_layer)
        self.disModel.trainable = False
        valid = self.disModel([sketch2real, sketch_input_layer])
        model = Model(inputs=[real_input_layer, sketch_input_layer], outputs=[valid, sketch2real])
        model.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=Adam(0.0002, 0.5))
        # model.summary()
        # plot_model(model, to_file=f'{utils.RESULT_MODELPATH}/{self.modelname}_adversial.png', show_shapes=True, show_layer_names=True)
        # Calculate output shape of D (PatchGAN)

        return model

    def train(self):
        self.dloss = []
        self.gloss = []

        for i in range(self.epochs):
            # self.sketchdata, self.realdata
            idx = np.random.randint(0, self.sketchdata.shape[0], self.batchsize)#;print(idx.shape)
            # print(i,idx)
            fake_images = self.genModel(self.sketchdata[idx])#;print(fake_images.shape)
            real_images = self.realdata[idx]#;print(real_images.shape)
            real_sketch = self.sketchdata[idx]

            y_real = np.ones((self.batchsize, ) + self.disc_patch)
            y_fake = np.zeros((self.batchsize, ) + self.disc_patch)
            # print(self.disModel([fake_images, real_images]).shape)
            d_loss_real = self.disModel.train_on_batch([real_images, real_sketch], y_real)
            d_loss_fake = self.disModel.train_on_batch([fake_images, real_sketch], y_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            self.dloss.append(d_loss[0])
            g_loss = self.advModel.train_on_batch([real_images, real_sketch], [y_real, real_images])
            # g_loss2 = self.advModel.train_on_batch([real_images, real_sketch], [y_real, real_images])
            # g_loss3 = self.advModel.train_on_batch([real_images, real_sketch], [y_real, real_images])
            self.gloss.append(g_loss)
            print(f"epochs:{i + 1} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]:.3f}] [G loss: {g_loss}]")
            if (i + 1) % 500 == 0 or i == 0 or (i+1 <= 3000 and (i+1)%40==0):
                # self.showinput(num_images=16,idx=idx,data=self.sketchdata)
                # self.showinput(num_images=16,idx=idx,data=self.realdata)
                num_images = self.batchsize
                # generated_images = 0.5 * generated_images + 0.5
                plt.figure(figsize=(8, 8))
                plt.title(f'{i + 1} epochs train')
                for j in range(num_images):
                    plt.subplot(4, 4, j + 1)
                    plt.imshow(fake_images[j, :, :, :])
                    plt.axis("off")
                plt.savefig(f"{path.IMAGES_TRAINPATH}/epoch{i + 1:0>5}.png")
        self.finish_training()

    def finish_training(self):
        self.genModel.save(f'{path.RESULT_H5PATH}/{self.modelname}_generator.h5')
        self.disModel.save(f'{path.RESULT_H5PATH}/{self.modelname}_discriminator.h5')
        self.advModel.save(f'{path.RESULT_H5PATH}/{self.modelname}_adversarial.h5')
        plt.clf()
        plt.title('loss')
        plt.plot(self.gloss, label='g')
        plt.plot(self.dloss, label='d')
        np.save(f'{path.RESULT_LOSSPATH}/gloss.npy', arr=self.gloss)
        np.save(f'{path.RESULT_LOSSPATH}/dloss.npy', arr=self.dloss)
        plt.legend(loc='best')
        plt.savefig(f"{path.RESULT_LOSSPATH}/loss.png")

    def predict(self, num_images=3):
        test_sketchdata, test_realdata = DataLoader().load_testing_data(metrics='[0,1]')
        idx = np.random.randint(0, test_sketchdata.shape[0], num_images)
        model = load_model(f'./result/h5/{self.modelname}_generator.h5')
        generated_images = model(test_sketchdata[idx])
        plt.figure(figsize=(8, 8))
        plt.title(f'predict')
        plt.suptitle(f'The Images Of {self.modelname} Generate', fontsize=17)


        for j in range(num_images):
            plt.subplot(num_images, num_images, num_images*j+1)
            plt.title('generated', fontsize=12)
            plt.imshow(generated_images[j])
            plt.axis("off")
            plt.subplot(num_images, num_images, num_images * j+2)
            plt.title('sketch', fontsize=12)
            plt.imshow(test_sketchdata[j])
            plt.axis("off")
            plt.subplot(num_images, num_images, num_images * j+3)
            plt.title('true', fontsize=12)
            plt.imshow(test_realdata[j])
            plt.axis("off")
        plt.savefig(f'{path.IMAGES_PREDICTPATH}/predict.png')
        # plt.show()

    def showinput(self, num_images=16, idx=[], data=None):
        #TODO 可以做新增資料夾&儲存輸入data 資料夾：trainpred,sketchinput,realinput (不要用epoch建立資料夾，影片會不好用)
        img = data[idx]
        plt.figure(figsize=(8, 8))
        plt.title(f'predict')
        for j in range(num_images):
            plt.subplot(4, 4, j + 1)
            plt.imshow(img[j, :, :, :])
            plt.axis("off")
        plt.show()


# if __name__ == '__main__':
    # gan = Pix2pix(model_name='pix2pix')
    # s = time.time()
    # gan.train() # 10000 epochs->used 25826.565 seconds7.17hour
    # e = time.time()
    # print(f'used {e-s:.3f} seconds')
    #
    # gan.predict()

