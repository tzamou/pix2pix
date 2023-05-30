# import utils
from pix2pix_model import Pix2pix
from utils import config
import time
import utils
import matplotlib.pyplot as plt

config = config.TrainConfig(epochs=50000,
                            batchsize=16,
                            gen_lr=0.0002,
                            dis_lr=0.0002)



gan = Pix2pix(model_name='pix2pix',config=config)
s = time.time()
gan.train() # 10000 epochs->used 25826.565 seconds7.17hour
e = time.time()
print(f'used {e-s:.3f} seconds')
gan.predict()

# g_gan_loss = utils.average_filter(gan.gloss[:,1],window_size=100)
# d_gan_loss = utils.average_filter(gan.dloss,window_size=100)
# plt.title('PIX2PIX Loss')
# plt.plot(g_gan_loss,label='g')
# plt.plot(d_gan_loss,label='d')
# plt.grid(True)
# plt.show()
#
# g_mae_loss = utils.average_filter(gan.gloss[:,2],window_size=100)
# plt.clf()
# plt.title('Generator MAE Loss')
# plt.plot(g_mae_loss,label='d')
# plt.grid(True)
# plt.show()
