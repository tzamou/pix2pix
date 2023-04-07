# import utils
from pix2pix_model import Pix2pix
from utils import config
import time

config = config.TrainConfig(epochs=10000,
                            batchsize=16,
                            gen_lr=0.0002,
                            dis_lr=0.0002)



gan = Pix2pix(model_name='pix2pix',config=config)
s = time.time()
gan.train() # 10000 epochs->used 25826.565 seconds7.17hour
e = time.time()
print(f'used {e-s:.3f} seconds')

gan.predict()
