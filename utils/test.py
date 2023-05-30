import numpy as np
import matplotlib.pyplot as plt
import utils

gloss = np.load(r'../log/2023-05-28-01-12-52/loss/gloss.npy',allow_pickle=True)
dloss = np.load(r'../log/2023-05-28-01-12-52/loss/dloss.npy',allow_pickle=True)

window_size = 100
g_gan_loss = utils.average_filter(gloss[:,1],window_size=window_size)
d_gan_loss = utils.average_filter(dloss,window_size=window_size)
plt.title('PIX2PIX Loss')
# plt.ylim(0,0.6)
plt.plot(g_gan_loss,label='g')
plt.plot(d_gan_loss,label='d')
plt.grid(True)
plt.show()

g_mae_loss = utils.average_filter(gloss[:,2],window_size=window_size)
# g_mae_loss = utils.smooth(gloss[:,2])
plt.clf()
plt.title('Generator MAE Loss')
plt.plot(g_mae_loss,label='d')
plt.grid(True)
plt.show()