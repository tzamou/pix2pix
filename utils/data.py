import glob
import cv2
import numpy as np

def standardization(func):
    def wrapper(*args,**kwargs):
        if kwargs['metrics'] == '[0,1]':
            x, y = func(*args, **kwargs)
            x /= 255
            y /= 255
            return x, y
        elif kwargs['metrics'] == '[-1,1]':
            x, y = func(*args, **kwargs)
            x = x/127.5 - 1
            y = y/127.5 - 1
            return x, y
        else:
            raise ValueError("The metrics parameter only provides '[0,1]' and '[-1,1]' to be entered")
    return wrapper

class DataLoader:
    def __init__(self):
        """
        路徑記得改!!!
        """
        self.realpath = 'F:/Dataset/FacadesDataset/trainA'
        self.sketchpath = 'F:/Dataset/FacadesDataset/trainB'
        self.testrealpath = 'F:/Dataset/FacadesDataset/testA'
        self.testsketchpath = 'F:/Dataset/FacadesDataset/testB'

    def __loadimg(self,datapath):
        data = cv2.imread(datapath,cv2.IMREAD_COLOR)
        data = cv2.cvtColor(data,cv2.COLOR_BGR2RGB).reshape(1,256,256,3)
        return data

    @standardization
    def load_data(self,metrics='[0,1]'):
        realdatapath = glob.glob(f'{self.realpath}/*.jpg')#load 400 data
        sketchdatapath = glob.glob(f'{self.sketchpath}/*.jpg')
        self.sketchdata = self.__loadimg(sketchdatapath[0])#sketchimg.reshape(1, 256, 256, 3)
        self.realdata = self.__loadimg(realdatapath[0])#realimg.reshape(1, 256, 256, 3)
        for index in range(1,len(realdatapath)):
            self.sketchdata = np.concatenate([self.sketchdata,self.__loadimg(sketchdatapath[index])],axis=0)
            self.realdata = np.concatenate([self.realdata, self.__loadimg(realdatapath[index])], axis=0)
            p = int(30*index/400)
            print(f"\rLoading training data: {'*'*p}{'.'*(29-p)} {100*round(index/399,3):.1f}%  ", end='')

        self.realdata = self.realdata.astype(np.float32)
        self.sketchdata = self.sketchdata.astype(np.float32)
        print('Loading success!\n')
        return self.sketchdata, self.realdata  #shape=(400, 256, 256, 3)

    @standardization
    def load_testing_data(self,metrics='[0,1]'):
        realdatapath = glob.glob(f'{self.testrealpath}/*.jpg')  # load 400 data
        sketchdatapath = glob.glob(f'{self.testsketchpath}/*.jpg')
        self.sketchdata = self.__loadimg(sketchdatapath[0])  # sketchimg.reshape(1, 256, 256, 3)
        self.realdata = self.__loadimg(realdatapath[0])  # realimg.reshape(1, 256, 256, 3)
        for index in range(1, len(realdatapath)):
            self.sketchdata = np.concatenate([self.sketchdata, self.__loadimg(sketchdatapath[index])], axis=0)
            self.realdata = np.concatenate([self.realdata, self.__loadimg(realdatapath[index])], axis=0)
            p = int(30 * index / 106)
            print(f"\rLoading testing data: {'*' * p}{'.' * (29 - p)} {100 * round(index / 105, 3):.1f}%  ", end='')

        self.realdata = self.realdata.astype(np.float64)
        self.sketchdata = self.sketchdata.astype(np.float64)
        print('Loading success!\n')
        return self.sketchdata, self.realdata  # shape=(400, 256, 256, 3)



if __name__=='__main__':
    # video(speed=2)
    d=DataLoader()
    sketchdata, realdata=d.load_data(metrics='[0,1]')
    test_sketchdata, test_realdata = d.load_testing_data(metrics='[0,1]')

