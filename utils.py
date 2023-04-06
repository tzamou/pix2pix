import os
import glob
import cv2
import numpy as np

IMAGESPATH = './images'
IMAGES_TRAINPATH = './images/train'
IMAGES_PREDICTPATH = './images/predict'
RESULTPATH = './result'
RESULT_LOSSPATH = './result/loss'
RESULT_MODELPATH = './result/model'
RESULT_H5PATH = './result/h5'


def path_process():
    if not os.path.isdir(IMAGESPATH): os.mkdir(IMAGESPATH)
    if not os.path.isdir(IMAGES_TRAINPATH): os.mkdir(IMAGES_TRAINPATH)
    if not os.path.isdir(IMAGES_PREDICTPATH): os.mkdir(IMAGES_PREDICTPATH)
    if not os.path.isdir(RESULTPATH): os.mkdir(RESULTPATH)
    if not os.path.isdir(RESULT_LOSSPATH): os.mkdir(RESULT_LOSSPATH)
    if not os.path.isdir(RESULT_MODELPATH): os.mkdir(RESULT_MODELPATH)
    if not os.path.isdir(RESULT_H5PATH): os.mkdir(RESULT_H5PATH)


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

def video(speed):
    """
    :param speed: 速度值代表影片放慢幾倍
    :return:
    """
    path = "./images/train/*.png"
    result_name = 'output.mp4'
    #
    frame_list = glob.glob(path)
    print(frame_list)
    print("frame count: ", len(frame_list))
    fps = 20
    shape = cv2.imread(frame_list[0]).shape  # delete dimension 3
    size = (shape[1], shape[0])
    print("frame size: ", size)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(result_name, fourcc, fps, size)

    for idx, path in enumerate(frame_list):
        frame = cv2.imread(path)
        current_frame = idx + 1
        total_frame_count = len(frame_list)
        percentage = int(current_frame * 30 / (total_frame_count + 1))
        print("\rProcess: [{}{}] {:06d} / {:06d}".format("#" * percentage, "." * (30 - 1 - percentage), current_frame,
                                                         total_frame_count), end='')
        for i in range(speed):
            out.write(frame)

    out.release()
    print("Finish making video !!!")

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

        self.realdata = self.realdata.astype(np.float64)
        self.sketchdata = self.sketchdata.astype(np.float64)
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

