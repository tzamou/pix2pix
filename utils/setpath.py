import os
import datetime

DOCSPATH = './docs'
LOGPATH = './log'
RESULTPATH = './result'
RESULT_TRAINPATH = './result/train'
RESULT_IMGPATH = './result/train/img'
RESULT_H5PATH = './result/h5'

#if not os.path.exists('./test/test2/test3'):
#    os.makedirs('./test/test2/test3')
def check_folder_path(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def path_process():
    check_folder_path(DOCSPATH)
    check_folder_path(LOGPATH)
    check_folder_path(RESULTPATH)
    check_folder_path(RESULT_TRAINPATH)
    check_folder_path(RESULT_IMGPATH)
    check_folder_path(RESULT_H5PATH)

def train_folder_process():
    global DATEPATH,LOG_CONFIGPATH,LOG_PREDICTPATH,LOG_LOSSPATH,LOG_MODELPATH
    nowtime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    DATEPATH = f'./log/{nowtime}'
    LOG_CONFIGPATH = f'./log/{nowtime}/config'
    LOG_PREDICTPATH = f'./log/{nowtime}/pred'
    LOG_LOSSPATH = f'./log/{nowtime}/loss'
    LOG_MODELPATH = f'./log/{nowtime}/model'
    check_folder_path(DATEPATH)
    check_folder_path(LOG_CONFIGPATH)
    check_folder_path(LOG_PREDICTPATH)
    check_folder_path(LOG_LOSSPATH)
    check_folder_path(LOG_MODELPATH)

def get_path():
    print('to-do')
if __name__=='__main__':
    get_path()