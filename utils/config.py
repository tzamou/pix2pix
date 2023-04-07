import json

class TrainConfig:
    def __init__(self,
                 epochs:int = 10000,
                 batchsize:int = 32,
                 gen_lr:float = 0.0002,
                 dis_lr:float = 0.0002,
                 ):
        self.batchsize = batchsize
        self.epochs = epochs
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr

    def save_to_json(self,path='./config.json'):
        keys = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        values = [self.batchsize,self.epochs]
        print(keys)
        data = dict(zip(keys, values))
        if path=='./config.json':
            print(f'\033[4;31m[WARNNING] json file is not in real path.\033[0m')
        with open(path,'w') as f:
            json.dump(data,f)

        '''
        當然可以！這是一個列表推導式，它的作用是過濾掉不需要的屬性名稱。讓我們來逐個解釋一下：
        dir(my_object)：獲取物件的所有屬性名稱。
        for attr in dir(my_object)：遍歷物件的所有屬性名稱。
        if not callable(getattr(my_object, attr))：判斷屬性是否可調用，如果不可調用，則保留該屬性。
        and not attr.startswith("")：判斷屬性名稱是否以“”開頭，如果是，則過濾掉該屬性。
        最終，我們得到了一個只包含指定屬性名稱的列表。希望這可以幫助您！
        
        當然可以！在Python中，callable()函數用於檢查物件是否可調用。如果物件可調用，則返回True；否則返回False。
        在這裡，我們使用了getattr()函數來獲取物件的屬性值，然後使用callable()函數來檢查該屬性是否可調用。如果該屬性不可調用，則保留該屬性。希望這可以幫助您！
        
        '''


if __name__=='__main__':
    t = TrainConfig()
    t.save_to_json()