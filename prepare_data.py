import os
import cv2
import numpy as np
import pickle
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,default='data\\test')
parser.add_argument('--data', type=str,default='test')
parser.add_argument('--size', type=int, default=120)
parser.add_argument('--count', type=int, default=500)
parser.add_argument('--channel', type=int, default=1)
args = parser.parse_args()


def prepare_data(path, size , data, count,channel):
    final_arr=[]
    

    if len([os.path.join(path,i) for i in os.listdir(path)]) > count:

        img_list=[os.path.join(path,i) for i in os.listdir(path)][:count]

        for im_path in tqdm(img_list):
            try:
                img=cv2.imread(im_path,channel)
                img=cv2.resize(img,(size,size))
                
                final_arr.append(img)

            except:
                pass
        print(f'total {len(final_arr)} images read')

        X=np.stack(final_arr)

        with open(f'{data}.pickle','wb') as f:
            pickle.dump(X, f)
        print(f'{data} data created successfully')

    else:
        print('please decrease the count value')



if __name__=='__main__':

    prepare_data(args.path, args.size, args.data, args.count,args.channel)
    

