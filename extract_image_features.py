from multiprocessing import Pool, cpu_count
from PIL import Image, ImageStat
from skimage import io
import pandas as pd
import numpy as np
import glob, cv2
import random
from scipy import stats

random.seed(1)
np.random.seed(1)
np.seterr(divide='ignore', invalid='ignore')

def get_features(path):
    try:
        st = []
        #pillow jpg
        img = Image.open(path)
        im_stats_ = ImageStat.Stat(img)
        st += im_stats_.sum
        st += im_stats_.mean
        st += im_stats_.rms
        st += im_stats_.var
        st += im_stats_.stddev
        img = np.array(img)[:,:,:3]
        st += [stats.kurtosis(img[:,:,0].ravel())]
        st += [stats.kurtosis(img[:,:,1].ravel())]
        st += [stats.kurtosis(img[:,:,2].ravel())]
        st += [stats.skew(img[:,:,0].ravel())]
        st += [stats.skew(img[:,:,1].ravel())]
        st += [stats.skew(img[:,:,2].ravel())]
        #cv2 jpg
        img = cv2.imread(path)
        bw = cv2.imread(path,0)
        st += list(cv2.calcHist([bw],[0],None,[256],[0,256]).flatten()) #bw
        st += list(cv2.calcHist([img],[0],None,[256],[0,256]).flatten()) #r
        st += list(cv2.calcHist([img],[1],None,[256],[0,256]).flatten()) #g
        st += list(cv2.calcHist([img],[2],None,[256],[0,256]).flatten()) #b
        try:
            #skimage tif
            p1 = path.replace('jpg','tif')
            p1 = p1.replace('train-tif','train-tif-v2') #Why make path changes so complex that they nullify old scripts
            p1 = p1.replace('test-tif-v2','test-tif-v3') #Why make path changes so complex that they nullify old scripts
            imgr = io.imread(p1)
            tf = imgr[:, :, 3]
            st += list(cv2.calcHist([tf],[0],None,[256],[0,65536]).flatten()) #near ifrared
            ndvi = ((imgr[:, :, 3] - imgr[:, :, 0]) / (imgr[:, :, 3] + imgr[:, :, 0])) #water ~ -1.0, barren area ~ 0.0, shrub/grass ~ 0.2-0.4, forest ~ 1.0
            st += list(np.histogram(ndvi,bins=20, range=(-1,1))[0])
            ndvi = ((imgr[:, :, 3] - imgr[:, :, 1]) / (imgr[:, :, 3] + imgr[:, :, 1]))
            st += list(np.histogram(ndvi,bins=20, range=(-1,1))[0])
            ndvi = ((imgr[:, :, 3] - imgr[:, :, 2]) / (imgr[:, :, 3] + imgr[:, :, 2]))
            st += list(np.histogram(ndvi,bins=20, range=(-1,1))[0])
        except:
            st += [-1 for i in range(256)]
            st += [-2 for i in range(60)]
            p1 = path.replace('jpg','tif')
            p1 = p1.replace('train-tif','train-tif-v2') #Why make path changes so complex that they nullify old scripts
            p1 = p1.replace('test-tif-v2','test-tif-v3') #Why make path changes so complex that they nullify old scripts
            print('err', p1)
        m, s = cv2.meanStdDev(img) #mean and standard deviation
        st += list(m)
        st += list(s)
        st += [cv2.Laplacian(bw, cv2.CV_64F).var()]
        st += [cv2.Laplacian(img, cv2.CV_64F).var()]
        st += [cv2.Sobel(bw,cv2.CV_64F,1,0,ksize=5).var()]
        st += [cv2.Sobel(bw,cv2.CV_64F,0,1,ksize=5).var()]
        st += [cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5).var()]
        st += [cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5).var()]
        st += [(bw<30).sum()]
        st += [(bw>225).sum()]
    except(IOError):
        print(path)
    return [path, st]

def normalize_img(paths):
    imf_d = {}
    p = Pool(cpu_count())
    ret = p.map(get_features, paths)
    for i in range(len(ret)):
        imf_d[ret[i][0]] = ret[i][1]
    ret = []
    fdata = [imf_d[f] for f in paths]
    return fdata

in_path = '../data/planet_amazon/'
train = pd.read_csv(in_path + 'train_v2.csv')[:100]
train['path'] = train['image_name'].map(lambda x: in_path + 'train-jpg/' + x + '.jpg')
train_id = np.array([p.split('/')[3].replace('.jpg','') for p in train['path']])
y = train['tags'].str.get_dummies(sep=' ')
xtrain = normalize_img(train['path']); print('train...')
print(np.array(xtrain).shape)
print(train_id.shape)
pd.DataFrame(np.hstack([train_id.reshape((-1,1)), xtrain])).to_csv("../data/planet_amazon/train_features.csv", index=False)

test_jpg = glob.glob(in_path + 'test-jpg-v2/*')[:100]
test = pd.DataFrame([[p.split('/')[3].replace('.jpg',''),p] for p in test_jpg])
test.columns = ['image_name','path']
xtest = normalize_img(test['path']); print('test...')
pd.DataFrame(np.hstack([test['image_name'], xtest])).to_csv("../data/planet_amazon/test_features.csv", index=False)
