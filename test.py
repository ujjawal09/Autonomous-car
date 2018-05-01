import cv2,os
import numpy as np
import matplotlib.image as mpimg
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL=160,320,3
INPUT_SHAPE=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)
def load_img():
    return mpimg.imread("center_2018_04_10_12_21_49_097.jpg")
def crop(image):
    return image[60:-25,:,:]
def resize(image):
    return cv2.resize(image,(IMAGE_WIDTH,IMAGE_HEIGHT),cv2.INTER_AREA)
def rbg2yuv(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
def random_brightness(image):
    hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    ratio=1.0 +0.4 *( 0.2)
    hsv[:,:,2]=hsv[:,:,2]*ratio
    return cv2.cvtColor(hsv,cv2.COLOR_HLS2RGB)
def random_shadow(image):
    x1, y1= IMAGE_WIDTH * np.random.rand(),0
    x2, y2= IMAGE_WIDTH * np.random.rand(),IMAGE_HEIGHT
    xm, ym= np.mgrid[0:IMAGE_HEIGHT,0:IMAGE_WIDTH]
    mask=np.zeros_like(image[:,:,1])
    mask[(ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) >0]=1
    cond= mask==np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)
    hls=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    hls[:,:,1][cond]=hls[:,:,1][cond]*s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def main():
    image=load_img()
    image2=crop(image)
    """print(image)
    print(image2)"""
    image3=resize(image2)
    image4=rbg2yuv(image3)
    image5=random_brightness(image4)
    image6=random_shadow(image)
    
    #cv2.imshow("HLS", image)
    #cv2.imshow("HLS", image2)
    cv2.imshow('image',image)
   
    cv2.imshow('image2',image2)
    cv2.imshow('image3',image3)
    cv2.imshow('image4',image4)
    cv2.imshow('image5',image5)
    cv2.imshow("image6",image6)
    cv2.waitkey(0)
    cv2.destroyAllWindows()


    
main()
 
