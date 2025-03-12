import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from .MatrixAnalyzer import MatrixAnalyzer

def boxenlarge(roi,enlarge_size,width,height):
    centery=(roi[0]+roi[1])/2
    centerx=(roi[3]+roi[2])/2
    height_half=(roi[1]-roi[0])/2
    width_half=(roi[3]-roi[2])/2
    result=[math.floor(centery-height_half*enlarge_size), math.ceil(centery+height_half*enlarge_size),math.floor(centerx-width_half*enlarge_size), math.ceil(centerx+width_half*enlarge_size)]
    result[0] = max(result[0], 0)
    result[1] = min(result[1], height - 1)
    result[2] = max(result[2], 0)
    result[3] = min(result[3], width - 1)
    return result

def centerOfMass(segmentation,roi):
    roi[0]=max(roi[0], 0)
    roi[1] = min(roi[1], segmentation.shape[0] -1)
    roi[2] = max(roi[2], 0)
    roi[3] = min(roi[3], segmentation.shape[1] -1)
    sum=0
    fmaxx=0
    fmaxy=0

    for j in range(roi[0], roi[1] + 1):  # j遍历bb[0]~bb[1],y
        for i in range(roi[2], roi[3] + 1):  # i遍历bb[2]~bb[3],x
            val=segmentation[j,i]
            fmaxy +=val*j
            fmaxx += val*i
            sum+=val
    if sum==0:
        maxy=(roi[0]+roi[1])/2
        maxx=(roi[3]+roi[2])/2
    else:
        maxy=round(fmaxy / sum)
        maxx=round(fmaxx / sum)
    return maxy,maxx

def setcenter(box,y,x):
    roi=box
    height_half = (roi[1] - roi[0]) / 2
    width_half = (roi[3] - roi[2]) / 2
    result=[round(y-height_half),round(y+height_half),round(x-width_half),round(x+width_half)]
    return result

def sumGreaterThanThreshold(segmentation,roi,threshold):
    roi[0] = max(roi[0], 0)
    roi[1] = min(roi[1], segmentation.shape[0] - 1)
    roi[2] = max(roi[2], 0)
    roi[3] = min(roi[3], segmentation.shape[1] - 1)
    result=0
    for j in range(roi[0], roi[1] + 1):  # j遍历bb[0]~bb[1],y
        for i in range(roi[2], roi[3] + 1):  # i遍历bb[2]~bb[3],x
            if segmentation[j,i]>threshold:
                result +=1

    return result

def area(roi):
    height=roi[1]-roi[0]+1
    width=roi[3]-roi[2]+1
    result=height*width
    return result


def voteMax(votemap,roi):
    roi[0] = max(roi[0], 1)
    roi[1] = min(roi[1], votemap.shape[0] - 2)
    roi[2] = max(roi[2], 1)
    roi[3] = min(roi[3], votemap.shape[1] - 2)
    maxvote=0
    maxx=(roi[2]+roi[3])/2
    maxy=(roi[0]+roi[1])/2
    for j in range (roi[0],roi[1]):
        for i in range (roi[2],roi[3]):
            sum=votemap[j,i]+votemap[j-1,i]+votemap[j+1,i]+votemap[j,i-1]+votemap[j-1,i-1]+votemap[j+1,i-1]+votemap[j,i+1]+votemap[j-1,i+1]+votemap[j+1,i+1]
            if sum>maxvote:
                maxvote=sum
                maxx=i
                maxy=j
    return maxx,maxy

def percentageChanged(seg,roi,offx,offy,img): #segmentation,cur_box,prev_shift_x,prev_shift_y,seg_prior
    roi[0] = max(roi[0], 1)
    roi[1] = min(roi[1], seg.shape[0] - 1)
    roi[2] = max(roi[2], 1)
    roi[3] = min(roi[3], seg.shape[1] - 1)

    offx=int(offx)
    offy=int(offy)

    roi[0] = max(roi[0], 1-offy)
    roi[1] = min(roi[1], seg.shape[0] - 1-offy)
    roi[2] = max(roi[2], 1-offx)
    roi[3] = min(roi[3], seg.shape[1] - 1-offx)  #偏移前后roi都要在img范围内
    changes=0
    for j in range (roi[0],roi[1]):
        for i in range (roi[2],roi[3]):
            segdata=seg[j,i]
            imgdata=img[j+offy,i+offx]
            if (abs(segdata-imgdata)>0.5):
                changes +=1
    res=changes/((roi[1]-roi[0]+1)*(roi[3]-roi[2]+1))
    return res

def convert_bgr_to_rgb(image):
    return image[:, :, ::-1]





def draw_histogram(data,x_labels=None):
    if x_labels is None:
        x_labels = range(1, len(data) + 1)

    plt.bar(x_labels, data, color='blue', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Bar Chart')
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def calculate_coordinates(mask):
    max_row = None
    min_row = None
    max_col = None
    min_col = None

    for i in range(len(mask)):
        for j in range(len(mask[0])):
            if mask[i][j] == 1:
                if max_row is None or i > max_row:
                    max_row = i
                if min_row is None or i < min_row:
                    min_row = i
                if max_col is None or j > max_col:
                    max_col = j
                if min_col is None or j < min_col:
                    min_col = j

    return [min_row, max_row, min_col, max_col]

def mask_analysis(mask):
    maskmap = mask.copy()
    maskmap[maskmap < 127] = 0
    maskmap[maskmap >= 127] = 1
    analyzer = MatrixAnalyzer(maskmap)
    maskmap = analyzer.max_connected_areas()
    mask = mask * maskmap
    return mask

def ioucal(seg,mask):
    seg=cv2.resize(seg,(180,120)).astype(np.uint8)
    mask=cv2.resize(mask,(180,120)).astype(np.uint8)
    I=0
    U=0
    seg[seg < 127] = 0
    seg[seg>=127]=1
    mask[mask<127]=0
    mask[mask>1]=1

    I=np.sum(seg & mask)
    U=np.sum(seg | mask )

    iou=I/U
    return iou


def masksearch(mask,searchbox):
    box=np.zeros((120,180))
    box[searchbox[0]:searchbox[1], searchbox[2]:searchbox[3]]=1
    box=cv2.resize(box, (1280, 960)).astype(np.uint8)
    mask=box*mask
    return mask
