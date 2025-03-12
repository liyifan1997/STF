import cv2
import numpy as np
from .HSVPixelGradientModel import HSVPixelGradientModel
from .BGR2HSVhistLUT import BGR2HSVhistLUT
from .PixelClassColourModel import PixelClassColourModel
from . import toolfunction
from .MatrixAnalyzer import MatrixAnalyzer
import matplotlib.pyplot as plt
#示例：python pixeltrack.py <options> <video_file>，例如python pixeltrack.py -b 10,20,30,40 -f 0 -k 5 -o -p -s -t 100 -u 0.2 -v 0.3 -w 1.5 -z your_video.mp4

width=150
height=100

class yifanTracking:
    def __init__(self,bayesbins,votbins,image):
        image=cv2.resize(image, (width,height))
        self.width = image.shape[1]
        self.height = image.shape[0]
        self.model=HSVPixelGradientModel.load_model_from_file(votbins, votbins, 8, 1, 60) #voting model
        self.lut = [None] * 2
        for s in range(2):
            self.lut[s] = BGR2HSVhistLUT.load_lut_from_file(bayesbins * (s + 1), bayesbins * (s + 1),
                                                       bayesbins * (s + 1))
        self.pccm = PixelClassColourModel(self.lut, bayesbins, bayesbins, bayesbins, 2, self.width,
                                     self.height)  # segmentation模型只考虑hsv bin，并计算两个尺度
        self.prev_shift_x = 0
        self.prev_shift_y = 0
        self.bayes_prior=[]
        self.search_box=[]
        self.cur_box=[]
        self.mask_prior=[]



    def firstTracking(self,image,mask):
        image = cv2.resize(image, (width,height))
        mask = cv2.resize(mask, (width,height))
        mask = (mask >= 127).astype(np.uint8)
        bayesImg = np.zeros((self.height, self.width))
        houghImg = np.zeros((self.height, self.width))
        initial_box = toolfunction.calculate_coordinates(mask)
        outer_box = initial_box
        outer_box = toolfunction.boxenlarge(outer_box, 1, self.width, self.height)  # outer-box,1.5 select box
        self.cur_box = initial_box
        self.search_box = self.cur_box
        self.search_box = toolfunction.boxenlarge(self.search_box, 2, self.width, self.height)  # search-box,2 select box
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        xgrad_img = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        ygrad_img = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        self.pccm.create(image, outer_box, initial_box, mask=mask)
        self.pccm.evaluateColour(image, outer_box, False, bayesImg)
        bayesImg=cv2.GaussianBlur(bayesImg, (3, 3), 0)
        bayesImg[bayesImg < 0.4] = 0
        bayesmap = bayesImg.copy()
        bayesmap[bayesmap > 0.4] = 1
        analyzer = MatrixAnalyzer(bayesmap)
        bayesmap = analyzer.max_connected_areas()
        bayesImg = bayesImg * bayesmap
        self.bayes_prior=bayesImg
        cm_maxy, cm_maxx = toolfunction.centerOfMass(bayesmap, outer_box)
        self.cur_box = toolfunction.setcenter(self.cur_box, cm_maxy, cm_maxx)
        maxy = (self.cur_box[1] + self.cur_box[0]) / 2
        maxx = (self.cur_box[3] + self.cur_box[2]) / 2
        larger_box = self.cur_box
        larger_box = toolfunction.boxenlarge(larger_box, 1.2, self.width, self.height)
        smaller_box = self.cur_box
        smaller_box = toolfunction.boxenlarge(smaller_box, 1 / 1.2, self.width, self.height)
        nbfg1 = toolfunction.sumGreaterThanThreshold(bayesImg, larger_box, 0.4)
        nbfg2 = toolfunction.sumGreaterThanThreshold(bayesImg, self.cur_box, 0.4)
        nbfg3 = toolfunction.sumGreaterThanThreshold(bayesImg, smaller_box, 0.4)  # 三种框的前景像素个数
        nbbg1 = toolfunction.area(larger_box) - nbfg1
        nbbg2 = toolfunction.area(self.cur_box) - nbfg2
        nbbg3 = toolfunction.area(smaller_box) - nbfg3  # 三个框背景像素个数
        fgbg_r1 = float(nbfg1) / nbbg1
        fgbg_r2 = float(nbfg2) / nbbg2
        fgbg_r3 = float(nbfg3) / nbbg3  # 三个框前景背景像素个数比值
        if fgbg_r1 > fgbg_r2:
            if fgbg_r1 > fgbg_r3:
                self.cur_box = larger_box
            else:
                self.cur_box = smaller_box
        else:
            if fgbg_r2 < fgbg_r3:
                self.cur_box = smaller_box

        self.search_box = self.cur_box
        self.search_box = toolfunction.boxenlarge(self.search_box, 2, self.width, self.height)  #

        self.model.learn(image, xgrad_img, ygrad_img, self.search_box, bayesImg, mask)  # 统计各个像素到中心点的向量
        self.model.backproject(image, xgrad_img, ygrad_img, self.search_box, houghImg, maxx, maxy)  # 根据投票结果，从中心点到目标区域的映射
        houghImg = cv2.GaussianBlur(houghImg, (3, 3), 0)
        houghImg = np.where(houghImg >= 1, 255, houghImg).astype(np.uint8)
        #houghImg=cv2.resize(houghImg, (1280, 960))
        return houghImg

    def tracking(self,image):
        image = cv2.resize(image, (width,height))
        bayesImg = np.zeros((self.height, self.width))
        houghImg = np.zeros((self.height, self.width))
        voting_map=np.zeros((self.height, self.width))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        xgrad_img = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        ygrad_img = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        self.model.vote(image, xgrad_img, ygrad_img, self.search_box, voting_map)
        maxx, maxy = toolfunction.voteMax(voting_map, self.search_box)
        # self.pccm.evaluateColourWithPrior(image, self.search_box, False, self.bayes_prior, bayesImg)
        self.pccm.evaluateColour(image,self.search_box,False,bayesImg)
        bayesImg = cv2.GaussianBlur(bayesImg, (3, 3), 0)
        bayesImg[bayesImg < 0.4] = 0
        bayesmap = bayesImg.copy()
        bayesmap[bayesmap > 0.4] = 1
        analyzer = MatrixAnalyzer(bayesmap)
        bayesmap = analyzer.max_connected_areas()
        bayesImg = bayesImg * bayesmap
        #cur_bayes_change = toolfunction.percentageChanged(bayesImg, self.cur_box, self.prev_shift_x, self.prev_shift_y,self.bayes_prior )
        self.bayes_prior=bayesImg
        uncertainty = 1
        self.model.backproject(image,xgrad_img,ygrad_img,self.search_box,houghImg,maxx,maxy)
        houghImg = cv2.GaussianBlur(houghImg, (3, 3), 0)



        # self.cur_box = toolfunction.setcenter(self.cur_box, uncertainty * maxy,maxx )


        self.search_box = self.cur_box  # 调整cur_box的大小，前景像素所占的比例最大
        self.search_box = toolfunction.boxenlarge(self.search_box, 1.5, self.width, self.height)
        nbfg = toolfunction.sumGreaterThanThreshold(bayesImg, self.search_box, 0.5)
        larger_box = self.cur_box
        larger_box = toolfunction.boxenlarge(larger_box, 1.2, self.width,self.height)
        nbfg2 = toolfunction.sumGreaterThanThreshold(bayesImg, self.cur_box, 0.5)

        if nbfg2<0.7*nbfg and nbfg2>30:
            self.cur_box=larger_box

        self.search_box = self.cur_box  # 调整cur_box的大小，前景像素所占的比例最大
        self.search_box = toolfunction.boxenlarge(self.search_box, 1.5, self.width, self.height)

        self.model.update(image, xgrad_img, ygrad_img, self.search_box, bayesImg, 0.2)


        # plt.imshow(houghImg, cmap=plt.cm.RdBu, vmin=0, vmax=5)
        # plt.colorbar()
        # plt.show()


        houghImg[houghImg<1]=0
        houghImg = np.where(houghImg >= 1, 255, houghImg).astype(np.uint8)
        if np.sum(houghImg)/255 <100:
            houghImg = np.zeros((self.height, self.width))

        #houghImg = cv2.resize(houghImg, (1280, 960))
        return houghImg,bayesImg




    def firstSeg(self,image,mask):
        image = cv2.resize(image, (width,height))
        mask = cv2.resize(mask, (width,height))
        mask = (mask >= 127).astype(np.uint8)
        houghImg = np.zeros((self.height, self.width))
        initial_box = toolfunction.calculate_coordinates(mask)
        self.cur_box = initial_box
        self.search_box = self.cur_box
        self.search_box = toolfunction.boxenlarge(self.search_box, 1.5, self.width,
                                                  self.height)  # search-box,2 select box
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        xgrad_img = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        ygrad_img = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)


        cm_maxy, cm_maxx = toolfunction.centerOfMass(mask, self.cur_box)
        self.cur_box = toolfunction.setcenter(self.cur_box, cm_maxy, cm_maxx)
        maxy = (self.cur_box[1] + self.cur_box[0]) / 2
        maxx = (self.cur_box[3] + self.cur_box[2]) / 2
        self.search_box = self.cur_box
        self.search_box = toolfunction.boxenlarge(self.search_box, 1.5, self.width, self.height)  #
        self.model.learn(image, xgrad_img, ygrad_img, self.search_box, mask, mask)  # 统计各个像素到中心点的向量
        self.model.backproject(image, xgrad_img, ygrad_img, self.search_box, houghImg, maxx,maxy)
        houghImg = cv2.GaussianBlur(houghImg, (3, 3), 0)
        self.mask_prior = mask
        houghImg = np.where(houghImg >= 1, 255, houghImg).astype(np.uint8)
        houghImg = cv2.resize(houghImg, (1280, 960))
        return houghImg

    def seging(self, image,mask):
        image = cv2.resize(image, (width,height))
        mask = cv2.resize(mask, (width,height))
        mask = (mask >= 127).astype(np.uint8)
        houghImg = np.zeros((self.height, self.width))
        voting_map = np.zeros((self.height, self.width))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        xgrad_img = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        ygrad_img = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        self.model.vote(image, xgrad_img, ygrad_img, self.search_box, voting_map)
        maxx, maxy = toolfunction.voteMax(voting_map, self.search_box)
        cur_seg_change = toolfunction.percentageChanged(mask, self.cur_box, self.prev_shift_x, self.prev_shift_y,
                                                          self.mask_prior)
        self.mask_prior = mask
        uncertainty = max(0.2, min(0.8, cur_seg_change))
        self.model.backproject(image, xgrad_img, ygrad_img, self.search_box, houghImg, maxx, maxy)
        houghImg = cv2.GaussianBlur(houghImg, (3, 3), 0)

        reduced_sw = self.search_box
        reduced_sw = toolfunction.boxenlarge(reduced_sw, 0.7, self.width, self.height)
        cm_maxy, cm_maxx = toolfunction.centerOfMass(mask, reduced_sw)
        prev_box = self.cur_box
        self.cur_box = toolfunction.setcenter(self.cur_box, uncertainty * maxy + (1 - uncertainty) * cm_maxy,
                                              uncertainty * maxx + (1 - uncertainty) * cm_maxx)
        self.prev_shift_y = (self.cur_box[1] + self.cur_box[0]) / 2 - (prev_box[1] + prev_box[0]) / 2
        self.prev_shift_x = (self.cur_box[3] + self.cur_box[2]) / 2 - (prev_box[3] + prev_box[2]) / 2  # 更新目标运动方向

        self.search_box = self.cur_box  # 调整cur_box的大小，前景像素所占的比例最大
        self.search_box = toolfunction.boxenlarge(self.search_box, 1.5, self.width, self.height)
        nbfg = toolfunction.sumGreaterThanThreshold(houghImg, self.search_box, 0.5)
        larger_box = self.cur_box
        larger_box = toolfunction.boxenlarge(larger_box, 1.2, self.width, self.height)
        nbfg2 = toolfunction.sumGreaterThanThreshold(houghImg, self.cur_box, 0.5)

        if nbfg2 < 0.7 * nbfg:
            self.cur_box = larger_box

        self.search_box = self.cur_box  # 调整cur_box的大小，前景像素所占的比例最大
        self.search_box = toolfunction.boxenlarge(self.search_box, 1.5, self.width, self.height)
        self.model.update(image, xgrad_img, ygrad_img, self.search_box, mask, 0.4)
        houghImg = np.where(houghImg >= 1, 255, houghImg).astype(np.uint8)
        houghImg = cv2.resize(houghImg, (1280, 960))
        return houghImg

    def segimproving(self,image,mask,value):
        image = cv2.resize(image, (width,height))
        mask = cv2.resize(mask, (width,height))
        mask = (mask >= 127).astype(np.uint8)
        bayesImg = np.zeros((self.height, self.width))
        initial_box = toolfunction.calculate_coordinates(mask)
        outer_box = initial_box
        outer_box = toolfunction.boxenlarge(outer_box, 1, self.width, self.height)  # outer-box,1.5 select box
        self.cur_box = initial_box
        self.search_box = self.cur_box
        self.search_box = toolfunction.boxenlarge(self.search_box, 1.5, self.width,
                                                  self.height)  # search-box,2 select box
        self.pccm.create(image, outer_box, initial_box, mask=mask)
        self.pccm.evaluateColour(image, outer_box, False, bayesImg)
        bayesImg[bayesImg < value] = 0
        bayesmap = bayesImg.copy()
        bayesmap[bayesmap > value] = 1
        analyzer = MatrixAnalyzer(bayesmap)
        bayesmap = analyzer.max_connected_areas()
        bayesImg = bayesImg * bayesmap
        bayesImg[bayesImg>=value]=255
        bayesImg = cv2.GaussianBlur(bayesImg, (3, 3), 0)
        bayesImg = cv2.resize(bayesImg, (1280, 960)).astype(np.uint8)

        return bayesImg


    def trackInitialization(self,image,mask):
        image = cv2.resize(image, (width,height))
        mask = cv2.resize(mask, (width,height))
        mask = (mask > 0).astype(np.uint8)
        initial_box = toolfunction.calculate_coordinates(mask)
        #if initial_box==[]:

        outer_box = initial_box
        outer_box = toolfunction.boxenlarge(outer_box, 1, self.width, self.height)  # outer-box,1.5 select box
        self.cur_box = initial_box
        self.search_box = self.cur_box
        self.search_box = toolfunction.boxenlarge(self.search_box, 1.5, self.width,
                                                  self.height)  # search-box,2 select box
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        xgrad_img = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        ygrad_img = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        maxy = (self.cur_box[1] + self.cur_box[0]) / 2
        maxx = (self.cur_box[3] + self.cur_box[2]) / 2
        self.model.reset()
        self.model.learn(image, xgrad_img, ygrad_img, initial_box, mask, mask)


























