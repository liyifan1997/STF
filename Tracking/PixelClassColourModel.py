import numpy as np
from .Histogram import Histogram
import math
FG_PRIOR_PROBABILITY=0.3
from . import toolfunction



class PixelClassColourModel:
    def __init__(self, lut, h_bins, s_bins, v_bins, ni_scales, imgw, imgh):
        self.mImageBB = np.zeros((imgh, imgw))
        self.mHist = [None] * 3
        self.mUpdateHist = [None] * 3

        for i in range(3):
            self.mHist[i] = Histogram(h_bins, s_bins, v_bins, ni_scales, self.mImageBB)
            self.mUpdateHist[i] = Histogram(h_bins, s_bins, v_bins, ni_scales, self.mImageBB)

        self.mLUT = lut
        self.mfMeanFGVoteErr = -1

    def __del__(self):
        for i in range(2):
            del self.mHist[i]
            del self.mUpdateHist[i]

    def setVoteErrParameters(self, mean_pos, var_pos, mean_neg, var_neg):
        self.mfMeanFGVoteErr = mean_pos
        self.mfVarFGVoteErr = var_pos
        self.mfMeanBGVoteErr = mean_neg
        self.mfVarBGVoteErr = var_neg

    def create(self, img, outer_object, object_region,mask=None, enlarge_factor_for_bg=1.5):  #图像，searchbox,bbox
        margin = outer_object  #margin为1 select box, 记为outer box
        outer = margin
        outer = toolfunction.boxenlarge(outer,enlarge_factor_for_bg,img.shape[1],img.shape[0])
        outer_top=[outer[0], margin[0]-1, outer[2],outer[3]]
        outer_bottom=[margin[1]+1,outer[1],outer[2],outer[3]]
        outer_left=[margin[0],margin[1],outer[2],margin[2]-1]
        outer_right=[margin[0],margin[1],margin[3]+1,outer[3]]
        sigmax=(object_region[3]-object_region[2]+1)/8
        sigmay=(object_region[1]-object_region[0]+1)/8

        #FG model
        self.mHist[0].compute(img, self.mLUT, object_region,mask=mask,set_zero=True,normalise_hist=True,grid=False)
        #BG model
        self.mHist[1].compute(img, self.mLUT, outer_top, mask=None, set_zero=True, normalise_hist=False, grid=False)
        self.mHist[1].compute(img, self.mLUT, outer_bottom, mask=None, set_zero=False, normalise_hist=False, grid=False)
        self.mHist[1].compute(img, self.mLUT, outer_left, mask=None, set_zero=False, normalise_hist=False, grid=False)
        self.mHist[1].compute(img, self.mLUT, outer_right, mask=None, set_zero=False, normalise_hist=True, grid=False)  #self.mHist[1].hsv_counts[s][hsv_bin_index]

    def update(self, img, outer_bb, segmentation, bp_img, update_factor, enlarge_factor_for_bg=1.5):
        self.mUpdateHist[0].compute(img, self.mLUT, outer_bb, mask=bp_img, set_zero=True, normalise_hist=True,
                                    grid=False)
        self.mHist[0].update(self.mUpdateHist[0], update_factor)











    # def update(self,img,outer_bb,segmentation,bp_img,update_factor,enlarge_factor_for_bg=2):
    #     margin=outer_bb.copy()
    #     outer=margin
    #     outer=toolfunction.boxenlarge(outer,enlarge_factor_for_bg,img.shape[1],img.shape[0])
    #     outer_top = [outer[0], margin[0] - 1, outer[2], outer[3]]
    #     outer_bottom = [margin[1] + 1, outer[1], outer[2], outer[3]]
    #     outer_left = [margin[0], margin[1], outer[2], margin[2] - 1]
    #     outer_right = [margin[0], margin[1], margin[3] + 1, outer[3]]
    #     sigmax=(outer_bb[3]-outer_bb[2]+1)/8
    #     sigmay = (outer_bb[1] - outer_bb[0] + 1) / 8
    #     rw2=(outer_bb[3]-outer_bb[2]+1)/2
    #     rh2=(outer_bb[1]-outer_bb[0]+1)/2
    #     outer_bb[0] = max(outer_bb[0], 1)
    #     outer_bb[1] = min(outer_bb[1], img.shape[0] - 2)
    #     outer_bb[2] = max(outer_bb[2], 1)
    #     outer_bb[3] = min(outer_bb[3], img.shape[1] - 2)
    #     sj0=0
    #     for j in range (outer_bb[0],outer_bb[1]+1):
    #         dy=sj0-rh2
    #         si0=0
    #         for i in range (outer_bb[2],outer_bb[3]+1):
    #             dx=si0-rw2
    #             spatial_prior = 0.7 * math.exp(-0.5 * (dx * dx / sigmax / sigmax + dy * dy / sigmay / sigmay))
    #             bp_img[j,i] += spatial_prior
    #             si0 += 1
    #             dx += 1
    #         sj0 += 1
    #         dy += 1
    #
    #     self.mUpdateHist[0].compute(img,self.mLUT,outer_bb,mask=bp_img,set_zero=True, normalise_hist=True, grid=False)
    #     self.mHist[0].update(self.mUpdateHist[0],update_factor)
    #
    #     self.mUpdateHist[1].compute(img,self.mLUT,outer_top,mask=None,set_zero=True, normalise_hist=False, grid=False)
    #     self.mUpdateHist[1].compute(img, self.mLUT, outer_bottom, mask=None, set_zero=False, normalise_hist=False,
    #                                 grid=False)
    #     self.mUpdateHist[1].compute(img, self.mLUT, outer_left, mask=None, set_zero=False, normalise_hist=False,
    #                                 grid=False)
    #     self.mUpdateHist[1].compute(img, self.mLUT, outer_right, mask=None, set_zero=False, normalise_hist=True,
    #                                 grid=False)
    #     self.mHist[1].update(self.mUpdateHist[1],update_factor)







    def evaluateColour(self,img,roi,use_spatial_prior,result):
        width = img.shape[1]
        height = img.shape[0]
        roi[0] = max(roi[0], 1)
        roi[1] = min(roi[1], height - 2)
        roi[2] = max(roi[2], 1)
        roi[3] = min(roi[3], width - 2)
        roi_height=roi[1]-roi[0]+1
        roi_width=roi[3]-roi[2]+1
        xgrid_step = 1 #max(1, roi_width // 90)
        ygrid_step = 1 #max(1, roi_height // 90)
        roiny= roi_height//ygrid_step
        roinx=roi_width//xgrid_step
        sigmax = (roi[3]-roi[2]+1) / 4
        sigmay = (roi[1]-roi[0]+1) / 4
        ptr_y = roi[0]
        for j in range(roiny):  # 遍历roiny的行
            dy = ptr_y - (roi[0]+roi[1])/2
            ptr_x =roi[2]
            for i in range(roinx):  # 每一行遍历每个格子
                dx = ptr_x - (roi[2]+roi[3])/2
                b, g, r = img[ptr_y, ptr_x]
                if use_spatial_prior:
                    spatial_prior = math.exp(-0.5 * (dx * dx / sigmax / sigmax + dy * dy / sigmay / sigmay)) #sigmax=searchbox.width/4
                tmpres = 1.0
                for s in range(self.mHist[0].ni_scales):
                    index=int(self.mLUT[s].bgr_to_colour_bin[((b *256) + g) *256 + r])
                    if use_spatial_prior:
                        colour_prob =spatial_prior*self.mHist[0].hsv_count[s][index]*FG_PRIOR_PROBABILITY+(1-spatial_prior)*self.mHist[1].hsv_count[s][index]*(1-FG_PRIOR_PROBABILITY)
                    else:
                        colour_prob=self.mHist[0].hsv_count[s][index]*FG_PRIOR_PROBABILITY+self.mHist[1].hsv_count[s][index]*(1-FG_PRIOR_PROBABILITY)
                    if colour_prob>0:
                        if use_spatial_prior:
                            tmpres *=spatial_prior*self.mHist[0].hsv_count[s][index]*FG_PRIOR_PROBABILITY/colour_prob
                        else:
                            tmpres *= self.mHist[0].hsv_count[s][index]*FG_PRIOR_PROBABILITY/colour_prob
                    else:
                        tmpres=0
                result[ptr_y, ptr_x]=tmpres
                ptr_x += xgrid_step
                dx += xgrid_step
            ptr_y += ygrid_step
            dy+= ygrid_step


    def evaluateColourWithPrior(self,img,roi,use_sapatial_prior,prior,result):
        height=img.shape[0]
        width=img.shape[1]
        roi[0] = max(roi[0], 1)
        roi[1] = min(roi[1], height - 2)
        roi[2] = max(roi[2], 1)
        roi[3] = min(roi[3], width - 2)
        roi_height = roi[1] - roi[0] + 1
        roi_width = roi[3] - roi[2] + 1
        xgrid_step = 1 #max(1, roi_width // 90)
        ygrid_step = 1 #max(1, roi_height // 90)
        roiny = roi_height // ygrid_step
        roinx = roi_width // xgrid_step
        sigmax = roi_width / 4
        sigmay = roi_height / 4
        ptr_y = roi[0]
        for j in range(roiny):  # 遍历roiny的行
            dy = ptr_y - (roi[0] + roi[1]) / 2
            ptr_x = roi[2]
            for i in range(roinx):  # 每一行遍历每个格子
                dx = ptr_x - (roi[2] + roi[3]) / 2
                if use_sapatial_prior:
                    spatial_prior=np.exp(-0.5 * (dx * dx / sigmax / sigmax + dy * dy / sigmay / sigmay))
                b, g, r = img[ptr_y, ptr_x]
                prior_val=prior[ptr_y, ptr_x]
                trans_fg_to_fg = 0.4
                trans_fg_to_bg = 0.6
                trans_bg_to_fg = 0.4
                trans_bg_to_bg = 0.6
                if prior_val>0.5:
                    trans=0.6
                else:
                    trans=0.4
                tmpres=1
                for s in range (self.mHist[0].ni_scales):
                    index=int(self.mLUT[s].bgr_to_colour_bin[((b *256) + g) *256 + r])
                    if use_sapatial_prior:
                        colour_prob=spatial_prior*self.mHist[0].hsv_count[s][index]*prior_val*trans+(1-spatial_prior)*self.mHist[1].hsv_count[s][index]*(1-prior_val)*(1-trans)
                        if colour_prob>0:
                            tmpres=spatial_prior*self.mHist[0].hsv_count[s][index]*prior_val*trans/colour_prob
                        else:
                            tmpres=0
                    else:
                        ttmp=self.mHist[0].hsv_count[s][index]*prior_val*trans_fg_to_fg+self.mHist[0].hsv_count[s][index]*(1-prior_val)*trans_bg_to_fg
                        ttmp_neg=self.mHist[1].hsv_count[s][index]*prior_val*trans_fg_to_bg+self.mHist[1].hsv_count[s][index]*(1-prior_val)*trans_bg_to_bg
                        if (ttmp+ttmp_neg>0):
                            tmpres=ttmp/(ttmp+ttmp_neg)
                        else:
                            tmpres=0
                result[ptr_y, ptr_x]=tmpres
                ptr_x += xgrid_step
                dx += xgrid_step
            ptr_y += ygrid_step
            dy += ygrid_step














