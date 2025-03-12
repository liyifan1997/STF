import numpy as np
import cv2


class Histogram:
    def __init__(self, h_bins, s_bins, v_bins, ni_scales, img_bb, v_bin_separate=True):
        self.ni_scales = ni_scales
        self.hsv_count = [np.zeros((h_bins * (s + 1) * s_bins * (s + 1) * v_bins * (s + 1),)) for s in range(self.ni_scales)]
        self.m_tmpres = [np.zeros((h_bins * (s + 1) * s_bins * (s + 1) * v_bins * (s + 1),)) for s in range(self.ni_scales)]
        self.mi_nhs_bins = [h_bins * (s + 1) * s_bins * (s + 1) for s in range(self.ni_scales)]
        self.mi_nv_bins = [v_bins * (s + 1) for s in range(self.ni_scales)]
        self.mi_ntotal_bins = np.zeros(self.ni_scales,dtype=int)

        for s in range(self.ni_scales):
            if v_bin_separate:
                self.mi_ntotal_bins[s] = h_bins * (s + 1) * s_bins * (s + 1) * v_bins * (s + 1)
            else:
                self.mi_ntotal_bins[s] = h_bins * (s + 1) * s_bins * (s + 1) * v_bins * (s + 1)
        self.m_image_bb = img_bb

    def __del__(self):
        pass

    def compute(self, img, luts, roi, mask=None, set_zero=True, normalise_hist=True, grid=True, kernel_sigma_x=None, kernel_sigma_y=None):
        if set_zero:
            self.set_zero()

        roi[0]=max(roi[0],0)  #roi矩形区域的firstline，roi范围内img[roi[0]:roi[1],roi[2]:roi[3]]
        roi[1]=min(roi[1],img.shape[0]-1) #roi[1],lastline
        roi[2] = max(roi[2], 0) #firstcol
        roi[3] = min(roi[3], img.shape[1]-1) #保证roi在图像之内
        roi_width=roi[3]-roi[2]+1
        roi_height=roi[1]-roi[0]+1

        n_fg_pixels = roi_width*roi_height
        ptr_yinitial = roi[0]
        ptr_xinitial = roi[2]

        if kernel_sigma_x is not None and kernel_sigma_y is not None:  #有kernel_sigma
            rw2=(roi[2]+roi[3])/2
            rh2=(roi[0]+roi[1])/2
            if grid and n_fg_pixels > 600:
                xgrid_step = max(1, roi_width // 60)  # 一格多少个像素
                ygrid_step = max(1, roi_height // 60)
                roinx = roi_width // xgrid_step  # 一格多少行多少列像素
                roiny = roi_height // ygrid_step

                ptr_y = ptr_yinitial
                for j in range(roiny):  # 遍历roiny的行
                    dy=ptr_y-rh2
                    ptr_x = ptr_xinitial
                    for i in range(roinx):  # 每一行遍历每个格子
                        dx=ptr_x-rw2
                        b, g, r = img[ptr_y, ptr_x]
                        spatial_prior = np.exp(-0.5 * (dx * dx / kernel_sigma_x / kernel_sigma_x + dy * dy / kernel_sigma_y / kernel_sigma_y))
                        for s in range(self.ni_scales):
                            bin_index = int(luts[s].bgr_to_colour_bin[
                                ((b *256 ) + g) *256 + r])  # 通过luts查找s尺度下的bgr_to_colour_bin中该rgb对应的索引
                            self.hsv_count[s][bin_index] += spatial_prior  # s尺度下的bin_index,+1
                        ptr_x += xgrid_step
                        dx += xgrid_step
                    ptr_y += ygrid_step
                    dy+= ygrid_step
                n_fg_pixels = roinx * roiny

            else:  # 不需要网格化
                ptr_y = ptr_yinitial
                for j in range(int(roi_height)):
                    dy = ptr_y - rh2
                    ptr_x = ptr_xinitial
                    for i in range(int(roi_width)):
                        dx = ptr_x - rw2
                        b, g, r = img[ptr_y, ptr_x]
                        spatial_prior = np.exp(-0.5 * (
                                    dx * dx / kernel_sigma_x / kernel_sigma_x + dy * dy / kernel_sigma_y / kernel_sigma_y))  #kernel_sigma_x=bbox_width/8，
                        for s in range(self.ni_scales):
                            bin_index = int(luts[s].bgr_to_colour_bin[((b *256) + g) *256 + r])
                            self.hsv_count[s][bin_index] += spatial_prior  # 统计不同尺度下各个hsv_bin的数量
                        ptr_x += 1
                        dx += 1
                    ptr_y += 1
                    dy+= 1
                    # n_fg_pixels = roi_width*roi_height不变

        elif mask is None:   #无kernel_sigma，无mask
            if grid and n_fg_pixels > 600:
                xgrid_step = max(1, roi_width // 60)  #一格多少个像素
                ygrid_step = max(1, roi_height // 60)
                roinx = roi_width // xgrid_step  #一格多少行多少列像素
                roiny = roi_height // ygrid_step

                ptr_y=ptr_yinitial
                for j in range(roiny):  #遍历roiny的行
                    ptr_x=ptr_xinitial
                    for i in range(roinx): #每一行遍历每个格子
                        b, g, r = img[ptr_y, ptr_x]
                        for s in range(self.ni_scales):
                            bin_index = int(luts[s].bgr_to_colour_bin[((b *256) + g) *256 + r]) #通过luts查找s尺度下的bgr_to_colour_bin中该rgb对应的hsv索引
                            self.hsv_count[s][bin_index] += 1   #s尺度下的bin_index,+1
                        ptr_x += xgrid_step
                    ptr_y += ygrid_step
                n_fg_pixels = roinx * roiny


            else:   #没有mask并且不需要网格化降采样
                ptr_y=ptr_yinitial
                for j in range(roi_height):
                    ptr_x = ptr_xinitial
                    for i in range(roi_width):
                        b, g, r = img[ptr_y, ptr_x]
                        for s in range(self.ni_scales):
                            bin_index = int(luts[s].bgr_to_colour_bin[((b *256) + g) *256 + r])
                            self.hsv_count[s][bin_index] += 1 #统计不同尺度下各个hsv_bin的数量
                        ptr_x += 1
                    ptr_y += 1
                    # n_fg_pixels = roi_width*roi_height不变


        else:  #无kernel_sigma，有mask

        #mask是和img同样大小的矩阵，mask>0.5的点计入hsv_count统计
            ptr_y = ptr_yinitial
            for j in range(roi_height):
                ptr_x = ptr_xinitial
                for i in range(roi_width):
                    b, g, r = img[ptr_y, ptr_x]
                    a=mask[51][76]
                    if mask[ptr_y][ptr_x]: #只有mask>0.5才会计入统计
                        for s in range(self.ni_scales):
                            bin_index = int(luts[s].bgr_to_colour_bin[((b *256) + g) *256 + r])
                            self.hsv_count[s][bin_index] += 1
                    n_fg_pixels += 1  #存疑
                    ptr_x += 1
                ptr_y += 1




        if normalise_hist:
            self.normalise()

        return n_fg_pixels


    def set_zero(self):
        for s in range(self.ni_scales):
            self.hsv_count[s] = np.zeros((self.mi_ntotal_bins[s],))

    def normalise(self):
        for s in range(self.ni_scales):
            total_sum = np.sum(self.hsv_count[s])
            if total_sum > 0:
                self.hsv_count[s] /= total_sum

    def update(self, h, factor, norm=True):
        for s in range(self.ni_scales):
            self.hsv_count[s] += factor * h.hsv_count[s]

        if norm:
            self.normalise()

    def distance(self, h):
        return self.distance_unnormalized(h) / self.ni_scales

    def distance_unnormalized(self, h):
        distance = 0.0

        for s in range(self.ni_scales):
            self.m_tmpres[s] = np.sqrt(self.hsv_count[s] * h.hsv_count[s])
            distance += np.sqrt(1.0 - np.sum(self.m_tmpres[s]))

        return max(0.0, distance)

