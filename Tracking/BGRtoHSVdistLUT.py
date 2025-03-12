import math
import numpy as np
#计算rgb到hsv_bin的查找表，导入方法为：
# h_bins = 10
# s_bins = 5
# v_bins = 3
# s_threshold = 0.2
# v_threshold = 0.4
#
# lut_instance = BGRtoHSVdistLUT(h_bins, s_bins, v_bins, s_threshold, v_threshold)
# lookup_table = lut_instance.bgr_to_dist
#hsv=lut_instance.RGBtoHSV(255, 0, 0)

class BGRtoHSVdistLUT:
    def __init__(self, h_bins, s_bins, v_bins, s_threshold=0.1, v_threshold=0.2):
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins

        self.bgr_to_dist = np.zeros(256 * 256 * 256)
        self.compute_luts(s_threshold, v_threshold)

    def __del__(self):
        del self.bgr_to_dist

    @staticmethod
    def RGBtoHSV(r, g, b):  #rgb到hsv的计算
        vmin = v = r
        if v < g: v = g
        if v < b: v = b    #v=max(r,g,b)
        if vmin > g: vmin = g
        if vmin > b: vmin = b

        diff = v - vmin
        s = diff / (float(abs(v)) + 1e-07)
        diff = 60. / (diff + 1e-07)

        if v == r:
            h = (g - b) * diff
        elif v == g:
            h = (b - r) * diff + 120.0
        else:
            h = (r - g) * diff + 240.0

        if h < 0:
            h += 360.0

        return h, s, v

    def compute_luts(self, s_threshold, v_threshold):


        h_step = 360. / self.h_bins   #h:0-360,
        s_step = 1. / self.s_bins    #s:0-1
        v_step = 1. / self.v_bins   #v:0-1
        norm = 1.0 / 255    #rgb:(0-255)->(0-1)


        for b in range(256):   #b:0-255
            indtmp1 = b * 256
            for g in range(256):
                indtmp2 = (indtmp1 + g) * 256
                for r in range(256):
                    h,s,v=self.RGBtoHSV(b * norm, g * norm, r * norm)

                    ind = indtmp2 + r  #ind=b*256^2+g*256+r

                    h_index = math.floor(h / h_step)   #h/h_step并向下取整，h_index={0,1,2...h_bins-1}
                    if h_index == self.h_bins:
                        h_index = self.h_bins - 1

                    s_index = math.floor(s / s_step)
                    if s_index == self.s_bins:
                        s_index = self.s_bins - 1

                    v_index = math.floor(v / v_step)
                    if v_index == self.v_bins:
                        v_index = self.v_bins - 1

                    if s < s_threshold or v < v_threshold:
                        self.bgr_to_dist[ind] = self.h_bins * self.s_bins + v_index  #v通道独立
                    else:
                        self.bgr_to_dist[ind] = h_index * self.s_bins + s_index

