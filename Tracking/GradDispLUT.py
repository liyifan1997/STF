import numpy as np
import math
#投票的方向向量转化为极坐标bin的形式，obin为极坐标角度的bin，mbin为极坐标长度的bin,该类的用法如下
# o_bins = 10
# m_bins = 5
# m_threshold = 0.1
#
# lut_instance = GradDispLUT(o_bins, m_bins, m_threshold)
# lookup_table = lut_instance.grad_to_disp



class GradDispLUT:
    def __init__(self, o_bins, m_bins, m_threshold=50):
        self.o_bins = o_bins
        self.m_bins = m_bins
        self.grad_to_disp = np.zeros(2041 *2041)
        self.compute_luts(m_threshold)

    def __del__(self):
        del self.grad_to_disp

    def compute_luts(self, m_threshold):
        maxmag = int(math.sqrt(2 * 1020 * 1020))  #距离中心点的最大距离

        for x in range(2041):  #x[0-2040]
            indtmp1 = x *2041
            for y in range(2041):
                ind = indtmp1 + y #索引ind为x*2^12+y

                gx = x - 1020 #gx[-1020,1020],x方向的sobel梯度
                gy = y - 1020  #gy
                angle = math.atan2(gy, gx)  #转化为极坐标的角度，-pi~pi
                magnitude = math.sqrt(gx * gx + gy * gy)  #极坐标的长度

                if magnitude < m_threshold:
                    self.grad_to_disp[ind] = (self.o_bins * self.m_bins)
                else:
                    obin = int((angle + math.pi) / (2 * math.pi) * self.o_bins) #范围为[0，o_bin)，然后int向下取整，[0~obin-1]
                    if obin >= self.o_bins:
                        obin = self.o_bins - 1
                    mbin = int((float(magnitude) / maxmag) * self.m_bins)
                    if mbin >= self.m_bins:
                        mbin = self.m_bins - 1
                    self.grad_to_disp[ind] = obin * self.m_bins + mbin