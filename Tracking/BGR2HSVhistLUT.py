import math
import numpy as np
from .BGRtoHSVdistLUT import BGRtoHSVdistLUT
import pickle
import os

class BGR2HSVhistLUT:
    def __init__(self, _h_bins, _s_bins, _v_bins, s_threshold=0.1, v_threshold=0.2):

        self.h_bins = _h_bins
        self.s_bins = _s_bins
        self.v_bins = _v_bins

        self.bgr_to_colour_bin = np.zeros(256 * 256 * 256)
        self.computeLUT(s_threshold, v_threshold)



    def __del__(self):
        pass

    def save_to_file(self):
        lutfilename = f"lut_{self.h_bins}_{self.s_bins}_{self.v_bins}.pkl"
        with open(lutfilename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_lut_from_file(cls,nb_hbins, nb_sbins, nb_vbins):
        lutfilename = f"lut_{nb_hbins}_{nb_sbins}_{nb_vbins}.pkl"
        if os.path.exists(lutfilename):
            with open(lutfilename, "rb") as f:
                loaded_lut = pickle.load(f)
            return loaded_lut
        else:
            new_lut = cls(nb_hbins, nb_sbins, nb_vbins)
            new_lut.save_to_file()  # 尝试保存新模型文件
            return new_lut

    def computeLUT(self, s_threshold, v_threshold):
        h_step = 360.0 / self.h_bins
        s_step = 1.0 / self.s_bins
        v_step = 1.0 / self.v_bins
        norm = 1.0 / 255

        for b in range(256):
            indtmp1 = b * 256
            for g in range(256):
                indtmp2 = (indtmp1 + g) * 256
                for r in range(256):
                    h, s, v = BGRtoHSVdistLUT.RGBtoHSV(b * norm, g * norm, r * norm)

                    ind = indtmp2 + r

                    h_index = math.floor(h / h_step)  # h/h_step并向下取整，h_index={0,1,2...h_bins-1}
                    if h_index == self.h_bins:
                        h_index = self.h_bins - 1

                    s_index = math.floor(s / s_step)
                    if s_index == self.s_bins:
                        s_index = self.s_bins - 1

                    v_index = math.floor(v / v_step)
                    if v_index == self.v_bins:
                        v_index = self.v_bins - 1

                    #if s < s_threshold or v < v_threshold:
                        #self.bgr_to_colour_bin[ind] = (self.h_bins * self.s_bins + v_index)  # v通道独立
                    #else:
                    self.bgr_to_colour_bin[ind] = h_index * self.s_bins * self.v_bins + s_index *self.v_bins + v_index

