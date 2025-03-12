import torch
from .BGRtoHSVdistLUT import BGRtoHSVdistLUT
from .GradDispLUT import GradDispLUT
import numpy as np
import math
import pickle
import os


class HSVPixelGradientModel:
    def __init__(self, nb_hsbins, nb_vbins, nb_obins, nb_mbins, mag_thresh):
        self.CLUSTER_SIZE = 3
        self.MAXVOTES=20
        self.h_bins = nb_hsbins
        self.s_bins = nb_hsbins
        self.v_bins = nb_vbins
        self.o_bins = nb_obins
        self.m_bins = nb_mbins
        self.maxcolourbin = self.h_bins * self.s_bins + self.v_bins
        self.totalbins = (self.h_bins * self.s_bins + self.v_bins) * (self.o_bins * self.m_bins + 1)  #(16*16+16)*(8+1)
        self.magnitude_threshold = mag_thresh

        self.disp = [{} for _ in range(self.totalbins)]  #disp是包含totalbins个空字典的集合，每个字典中包含一个'array'矢量和一个'count'数值
        self.m_LUTColour = BGRtoHSVdistLUT(self.h_bins, self.s_bins, self.v_bins) #r[0-255],g,b -> h[0-15]sv, b*256*25+g*256+r ~ 16*h+s或者 16*16+v
        self.m_LUTGradient = GradDispLUT(self.o_bins, self.m_bins) #gx[-1020~1020],gy[-1020~1020] -> angle[0-7]m[0-1], ((gx+1020) *2041)+(gy+1020) ~ angle[0~obin-1] 或者obin (m值小于阈值的情况)
        self.reset()


    def __del__(self):  #删除文件，减少资源占用
        for i in range(self.totalbins):
            self.disp[i].clear()
        del self.disp
        del self.m_LUTColour
        del self.m_LUTGradient

    def save_model_to_file(self):
        modelfilename = f"model_{self.h_bins}_{self.v_bins}_{self.o_bins}_{self.m_bins}_{self.magnitude_threshold}.pkl"
        with open(modelfilename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model_from_file(cls,nb_hsbins, nb_vbins, nb_obins, nb_mbins, mag_thresh):
        modelfilename = f"model_{nb_hsbins}_{nb_vbins}_{nb_obins}_{nb_mbins}_{mag_thresh}.pkl"
        if os.path.exists(modelfilename):
            with open(modelfilename, "rb") as f:
                loaded_model = pickle.load(f)
            return loaded_model
        else:
            new_model = cls(nb_hsbins, nb_vbins, nb_obins, nb_mbins, mag_thresh)
            new_model.save_model_to_file()  # 尝试保存新模型文件
            return new_model


    def reset(self):
        for i in range(self.totalbins):
            self.disp[i].clear()

    def learn(self, img, xgradimg, ygradimg, bb, segmentation,mask):  #bb为所框选的区域
        for i in range(self.totalbins):
            self.disp[i].clear()
        #将各个像素按照hsv_om bin划分，统计各个bin类别中指向区域中心的向量cur=(xi,yi)及该向量的个数
        bb[0]=max(bb[0], 1)
        bb[1] = min(bb[1], img.shape[0] - 2)  # bb[1],lastline
        bb[2] = max(bb[2], 1)  # firstcol
        bb[3] = min(bb[3], img.shape[1] - 2)  #lastcol   因为bb计算梯度，所以最外圈一层不考虑（在最外圈一层无法计算梯度）
        maxlocy=(bb[0]+bb[1])/2
        maxlocx = (bb[2] + bb[3]) / 2

        for j in range(bb[0],bb[1]+1):   #j遍历bb[0]~bb[1]
            for i in range(bb[2],bb[3]+1 ):  #i遍历bb[2]~bb[3]
                if mask[j,i]:
                    if segmentation[j][i] > 0.5:
                        b, g, r = img[j, i]
                        gx = int(xgradimg[j, i])
                        gy = int(ygradimg[j, i])

                        index_colour = int(self.m_LUTColour.bgr_to_dist[((b * 256) + g) * 256 + r])  # hsv bins的索引
                        index_gradient = int(
                            self.m_LUTGradient.grad_to_disp[((gx + 1020) * 2041) + (gy + 1020)])  # om bins的索引
                        index = index_gradient * self.maxcolourbin + index_colour  # hsv-om bins的索引

                        cur_disp = [j - maxlocy, i - maxlocx]  # 中心区域指向像素点的向量[y,x]
                        cur_disp[1] = cur_disp[1] // self.CLUSTER_SIZE
                        cur_disp[0] = cur_disp[0] // self.CLUSTER_SIZE
                        cur = (cur_disp[0], cur_disp[1])  # cur:(y//3,x//3)
                        if cur in self.disp[index]:
                            self.disp[index][cur] += 1
                        else:
                            self.disp[index][cur] = 1

        for i in range(self.totalbins):
            self.disp[i] = dict(sorted(self.disp[i].items(), key=lambda item: item[1], reverse=True))


    def vote(self, img, xgradimg, ygradimg, bb, voting_map, scale=1):
        bb[0] = max(bb[0], 1)
        bb[1] = min(bb[1], img.shape[0] - 2)  # bb[1],lastline
        bb[2] = max(bb[2], 1)  # firstcol
        bb[3] = min(bb[3], img.shape[1] - 2)  # lastcol
        bb_width=bb[3]-bb[2]+1
        bb_height=bb[1]-bb[0]+1
        width=img.shape[1]
        height=img.shape[0]
        xgrid_step = 1
        ygrid_step = 1
        roinx=bb_width//xgrid_step
        roiny=bb_height//ygrid_step

        ptr_y=bb[0]
        for j in range (roiny):
            ptr_x=bb[2]
            for i in range (roinx):
                b,g,r = img[ptr_y, ptr_x]
                gx= int(xgradimg[ptr_y, ptr_x])
                gy=int(ygradimg[ptr_y, ptr_x])
                index_colour = int(self.m_LUTColour.bgr_to_dist[((b * 256) + g) * 256 + r])  # hsv bins的索引
                index_gradient = int(self.m_LUTGradient.grad_to_disp[((gx + 1020) * 2041) + (gy + 1020)])  # om bins的索引
                index = index_gradient * self.maxcolourbin + index_colour
                disp_index=self.disp[index]  #这个bin中learn模块中的统计结果，从大到小排序,
                counter=0   #投票计数器
                for it in range (len(disp_index)):
                    counter += 1
                    if counter> self.MAXVOTES: #最多向20个点投票，每个点可能投多个票
                        break
                    ny1,nx1=list(disp_index.keys())[it] #中心到像素的向量
                    ny=int(ptr_y-ny1*self.CLUSTER_SIZE*scale)   #投票的y，
                    nx=int(ptr_x-nx1*self.CLUSTER_SIZE*scale)   #投票的x
                    if ny > 0 and nx > 0 and ny < height-1 and nx < width-1:  #投票的点在图像img范围内
                        voting_map[ny,nx] += disp_index[(ny1,nx1)]  #这个点的票数增加对应的值
                ptr_x += xgrid_step
            ptr_y += ygrid_step

    def backproject(self, img, xgradimg, ygradimg, bb, bpimg, maxlocx, maxlocy):
        bb[0] = max(bb[0], 1)
        bb[1] = min(bb[1], img.shape[0] - 2)  # bb[1],lastline
        bb[2] = max(bb[2], 1)  # firstcol
        bb[3] = min(bb[3], img.shape[1] - 2)  # lastcol
        bb_width = bb[3] - bb[2] + 1
        bb_height = bb[1] - bb[0] + 1
        xgrid_step = 1
        ygrid_step = 1
        roinx = bb_width // xgrid_step
        roiny = bb_height // ygrid_step

        ptr_y = bb[0]
        y_minus_maxlocy=ptr_y - maxlocy  #中心点到该像素的向量的y
        for j in range(roiny):
            ptr_x = bb[2]
            x_minus_maxlocx=ptr_x - maxlocx  #中心点到该像素的向量的x
            for i in range(roinx):
                b,g,r = img[ptr_y, ptr_x]
                gx = int(xgradimg[ptr_y, ptr_x])
                gy = int(ygradimg[ptr_y, ptr_x])
                index_colour = int(self.m_LUTColour.bgr_to_dist[((b * 256) + g) * 256 + r])  # hsv bins的索引
                index_gradient = int(self.m_LUTGradient.grad_to_disp[((gx + 1020) * 2041) + (gy + 1020)])  # om bins的索引
                index = index_gradient * self.maxcolourbin + index_colour
                disp_index = self.disp[index]  # 这个bin中learn模块中的统计结果，从大到小排序,
                counter = 0  # 投票计数器

                for it in range(len(disp_index)):

                    ytrans1, xtrans1 = list(disp_index.keys())[it]
                    ytrans=ytrans1*self.CLUSTER_SIZE
                    xtrans=xtrans1*self.CLUSTER_SIZE
                    center_dist = math.sqrt((ytrans - y_minus_maxlocy)**2 + (xtrans - x_minus_maxlocx)**2)
                    if center_dist<self.CLUSTER_SIZE*2.5:
                        counter += disp_index[(ytrans1,xtrans1)]
                        bpimg[ptr_y][ptr_x] += counter * math.exp(-0.3 * center_dist / self.CLUSTER_SIZE)
                    if counter >= self.MAXVOTES:  # 最多统计投向这个点的20张票
                        break
                ptr_x += xgrid_step
                x_minus_maxlocx +=xgrid_step
            ptr_y+=ygrid_step
            y_minus_maxlocy+=ygrid_step



    def backprojectWithSegmentation (self,img,xgradimg,ygradimg,bb,bpimg,maxlocx,maxlocy,segmentation): #backproject
        bb[0] = max(bb[0], 1)
        bb[1] = min(bb[1], img.shape[0] - 2)  # bb[1],lastline
        bb[2] = max(bb[2], 1)  # firstcol
        bb[3] = min(bb[3], img.shape[1] - 2)  # lastcol
        bb_width = bb[3] - bb[2] + 1
        bb_height = bb[1] - bb[0] + 1
        xgrid_step = max(1, bb_width // 90)
        ygrid_step = max(1, bb_height // 90)
        roinx = bb_width // xgrid_step
        roiny = bb_height // ygrid_step
        vote_err_pos = 0
        vote_err_pos2= 0
        vote_err_neg = 0
        vote_err_neg2 = 0
        fg_sum = 0
        bg_sum = 0


        ptr_y = bb[0]
        y_minus_maxlocy = ptr_y - maxlocy  # 中心点到该像素的向量的y
        for j in range(roiny):
            ptr_x = bb[2]
            x_minus_maxlocx = ptr_x - maxlocx  # 中心点到该像素的向量的x
            for i in range(roinx):
                fg=segmentation[ptr_y][ptr_x]
                bg=1-fg
                b,g,r = img[ptr_y, ptr_x]
                gx = int(xgradimg[ptr_y, ptr_x])
                gy = int(ygradimg[ptr_y, ptr_x])
                index_colour = self.m_LUTColour.bgr_to_dist[((b * 256) + g) * 256 + r]  # hsv bins的索引
                index_gradient = self.m_LUTGradient.grad_to_disp[((gx + 1020) * 2041) + (gy + 1020)]  # om bins的索引
                index = int(index_gradient * self.maxcolourbin + index_colour)
                disp_index = self.disp[index]  # 这个bin中learn模块中的统计结果，从大到小排序,
                counter = 0  # 投票计数器
                center_dist = 0  # 投票的点与中心点的距离
                count_votes=0
                if fg>0.5 :  #前景的像素
                    for it in range(len(disp_index)):
                        ytrans1, xtrans1 = list(disp_index.keys())[it]
                        ytrans = ytrans1 * self.CLUSTER_SIZE
                        xtrans = xtrans1 * self.CLUSTER_SIZE
                        center_dist1 = math.sqrt((ytrans - y_minus_maxlocy) ** 2 + (xtrans - x_minus_maxlocx) ** 2)
                        count_votes += 1
                        if center_dist1 < self.CLUSTER_SIZE*2.5:
                            counter += disp_index[(ytrans1, xtrans1)]
                            center_dist += center_dist1
                        if counter >= self.MAXVOTES:  # 最多统计投向这个点的20张票
                            break
                    if counter>0:
                        vote_err_pos += fg*center_dist/count_votes;
                        vote_err_pos2+=fg * center_dist * center_dist / (count_votes * count_votes)
                        fg_sum+=fg
                        bpimg[ptr_y][ptr_x]=counter
                else:
                    for it in range(len(disp_index)):
                        ytrans1, xtrans1 = list(disp_index.keys())[it]
                        ytrans = ytrans1 * self.CLUSTER_SIZE
                        xtrans = xtrans1 * self.CLUSTER_SIZE
                        center_dist = math.sqrt((ytrans - y_minus_maxlocy) ** 2 + (xtrans - x_minus_maxlocx) ** 2)
                        count_votes += 1
                        if center_dist < self.CLUSTER_SIZE:
                            counter += disp_index[(ytrans1, xtrans1)]
                        if counter >= self.MAXVOTES:  # 最多统计投向这个点的20张票
                            break
                    if counter > 0:
                        vote_err_neg +=bg*center_dist/count_votes
                        vote_err_neg2+=bg * center_dist * center_dist / (count_votes * count_votes)
                        bg_sum+=bg
                        bpimg[ptr_y][ptr_x] = counter

                ptr_x +=xgrid_step
                x_minus_maxlocx +=xgrid_step
            ptr_y +=ygrid_step
            y_minus_maxlocy +=ygrid_step

        if fg_sum >0:
            mean_pos =0
            variance_pos=vote_err_pos2/fg_sum -mean_pos*mean_pos

        if bg_sum >0:
            mean_neg =vote_err_neg/bg_sum
            variance_neg=vote_err_neg2/bg_sum -mean_neg*mean_neg
        print("voting error positive:  mean:", mean_pos)
        print("voting error positive:  var:  ", variance_pos)
        print("voting error negative:  mean:", mean_neg)
        print("voting error negative:  var:  ", variance_neg)

    def update(self,img,xgradimg,ygradimg,bb,segmentation,update_factor):
        bb[0] = max(bb[0], 1)
        bb[1] = min(bb[1], img.shape[0] - 2)  # bb[1],lastline
        bb[2] = max(bb[2], 1)  # firstcol
        bb[3] = min(bb[3], img.shape[1] - 2)  # lastcol
        one_minus_uf = 1 - update_factor
        for i in range (self.totalbins):
            for key in self.disp[i]:
                self.disp[i][key] *=one_minus_uf

        for j in range (bb[0],bb[1]+1):
            for i in range (bb[2],bb[3]+1):
                if segmentation[j][i]>0.5:
                    b,g,r = img[j, i]
                    gx = int(xgradimg[j, i])
                    gy = int(ygradimg[j, i])
                    index_colour = self.m_LUTColour.bgr_to_dist[((b * 256) + g) * 256 + r]  # hsv bins的索引
                    index_gradient = self.m_LUTGradient.grad_to_disp[((gx + 1020) * 2041) + (gy + 1020)]  # om bins的索引
                    index = int(index_gradient * self.maxcolourbin + index_colour)

                    cur_disp = [j - (bb[1] + bb[0]) / 2, i - (bb[2] + bb[3]) / 2]  # 中心区域指向像素点的向量[y,x]
                    cur_disp[1] = cur_disp[1] // self.CLUSTER_SIZE
                    cur_disp[0] = cur_disp[0] // self.CLUSTER_SIZE
                    cur = (cur_disp[0], cur_disp[1])  # cur:(y,x)
                    if cur in self.disp[index]:
                        self.disp[index][cur] += update_factor*segmentation[j][i]
                    else:
                        self.disp[index][cur] = update_factor*segmentation[j][i]

        for i in range(self.totalbins):
            self.disp[i] = dict(sorted(self.disp[i].items(), key=lambda item: item[1], reverse=True))






































