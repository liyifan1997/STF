from Tracking.yifanTracking import yifanTracking
from Tracking import toolfunction
import cv2
import os
from Segmentation import segmentation_fn
import numpy as np
import time

# 设置输入视频
video_path = 'test1.mp4'
cap = cv2.VideoCapture(video_path)
current_frame = 0

# 创建实时显示窗口
cv2.namedWindow('Tracking Result', cv2.WINDOW_NORMAL)

# 跳到起始帧
while current_frame < 206:  # 从清晰的帧开始
    ret, img = cap.read()
    current_frame += 1

# 第一次分割
segmodel = segmentation_fn.model_loading("best_model.pth")
ret, img = cap.read()
current_frame += 1

start_time = time.time()
preds, mask = segmentation_fn.inference(img, segmodel)
mask = toolfunction.mask_analysis(mask)

overlap = segmentation_fn.superimposition(mask, img)

# 显示第一次分割结果
cv2.imshow('Tracking Result', overlap)
cv2.waitKey(1)  # 更新显示

trackModel = yifanTracking(12, 12, img)
mask = trackModel.segimproving(img, mask, 0.3)
mask = toolfunction.mask_analysis(mask)

overlap = segmentation_fn.superimposition(mask, img)
cv2.imshow('Tracking Result', overlap)
cv2.waitKey(1)  # 更新显示

trackresult = trackModel.firstTracking(img, mask)
trackresult = toolfunction.mask_analysis(trackresult)
end_time = time.time()
elapsed_time = end_time - start_time
resegvalue = np.sum(trackresult) / 255
overlap = segmentation_fn.superimposition(trackresult, img)

# 显示第一次跟踪结果
cv2.imshow('Tracking Result', overlap)
cv2.waitKey(1)  # 更新显示

reseg = 0
while True:
    ret, img = cap.read()
    current_frame += 1
    if not ret:
        break

    start_time = time.time()
    trackresult, bayresult = trackModel.tracking(img)
    trackresult = toolfunction.mask_analysis(trackresult)

    kernel = np.ones((3, 3), np.uint8)

    # 应用形态学操作
    trackresult = cv2.dilate(trackresult, kernel, iterations=1)
    trackresult = cv2.erode(trackresult, kernel, iterations=1)
    trackresult = cv2.morphologyEx(trackresult, cv2.MORPH_OPEN, kernel)
    trackresult = cv2.morphologyEx(trackresult, cv2.MORPH_CLOSE, kernel)

    end_time = time.time()
    elapsed_time = end_time - start_time

    if elapsed_time > 0.1:
        trackModel.trackInitialization(img, bayresult)
    overlap = segmentation_fn.superimposition(trackresult, img)

    # 实时显示当前跟踪结果
    cv2.imshow('Tracking Result', overlap)

    # 检查按键（例如，按'q'退出）
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if current_frame == 1120:
        preds, mask = segmentation_fn.inference(img, segmodel)
        mask = toolfunction.masksearch(mask, trackModel.search_box)
        mask = toolfunction.mask_analysis(mask)
        iou = toolfunction.ioucal(mask, trackresult)

        if iou > 0.5:
            trackModel = yifanTracking(12, 12, img)
            mask = trackModel.segimproving(img, mask, 0.3)
            mask = toolfunction.mask_analysis(mask)
            # mask = trackresult
            trackresult = trackModel.firstTracking(img, mask)
            resegvalue = np.sum(trackresult) / 255

            reseg = 1

    if current_frame == 1500:
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()











