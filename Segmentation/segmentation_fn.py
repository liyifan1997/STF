import torch
from Segmentation.models.Unet import UNET as U_Net
import time
import cv2
import torch.backends.cudnn as cudnn
import numpy as np

def model_loading(path):
    '''Load model 
        input: path of the weigths
    '''
    model = U_Net().to('cuda')
    cudnn.benchmark = True
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['net'])
    print('\033[0;33mModel Loaded!\033[0m')
    return model

def inference(frame, model):
    '''Inference
        input: 
        image--> imported in cv2,
        model

        output:
            prediction,
            binary mask
    '''
    frame_width=frame.shape[1]
    frame_height=frame.shape[0]
    frame = cv2.resize(frame, (240, 160))
    frame = np.asarray(frame)
    frame = frame.transpose((2, 0, 1))  
    frame = frame / 255.0

    frame_tensor = torch.from_numpy(frame).float().unsqueeze(0).to('cuda') 
    with torch.no_grad():
        start = time.time()
        preds = torch.sigmoid(model(frame_tensor))
        preds = (preds > 0.55).float()
        end = time.time() - start
    
    segmented_frame = (preds.squeeze().cpu().numpy() * 255).astype(np.uint8)
    segmented_frame = cv2.resize(segmented_frame, (frame_width, frame_height))
    return preds, segmented_frame

def superimposition(mask, img, alpha=0.7, beta=0.3):
    mask=cv2.resize(mask, (1280, 960)).astype(np.uint8)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blended_frame = cv2.addWeighted(img, alpha, mask, beta, 0)
    return blended_frame

if __name__=="__main__":
    model=model_loading("best_model.pth")
    img=cv2.imread("img_demo.jpg")
    preds,mask=inference(img, model)
    cv2.imwrite("mask_demo.png", mask)
    overlap=superimposition(mask,img)
    cv2.imwrite("overlap.png", overlap)