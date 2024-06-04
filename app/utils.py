import ultralytics
from ultralytics import YOLO
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import cv2
import warnings
import ffmpeg

warnings.filterwarnings("ignore")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')




def segment(video_path: str, model_path: str, start: int, fstep: int, crop=None) -> tuple:
    """
    runs YOLO segmentation model and calculates the LV Area
    """
    model = YOLO(model_path)#.to(device)
    cap = cv2.VideoCapture(video_path)
    stop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lv_area = []
    frames = []
    message = 'Video processing succeeded'
    for fr in tqdm(range(start, stop, fstep), desc=f'processing ECHO'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        _, frame = cap.read()
        if crop is not None:
            new_w = int(frame.shape[1] * crop[1])
            new_h = int(frame.shape[0] * crop[0])
            new_left = int(frame.shape[1] / 2 + crop[3] * new_w - new_w / 2)
            new_top = int(frame.shape[0] / 2 + crop[2] * new_h - new_h / 2)
            frame = frame[new_top:new_top + new_h, new_left:new_left + new_w]
        frame_m = frame
        inputs = frame #torch.Tensor(frame).to(device)
        with torch.no_grad():
            result = model(inputs, verbose=False)
        result = result#.to('cpu')
        classes = result[0].names
        if len(classes) == 0:
            pass
        overlay = frame.copy()
        color_list = [(255, 0, 0),
                      (255, 255, 0),
                      (255, 0, 255),
                      (0, 255, 0),
                      (0, 0, 255),
                      (128, 128, 128)]
        for i, res in enumerate(result[0]):
            bx = res.boxes
            m = res.masks.xy
            label = int(bx.cls.squeeze().cpu())
            if label == 1:
                lv_area.append(cv2.contourArea(m[0]))
            box = list(map(int, bx.xyxy.squeeze().cpu().tolist()))
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (36, 255, 12), 2)
            cv2.putText(overlay, classes[label], (box[0], box[1] - 5), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
            cv2.fillPoly(overlay, pts=np.int32([m]), color=color_list[i % 6])
            alpha = 0.4
            frame_m = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        frames.append(frame_m)
        if len(lv_area) == 0:
            message = 'Video processing failed'
    return lv_area, frames, message


def plotter(lv_data: list, window: int) -> tuple:
    """
    plots the rolling mean graph for LV area.
    calculates the average ejection fracture
    """
    lv_rolling = pd.Series(lv_data).rolling(window=window).mean().dropna()
    ef = (max(lv_rolling) - min(lv_rolling)) / max(lv_rolling)
    dataframe = pd.DataFrame({
      'Frame': np.array(range(len(lv_rolling))),
      'Left ventricle visible area, px*px': lv_rolling.values
    }).astype('int32')
    txt = f'Ejection fraction - {ef:.1%}'
    return dataframe, txt
    

def writer(fn, images, framerate=25, vcodec='libx264'):
    if not isinstance(images, np.ndarray):
        images = np.asarray(images)
    n, height, width, channels = images.shape
    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
            .output(fn, pix_fmt='yuv420p', vcodec=vcodec, r=framerate)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )
    for frame in images:
        process.stdin.write(
            frame.astype(np.uint8).tobytes()
        )
    process.stdin.close()
    process.wait()