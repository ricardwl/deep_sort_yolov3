#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from matplotlib import pyplot as plt
import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import pandas as pd
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

warnings.filterwarnings('ignore')
def findbbx(frame, frame_idx, df):
    '''

    :param frame: number of frame
    :param df: dataframe which saves the det
    :return: a list of bbox[x,y,w,h]
    '''
    bbx = []
    while True:
        if frame_idx >= df.shape[0]:
            break
        if df['frame'][frame_idx] != frame:
            break
        t, l, w, h = df['t'][frame_idx] , df['l'][frame_idx] , df['w'][frame_idx] , df['h'][frame_idx]
        bbx.append([t, l, w, h])

        frame_idx += 1
    #print(bbx)
    return bbx, frame_idx


def test():
    MOT_path = r'E:\projects\deep_sort_yolov3\data\MOT17/'
    img_path = '1.jpg'
    frame = cv2.imread(img_path)
    print(frame)
    bbox = [343.8498840332031,821.2205810546875,125.63116455078125,258.70751953125]
    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0])+int(bbox[2]), int(bbox[1])+int(bbox[3])), (255, 0, 0), 2)
    cv2.imshow(' ', frame)
    cv2.waitKey()
    cv2.destroyAllWindows()

def main(yolo):
    MOT_path = '/home/wangshuai/projects/data/CVPR19/test/CVPR19-06'
    # Definition of the parameters
    max_cosine_distance = 0.70
    nn_budget = None
    nms_max_overlap = 1.0
    # load det
    det_df = pd.read_csv(MOT_path+'/det/det_csv.csv')
    # deep_sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # det_df_idx
    det_df_idx = 0
    metric = nn_matching.NearestNeighborDistanceMetric("ad_cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, n_init=3, max_age=30)

    writeVideo_flag = False



    if writeVideo_flag:
        # Define the codec and create VideoWriter object
        w = 1920
        h = 734
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
        list_file = open('detection.txt', 'w')
    frame_index = 0
    imgs = os.listdir(MOT_path+'/img1')
    fps = 0.0
    imgs.sort()
    print(imgs)
    for img_path in imgs:
        if '.jpg' not in img_path:
            continue
        print('---------------------------------', frame_index + 1, '------------------')
        print(img_path)

        frame = cv2.imread(MOT_path + '/img1/' + img_path)
        # while True:
        #   ret, frame = video_capture.read()  # frame shape 640*480*3
        #  if ret != True:
        #     break
        t1 = time.time()

        image = Image.fromarray(frame)
        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxs,det_df_idx = findbbx(frame_index+1,det_df_idx, det_df)
        # print("box_num",len(boxs))
        features = encoder(frame, boxs)

        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        matches = tracker.update(detections)

        cv2.putText(frame, 'frame: ' + str(frame_index + 1), (30, 30), 0, 5e-3 * 200, (0, 255, 0), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)  # 白色
            cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0),
                        2)  # 绿色的字 + ' (' + str(bbox[0]) + ',' + str(bbox[1]) + ')'

        for i, det in enumerate(detections):
            bbox = det.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

       # cv2.imshow('', frame)

        if writeVideo_flag:
            # save a frame

            out.write(frame)
            frame_index = frame_index + 1
            # print('frame',frame_index)
            # print('check', matches)
            # print(len(tracker.tracks))
            for track_id, det_idx in matches:
                bbox = detections[det_idx].tlwh
                # print('(',track_id,det_idx,')',end=' ')
                id = track_id
                list_file.write(str(frame_index) + ',' + str(id) + ',' + str(bbox[0]) + ',' + str(bbox[1]) + ',' + str(
                    bbox[2]) + ',' + str(bbox[3]) + ',1,-1,-1')
                list_file.write('\n')
            # list_file.write(str(frame_index)+' ')
            # if len(boxs) != 0:
            #     for i in range(0,len(boxs)):
            #         list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
        # list_file.write('\n')

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %f" % (fps))

        # Press Q to stop!
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


    if writeVideo_flag:
        out.release()
        list_file.close()
    #cv2.destroyAllWindows()


if __name__ == '__main__':
    #test()
    main(YOLO())
