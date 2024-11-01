import os
import cv2
import csv
import glob
import argparse
from collections import defaultdict

def extract_unannoted(input_dir, extract_every_n_frames):
    for dirpath in glob.glob(os.path.join(input_dir, "videos", "*")):

        video_name = os.path.basename(dirpath)

        if video_name == "unannotated":
            continue

        print("Preparing %s..." % video_name)
        video_frame_output_path = os.path.join("../../../Datasets/NjordVAE", video_name)

        video_filepath = os.path.join(dirpath, "%s.mp4" % video_name)
        video = cv2.VideoCapture(video_filepath)
        extracted_frame_ids = []
        frame_index = 0

        while (video.isOpened()):

            ret, frame = video.read()

            if ret == False:
                break

            if frame_index % extract_every_n_frames == 0:
                extracted_frame_ids.append(frame_index)
                frame_filepath = os.path.join(video_frame_output_path, "%s_frame_%i.jpg" % (video_name, frame_index))
                cv2.imwrite(frame_filepath, frame)

            frame_index += 1

        video.release()

if __name__ == '__main__':
    extract_unannoted("/home/birk/PhDPlayground/Datasets/Njord", 25)
