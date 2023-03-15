#-*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary, Mobassir Hossain, Md. Rezwanul Haque
"""
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
import os 
import math
import random
import numpy as np 
import torch
import cv2 
import youtube_dl
from torchvision import transforms
from piqa import SSIM
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
import img2pdf
from shapely.geometry import Polygon
# from hausdorff import hausdorff_distance
from scipy.spatial.distance import directed_hausdorff
#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
def create_dir(base,ext):
    '''
        creates a directory extending base
        args:
            base    =   base path 
            ext     =   the folder to create
    '''
    _path=os.path.join(base,ext)
    if not os.path.exists(_path):
        os.mkdir(_path)
    return _path
#---------------------------------------------------------------
def is_supported(url):
    '''
        * Source: https://stackoverflow.com/a/61489622/5424617

        check if a url is valid that youtube-dl supports
    '''
    extractors = youtube_dl.extractor.gen_extractors()
    for e in extractors:
        if e.suitable(url) and e.IE_NAME != 'generic':
            return True
    return False
#---------------------------------------------------------------
def video_to_images(video_path, imgs_path, frames_per_second=1):
    '''
        * Source: https://stackoverflow.com/questions/54045766/python-extracting-distinct-unique-frames-from-videos
        
        Extracting distinct unique frames from videos
        args:
            video_path      =   path directory of the selected video.
            imgs_path       =   path directoy of extracted frames.
    '''
    cam = cv2.VideoCapture(video_path)
    frame_list = []
    # video frame rate
    frame_rate = cam.get(cv2.CAP_PROP_FPS) 

    # frame
    current_frame = 0
    
    # create directory if it does not exist
    video_filename = (video_path.split("/")[-1]).split(".")[0]
    images_path = create_dir(imgs_path, str(video_filename))


    if frames_per_second > frame_rate or frames_per_second == -1:
        frames_per_second = frame_rate
    
    while True:
        # reading from frame
        ret,frame = cam.read()
        if ret:
            # if video is still left continue creating images
            file_name = os.path.join(images_path, 'frame'+str(current_frame)+'.jpg')
            if current_frame % (math.floor(frame_rate/frames_per_second)) == 0:
                # adding frame to list
                frame_list.append(frame)
                # writing selected frames to images_path
                cv2.imwrite(file_name, frame)
    
            # increasing counter so that it will show how many frames are created
            current_frame += 1
        else:
            break
    
    # Release all space and windows once done
    cam.release()

    return frame_list, images_path
#---------------------------------------------------------------
def zipdir(path, ziph):
    """
        * ziph is zipfile handle
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))
#---------------------------------------------------------------
def calculate_ssim(hp, image, image1, filtered_images_path, image_name, write_img=True):
    """
        * structural similarity index measure (SSIM) calculation using GPU
        * if the consecutive image pair gets ssim score above "hp.ssim_threshold" 
        then we consider first image from the pair as it's duplicate or 
        say near duplicate then we increment i,j for next comparison to 
        filter redundant samples using high ssim threshold like 
        ->  score < hp.ssim_threshold
        args:
            image    =   1st image from image pair
            image1   =   2nd image from image pair

    """
    transform = transforms.ToTensor()
    gpu = torch.cuda.is_available()
    # print(gpu)
    if gpu:
        ssim = SSIM().cuda()
        x = transform(image).unsqueeze(0).cuda() # .cuda() for GPU
        y = transform(image1).unsqueeze(0).cuda()
    else:
        ssim = SSIM()
        x = transform(image).unsqueeze(0)
        y = transform(image1).unsqueeze(0)

    score = ssim(x, y).data.tolist()

    if(score > hp.ssim_threshold):
        return score
    if(write_img):
        cv2.imwrite(os.path.join(filtered_images_path, image_name), image)
    return score
#---------------------------------------------------------------
def calculate_hausdorff_distance(hp, image, image1, filtered_images_path, image_name, write_img=True):
    """
        * computes the Hausdorff distance between the rows of X and Y using the Euclidean distance as metric.
        * Link: https://github.com/mavillan/py-hausdorff
        args:
            image    =   1st image from image pair
            image1   =   2nd image from image pair

    """
    # score = hausdorff_distance(image, image1, distance='euclidean')
    score = directed_hausdorff(image, image1)[0]
    if(score < hp.hausdorff_threshold):
        return score
    if(write_img):
        cv2.imwrite(os.path.join(filtered_images_path, image_name), image)
    return score
#---------------------------------------------------------------
def inpaintredBox(img):
    #Invert and convert to HSV
    img_hsv = cv2.cvtColor(255 - img, cv2.COLOR_BGR2HSV)

    #mask all red pixels (cyan in inverted image)
    lo = np.uint8([80, 30, 0])
    hi = np.uint8([95, 255, 255])

    mask = cv2.inRange(img_hsv, lo, hi)

    # Inpaint red box
    result = cv2.inpaint(img, mask, random.randint(3,7), cv2.INPAINT_TELEA)

    return result
#---------------------------------------------------------------
def viz_img_pair(ocr, images, filtered_images_path, i,j):
    """
        * vizualize image pair
        * structural_similarity score show  
        args:
            ocr     =   PaddleOCR
            images  =   list of images
            image   =   1st image from image pair
            image1  =   2nd image from image pair 
    """
    image = cv2.imread(os.path.join(filtered_images_path, images[i]))
    image1 = cv2.imread(os.path.join(filtered_images_path, images[j]))
    
    result = ocr.ocr(image, rec=False)
    boxes = [line[0] for line in result]
    for box in result:
        box = np.reshape(np.array(box), [-1,1,2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255,0,0), 2)

    f = plt.figure(figsize=(20,20))
    f.add_subplot(1,2, 1)
    plt.title(f'{images[i]}', fontdict={'fontsize':20})
    plt.imshow(image)
    
    result = ocr.ocr(image1, rec=False)
    boxes = [line[0] for line in result]
    for box in result:
        box = np.reshape(np.array(box), [-1,1,2]).astype(np.int64)
        image1 = cv2.polylines(np.array(image1), [box], True, (255,0,0), 2)

    f.add_subplot(1,2, 2)
    plt.title(f'{images[i+1]}', fontdict={'fontsize':20})
    plt.imshow(image1)
    plt.show(block=True)

    # Convert images to grayscale
    before_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    LOG_INFO(f"Image Similarity: {score}", mcolor="red")
#---------------------------------------------------------------
def sorted_boxes(dt_boxes):
    """
        * Source: https://github.com/vigneshgig/sorting_algorthim_for_bounding_box_from_left_to_right_and_top_to_bottom
        Sort text boxes in order from top to bottom, left to right
        args:
            dt_boxes(array) =   detected text boxes with shape [4, 2]
        return:
            sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 100 and (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes
#---------------------------------------------------------------
def calculate_iou(box_1, box_2):
    """
        * calculate Intersection Over Union
    """
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou
#---------------------------------------------------------------
def count_matched_bboxes(hp, img, img1, detector):
    """
    count the matched bboxes
    args:
        hp          =   Hyperparameters params Class
        img1        =   1st image from image pair
        img2        =   2nd image from image pair
        detector    =   PaddleOCR
    """
    
    dt_boxes,_= detector.text_detector(img) 
    dt_boxes=sorted_boxes(dt_boxes)
    
    dt_boxes1,_= detector.text_detector(img1) 
    dt_boxes1=sorted_boxes(dt_boxes1)

    minimum = min(len(dt_boxes), len(dt_boxes1))
    
    if(len(dt_boxes)<1):
        # we don't care if there is no text
        return 0,minimum
    
    if(len(dt_boxes1)<1):
        # we don't care if there is no text
        return 0,minimum

    count = 0
    if(len(dt_boxes)>len(dt_boxes1)):
        for box_n in range(len(dt_boxes1)):
            for box_num in range(len(dt_boxes)):
                iou = calculate_iou(dt_boxes[box_num], dt_boxes1[box_n])
                if(iou > hp.iou_threshold):
                    count+=1
    else:
        for box_num in range(len(dt_boxes)):
            for box_n in range(len(dt_boxes1)):
                iou = calculate_iou(dt_boxes[box_num], dt_boxes1[box_n])
                if(iou > hp.iou_threshold):
                    count+=1
    
    return count, minimum
#---------------------------------------------------------------
def unique_frames_to_pdf(hp, output_folder_screenshot_path, out_pdf_path, without_final_filter = False):
    """
        * Function for creating PDF from unique frames 
    """
    _file_name = hp.title
    _file_name = _file_name.split("/")[-1]
    if(without_final_filter):
        output_pdf_path = os.path.join(out_pdf_path, f'w.o.f_{_file_name}.pdf')
    else:
        output_pdf_path = os.path.join(out_pdf_path, f'{_file_name}.pdf')

    # LOG_INFO(f"output folder screenshot path: {output_folder_screenshot_path}",mcolor="green")
    # LOG_INFO(f"output pdf path: {output_pdf_path}",mcolor="green")
    LOG_INFO(f"Converting Images to pdf...",mcolor="green")

    images = os.listdir(output_folder_screenshot_path)
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    for i in range(len(images)):
        images[i] = os.path.join(output_folder_screenshot_path, images[i])

    with open(output_pdf_path, "wb") as f:
        f.write(img2pdf.convert(images))

    LOG_INFO(f"PDF Created!",mcolor="green")
    LOG_INFO(f"PDF saved at: {output_pdf_path}", mcolor="green")
#---------------------------------------------------------------