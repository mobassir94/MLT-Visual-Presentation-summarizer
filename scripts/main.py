#-*- coding: utf-8 -*-
"""
@author: MD.Nazmuddoha Ansary, Mobassir Hossain, Md. Rezwanul Haque
"""
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import os, errno
import cv2
import glob
import shutil
import argparse
from IPython.display import Video, Image
import pytube
from pytube import YouTube 
import zipfile
from paddleocr import PaddleOCR
import sys
sys.path.append('../')
from coreLib.utils import create_dir, is_supported, LOG_INFO, video_to_images, zipdir,\
    calculate_ssim, calculate_hausdorff_distance, inpaintredBox, viz_img_pair, count_matched_bboxes, unique_frames_to_pdf
from coreLib.config import Hparams

def main(args):
    ##---------------------------------
    data_dir        =   args.data_dir
    isYouTube_video =   args.isYouTube_video
    imgs_dir        =   args.imgs_dir
    output_pdf_path =   args.output_pdf_path
    ##---------------------------------

    ## Whether "imgs_dir" is exist or not
    try:
        os.makedirs(imgs_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    ## youtube video presentations
    if is_supported(isYouTube_video):
        yt = YouTube(isYouTube_video)
        # this method will download the highest resolution that video is available
        yt_video = yt.streams.get_highest_resolution()
        file_name = yt_video.download()

        ## check whether file is exist or not
        _file_name = file_name.split("/")[-1]
        file_exist = os.path.join(data_dir, _file_name)
        if os.path.exists(file_exist):
            os.remove(file_exist)
        shutil.move(file_name, data_dir)
        file_name = file_name.split("/")[-1]
        filename = os.path.join(data_dir, file_name)
        LOG_INFO(f"{filename}",mcolor="green") 

        #choosing dynamic fps based on video length for making computation fast without sacrificing vital informations
        if(yt.length>=1800):
            fps = 0.5
        elif(yt.length<1800 and yt.length>1200):
            fps = 0.75
        else:
            fps = 1.0
    else:
        filename = os.path.join(data_dir, isYouTube_video) 
        LOG_INFO(f"{filename}",mcolor="green") 
        video = cv2.VideoCapture(filename)
        length = video.get(cv2.CAP_PROP_POS_MSEC)
        #choosing dynamic fps based on video length for making computation fast without sacrificing vital informations
        if(length>=1800):
            fps = 0.5
        elif(length<1800 and length>1200):
            fps = 0.75
        else:
            fps = 1.0


    ## Controllable Parameters
    hp = Hparams(filename, fps)
    LOG_INFO(f"{hp.frames_per_second}", mcolor="green")

    if(hp.embed_video):
        Video(filename,embed=True)

    ## collecting least frames,required for finding unique slides
    frames, images_path = video_to_images(filename, imgs_dir, frames_per_second=hp.frames_per_second)
    LOG_INFO(f"Frames Number: {len(frames)}", mcolor="green")
    LOG_INFO(f"Images Path: {images_path}", mcolor="green")

    # Converting saved images directory into zip directory so that we can remove the folder later
    with zipfile.ZipFile('images.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir(images_path, zipf) # f'./images'
    
    ## check images whether extracted: f'./images'
    images = os.listdir(images_path)
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    ## annotation check in the key frames
    for idx in range(len(images)):
        img = cv2.imread(os.path.join(images_path, images[idx]))
        # remove annotattion paintng
        annot_res = inpaintredBox(img) 
        cv2.imwrite(os.path.join(images_path, images[idx]), annot_res)

    ## filtered images w.r.t. SSIM
    filtered_images_path = create_dir(imgs_dir, 'filtered')
    for idx in range(len(images) - 1):
        img1 = cv2.imread(os.path.join(images_path, images[idx]))
        img2 = cv2.imread(os.path.join(images_path, images[idx+1]))
        score = calculate_ssim(hp, img1, img2, filtered_images_path, images[idx])

    # ## filtered images w.r.t. Hausdorff Distance
    # filtered_images_path = create_dir(imgs_dir, 'filtered')
    # for idx in range(len(images) - 1):
    #     img1 = cv2.imread(os.path.join(images_path, images[idx]), 0)
    #     img2 = cv2.imread(os.path.join(images_path, images[idx+1]), 0)
    #     score = calculate_hausdorff_distance(hp, img1, img2, filtered_images_path, images[idx])
    #     # print("hausdorff distance", score)

    ## filtered images: f'./filtered' 
    images = os.listdir(filtered_images_path)
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    with zipfile.ZipFile('filtered.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir(filtered_images_path, zipf) # f'./filtered'

    ## frame pair visualization
    """
        * Source: https://gist.github.com/mstankie/71e49f628beac320953e0460b8ee78c2
        * Declare PaddleOCR class
    """
    ocr = PaddleOCR(use_angle_cls=True, lang='ar',use_gpu = True)  
    if(hp.visualize_img_pairs):
        viz_img_pair(ocr=ocr, images=images, filtered_images_path=filtered_images_path, i=4, j=5)

    ## Detect all unique and informative images and save them in folder name unique
    unique_images_path = create_dir(imgs_dir, 'unique')  #f'./unique' 
    '''
        detect all unique and informative images and save them
        images using db_resnet50 text detection algorithm of paddleocr
    '''
    files = os.listdir(filtered_images_path) # f'./filtered'
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for i in range(len(files)-1):
        prev = 1
        for j in range(i+1, len(files)):
            img=cv2.imread(os.path.join(filtered_images_path, files[i]))
            img1=cv2.imread(os.path.join(filtered_images_path, files[j]))

            try:
                img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
            except Exception as e:
                LOG_INFO(f"{e}",mcolor="red") 
           
            count,_ = count_matched_bboxes(hp, img,img1,detector=ocr)
            if(count>= prev):
                prev = count
                cv2.imwrite(os.path.join(unique_images_path, files[i]), img1)
            else:    
                break
    
    ## save unique Frames to pdf without final filtering # './unique'
    if(hp.withoutfinal_filter):
        unique_frames_to_pdf(hp, unique_images_path, output_pdf_path, True) 

    ## final filtering
    """
    *   at this point,it's possible that there still can exist few more redundant samples,
        they don't always look like redundant because of complex animation or other stuffs but 
        according to their other key features like "mid to near high bbox overlap coverage", 
        they are redundant,we try to do one last filtering to detect and eliminate those redundant samples.
    """
    images = os.listdir(unique_images_path)
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    for idx in range(len(images) - 1):
        image = cv2.imread(os.path.join(unique_images_path, images[idx]))
        image1 = cv2.imread(os.path.join(unique_images_path, images[idx+1]))
        if(hp.is_ssim):
            score = calculate_ssim(hp, image, image1, unique_images_path, images[idx], write_img=False)
            if(score>hp.ssim_threshold):
                os.remove(os.path.join(unique_images_path, images[idx]))
                continue
        try:
            img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            img1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
        except Exception as e:
            LOG_INFO(f"{e}",mcolor="red") 

        count,minimum = count_matched_bboxes(hp, img,img1,detector=ocr)
        if(count>= minimum * hp.conf_thr):
            os.remove(os.path.join(unique_images_path, images[idx]))

    LOG_INFO(f"Final Filtering Length: {len(os.listdir(unique_images_path))}", mcolor="green") 
    ## Save final PDF
    unique_frames_to_pdf(hp, unique_images_path, output_pdf_path)

    ## Save as a unique.zip
    with zipfile.ZipFile('unique.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir(unique_images_path, zipf)

    ## romove the the folders of extracted images
    if(hp.rmdir):
        sub_folders_pathname = imgs_dir
        sub_folders_list = glob.glob(sub_folders_pathname)
        for sub_folder in sub_folders_list:
            shutil.rmtree(sub_folder)

    ## move .zip file to "data_dir" path
    # images.zip
    _file_exist = os.path.join(data_dir, 'images.zip')
    if os.path.exists(_file_exist):
        os.remove(_file_exist)
    shutil.move(f'./images.zip', data_dir) 

    # filtered.zip
    _file_exist = os.path.join(data_dir, 'filtered.zip')
    if os.path.exists(_file_exist):
        os.remove(_file_exist)
    shutil.move(f'./filtered.zip', data_dir) 

    # unique.zip
    _file_exist = os.path.join(data_dir, 'unique.zip')
    if os.path.exists(_file_exist):
        os.remove(_file_exist)
    shutil.move(f'./unique.zip', data_dir)     

   
if __name__=="__main__":
    '''
        parsing and executions
    '''
    parser = argparse.ArgumentParser("MLT-Visual-Presentation-summarizer.")
    parser.add_argument("-dd", "--data_dir", help="Path to source data") 
    parser.add_argument("-vn", "--isYouTube_video", help="Link of Youtube Video")
    parser.add_argument("-imd", "--imgs_dir", help="Path to save extracted frames")
    parser.add_argument("-opdf", "--output_pdf_path", help="Path to save PDF")
    
    args = parser.parse_args()
    main(args)