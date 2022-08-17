#!/bin/sh

##------------------------------------------main.py------------------------------------------------
data_dir="../datas/"
# isYouTube_video="https://www.youtube.com/watch?v=k6lCD0iVExo"
isYouTube_video="test_1.mp4"
imgs_dir="../images/"
output_pdf_path="../outputs/"

python main.py -dd $data_dir -vn $isYouTube_video -imd $imgs_dir -opdf $output_pdf_path
#--------------------------------------------------------------------------------------------------------------
echo succeded