# MLT-Visual-Presentation-summarizer
Multilingual (MLT) visual presentation summarizer is a deep learning based solution that tries to convert any multilingual video presentation into readable pdf file as a summary of the presentation 


# Motivation

the main idea of this approach was to  find unique slides from presentation videos using the dbresnet text detection models detection output result on consecutive image pairs (it gives us information about similar text between image pairs and this feature can indicate the image pair are near duplicate or not.if duplicate then, one of them needs to be removed. but few difficulties were observed :

1. slide can contain various animation and in that case positional information are lost
2. launching dbnet on many images (frame images) is time consuming

to tackle these issues we combine both SSIM and dbresnet50 for efficient 3 step filtering. SSIM first reduces many frames and it is faster,then on reduced frames we apply dbresnet50 and ensemble of SSIM and dbresnet50 for efficient filtering


# summary / contributions

this notebook demonstrates how to convert multilingual presentation videos into readable pdf file. the solution is developed by combining structural similarity index measure (SSIM) with [Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion](https://arxiv.org/abs/2202.10304) of [paddleocr](https://github.com/PaddlePaddle/PaddleOCR)

we compute both ssim and dbnet using gpu,so the code is enough fast.



**Rough Pseudo Code :**

step 1 -> gather youtube video

step 2 -> extract distinct unique frames from videos

step 3 -> filter 1 : apply ssim on consecutive frame pairs with high confident on step 2 dataset to reduce redundant frames (use ssim on gpu)

step 4 -> filter 2 : select least frames by eliminating duplicate (use high IOU hit count between  image pairs for more filtering on filter 1 dataset)

step 5 -> filter 3(high detection overlap count followed by high ssim checking) : on filter 2 data, we again count matched bboxes between image pair(using dbresnet50) and check if we have high bbox detection coverage or not,if yes,then we eliminate first image from image pair as we have detected near duplicate again

**NOTE ->  the solution is multilingual,hence it should work on english,bangla,hindi,arabic etc presentation videos**
