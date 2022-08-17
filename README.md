# MLT-Visual-Presentation-summarizer
Multilingual (MLT) visual presentation summarizer is a deep learning based solution that tries to convert any multilingual video presentation into readable pdf file as a summary of the presentation. 

**NOTE: This solution is multilingual, hence it should work on english, bangla, hindi, arabic etc presentation videos**


## Motivation: 

The main idea of this approach was to  find unique slides from presentation videos using the `dbresnet` text detection models detection output result on consecutive image pairs (It gives us information about similar text between image pairs and this feature can indicate the image pair are near duplicate or not. If duplicate then, one of them needs to be removed, but few difficulties were observed:

1. Slide can contain various animation and in that case positional information are lost.
2. Launching `dbnet` on many images (frame images) is time consuming.

To tackle these issues, we combine both `SSIM` and `dbresnet50` for efficient 3 step filtering. `SSIM` first reduces many frames and it is faster, then on reduced frames we apply `dbresnet50` and ensemble of `SSIM` and `dbresnet50` for efficient filtering.


## Summary / Contributions:

This work demonstrates how to convert multilingual presentation videos into readable pdf file. The solution is developed by combining structural similarity index measure (`SSIM`) with [Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion](https://arxiv.org/abs/2202.10304) of [paddleocr](https://github.com/PaddlePaddle/PaddleOCR).

**NB: We compute both ssim and dbnet using gpu, so the code is enough fast.**

---
**Rough Pseudo Code:**

* Step 1 -> Gather youtube videos / presentation videos. 

* Step 2 -> Extract distinct unique frames from videos.

* Step 3 -> Filter 1: Apply ssim on consecutive frame pairs with high confident on `Step 2` dataset to reduce redundant frames (use ssim on gpu).

* Step 4 -> Filter 2: Select least frames by eliminating duplicate (use high IOU hit count between image pairs for more filtering on `Filter 1` dataset).

* Step 5 -> Filter 3 (high detection overlap count followed by high ssim checking): On `Filter 2` data, we again count matched bboxes between image pair(using dbresnet50) and check if we have high bbox detection coverage or not, if yes, then we eliminate first image from image pair as we have detected near duplicate again.

* Step 6 -> Sort `Step 5` images and convert to pdf.

---
# Execution:

- ```conda activate your_env```
- ```cd scripts```
- run: ```./server.sh```
- OR, run:
    ```python
    python main.py -dd <data_dir> -vn <video_name / youtube link of video> -imd <imgs_dir> -opdf <output_dir_pdf>
    ```
    Ex:
    ```python
    python main.py -dd ../datas/ -vn test_1.mp4 -imd ../images/ -opdf ../outputs/
    ```
    ```python
    python main.py -dd ../datas/ -vn https://www.youtube.com/watch?v=k6lCD0iVExo -imd ../images/ -opdf ../outputs/
    ```

