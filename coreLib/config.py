#-*- coding: utf-8 -*-
"""
@author: Mobassir Hossain
"""
class Hparams:
    '''
        * choosing dynamic fps based on video length 
            for making computation fast without sacrificing vital informations.
    '''
    def __init__(self, title, fps):
        '''
            * Controllable Parameters
            args:
                title   :   title of the video.
                fps     :   video's fps
        '''
        self.title                  =   title
        self.embed_video            =   False
        self.rmdir                  =   True
        self.frames_per_second      =   fps
        self.ssim_threshold         =   0.98
        self.visualize_img_pairs    =   False
        self.iou_threshold          =   0.9
        self.min_match_thr          =   2
        self.conf_thr               =   0.8
        self.withoutfinal_filter    =   True
        self.is_ssim                =   False