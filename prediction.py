#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:41:13 2016

@author: vijay
"""

import caffe
import cv2
import numpy as np

###############################################################################

class OCR:
    
    def __init__(self,prototxt,model,synset_file):
        
        caffe.set_mode_cpu()
        self.prototxt=prototxt
        self.model=model
        self.synset_file=synset_file
        
        self.synset=self.gen_synset(self.synset_file)        
        (self.net,self.transformer)=self.init_network(self.prototxt,self.caffemodel)
        
        
    #### translate synsetfile
    
    def gen_synset(self,synset_file):
        
        synset={}
        cnt=0
        with open(synset_file) as f:
            content = f.readlines()
            synset[cnt]=content
            cnt+=1
            
        return synset
        
    #### init network
    
    def init_network(self,prototxt,caffemodel):
        
        net = caffe.Net(prototxt,caffemodel,caffe.TEST)
        
        # load input and configure preprocessing
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_channel_swap('data', (2,1,0))
        transformer.set_raw_scale('data', 255.0)
        net.blobs['data'].reshape(1,3,32,32)
        
        return (net,transformer)
            
    #### predict image
    
    def predict(self,im):
        
        net=self.net
        transformer=self.transformer
        synset=self.synset
        
        img=caffe.io.load_image(im)
        net.blobs['data'].data[...] = transformer.preprocess('data', img)
        out = net.forward()
        pre=out['softmax']
        maxim=pre.argmax()
        
        pred=synset[maxim]

        return pred