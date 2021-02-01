import random
import hyperparams as hyp
from collections import defaultdict
import ipdb
import torch
st = ipdb.set_trace
import numpy as np
class SinglePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.used = False
        random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.embeds = []
            
    def fetch(self):
        return self.embeds
            
    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full
            
    def update(self, embeds):
        # embeds is B x ... x C
        # images is B x ... x 3
        for embed  in embeds:
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                # the pool is full
                # pop from the front
                self.embeds.pop(0)
            # add to the back
            self.embeds.append(embed)
        return self.embeds

class ClusterPool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.embeds = []
            self.classes = []
            
    def fetch(self):
        return self.embeds, self.classes
            
    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full
    def empty(self):
        self.num = 0
        self.embeds = []
        self.classes = []
    def update(self, embeds, classes):
        # embeds is B x ... x C
        # images is B x ... x 3

        for embed, class_val in zip(embeds, classes):
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                self.embeds.pop(0)
                self.classes.pop(0)
            self.embeds.append(embed)
            self.classes.append(class_val)
        return self.embeds, self.classes


class DetPool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.filenames = []
            self.feat_masks = []
            self.boxes = []
            self.gt_boxes = []
            self.gt_scores = []
            self.scores = []
            
    def fetch(self):
        return self.filenames, self.feat_masks,self.boxes,self.gt_boxes, self.scores, self.gt_scores
            
    def is_full(self):
        full = self.num==self.pool_size
        return full

    def update(self,feat_masks, boxes, gt_boxes, scores, gt_scores, filenames):
        for feat_mask, box, gt_box, score, gt_score, filename in zip(feat_masks, boxes,gt_boxes, scores, gt_scores,   filenames):
            if self.num < self.pool_size:
                self.num = self.num + 1
            else:
                self.filenames.pop(0)
                self.feat_masks.pop(0)
                self.boxes.pop(0)
                self.gt_boxes.pop(0)
                self.gt_scores.pop(0)
                self.scores.pop(0)                                

            self.filenames.append(filename)
            self.feat_masks.append(feat_mask)
            self.boxes.append(box)
            self.scores.append(score)
            self.gt_boxes.append(gt_box)
            self.gt_scores.append(gt_score)


class DoublePool_O():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.embeds = []
            self.visual2D = []
            self.images = []
            self.classes = []
            
    def fetch(self):
        return self.embeds, self.images, self.classes,None, self.visual2D
            
    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full
            
    def update(self, embeds, images, classes, vis2Ds):
        # embeds is B x ... x C
        # images is B x ... x 3

        for embed, image, class_val,vis2D in zip(embeds, images,classes, vis2Ds):
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                # the pool is full
                # pop from the front
                self.embeds.pop(0)
                self.images.pop(0)
                self.classes.pop(0)
                self.visual2D.pop(0)
            # add to the back
            self.embeds.append(embed)
            self.images.append(image)
            self.classes.append(class_val)
            self.visual2D.append(vis2D)
        # return self.embeds, self.images

class DoublePool_O_f():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.embeds = []
            self.images = []
            self.classes = []
            self.filenames = []
            self.unp_visRs = []
            
    def fetch(self):
        return self.embeds, self.images, self.classes, self.filenames,None
    
    def fetchUnpRs(self):
        return self.unp_visRs
            
    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full

    def update(self, embeds, images, classes, filenames, unp_visRs=None):
        # embeds is B x ... x C
        # images is B x ... x 3
        if unp_visRs is None:
            unp_visRs = [None]*embeds.shape[0]

        for embed, image, class_val, filename, unp_visR in zip(embeds, images,classes,filenames, unp_visRs):
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                # the pool is full
                # pop from the front
                self.embeds.pop(0)
                self.images.pop(0)
                self.classes.pop(0)
                self.filenames.pop(0)

                self.unp_visRs.pop(0)
            # add to the back
            self.embeds.append(embed)
            self.images.append(image)
            self.classes.append(class_val)
            self.filenames.append(filename)
            self.unp_visRs.append(unp_visR)

        return self.embeds, self.images


class MOC_QUEUE():
    def __init__(self,val):
        self.pool_size = val
        random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.embeds = []
            
    def fetch(self):
        return self.embeds
            
    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full
            
    def update(self, embeds,classes=None):
        # embeds is B x ... x C
        # images is B x ... x 3
        if not hyp.max.object_level_gt:
            embeds = embeds.t()
        for embed in embeds:
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                # the pool is full
                # pop from the front
                self.embeds.pop(0)
            # add to the back
            self.embeds.append(embed)


class MOC_DICT():
    def __init__(self,val):
        self.dictionary_embeds = defaultdict(lambda:MOC_QUEUE(val))
        random.seed(125)
        self.size = val
        self.is_full = lambda:self.check_full()
    def check_full(self):
        check = True
        if not hyp.exp.do_debug:
            if len(self.dictionary_embeds.keys()) < hyp.num_classes:
                print("All classes are not present yet!")
                return False 
        for key,item in self.dictionary_embeds.items():
            is_full = item.is_full()
            if not is_full:
                check = False
                break
        return check
        
    def fetch_negatives(self,class_val):
        embeds = []
        
        dictionary_embeds = dict(self.dictionary_embeds)
        for key,val in dictionary_embeds.items():
            if key != class_val:
                embeds.append(torch.stack(val.fetch(),dim=0))
        embeds = torch.cat(embeds,dim=0)
        size = embeds.shape[0]
        r = torch.randperm(size)
        embeds = embeds[r]
        embeds =  embeds[:self.size]
        return embeds

    def fetch(self):
        return self.dictionary_embeds
            
    def update(self, embeds, classes):
        
        for embed, class_val in zip(embeds,classes):
            if hyp.max.object_level_gt:
                embed = embed.unsqueeze(0) # For compatibility with existing code.
            self.dictionary_embeds[class_val].update(embed)


class MOC_QUEUE_NORMAL():
    def __init__(self,val):
        self.pool_size = val
        random.seed(125)
        if self.pool_size > 0:
            self.num = 0
            self.embeds = []
            
    def fetch(self):
        return self.embeds
            
    def is_full(self):
        full = self.num==self.pool_size
        # print 'num = %d; full = %s' % (self.num, full)
        return full
            
    def update(self, embeds,classes=None):
        # embeds is B x ... x C
        # images is B x ... x 3
        embeds = embeds.permute(0,2,1).reshape(-1,hyp.feat_dim)
        for embed in embeds:
            if self.num < self.pool_size:
                # the pool is not full, so let's add this in
                self.num = self.num + 1
            else:
                # the pool is full
                # pop from the front
                self.embeds.pop(0)
            # add to the back
            self.embeds.append(embed)