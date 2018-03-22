#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy
import random
from subprocess import *

from conf import *
import utils

random.seed(args.random_seed)

import cPickle
sys.setrecursionlimit(1000000)

MAX = 2

class DataGnerater():
    def __init__(self,file_type,max_pair): #file_type: train or test
        self.embedding = numpy.load(args.data + "embedding.npy")
        data_path = args.data+file_type+"/" 
        if args.reduced == 1:
            data_path = args.data+file_type + "_reduced/"

        self.candi_vec = numpy.load(data_path+"candi_vec.npy")
        self.candi_vec_mask = numpy.load(data_path+"candi_vec_mask.npy")
        self.ifl_vec = numpy.load(data_path+"ifl_vec.npy")

        self.zp_post_vec = numpy.load(data_path+"zp_post.npy")
        self.zp_post_vec_mask = numpy.load(data_path+"zp_post_mask.npy")
        self.zp_pre_vec = numpy.load(data_path+"zp_pre.npy")
        self.zp_pre_vec_mask = numpy.load(data_path+"zp_pre_mask.npy")

        read_f = file(data_path + "zp_candi_pair_info","rb")
        zp_candis_pair = cPickle.load(read_f)
        read_f.close()

        self.data_batch = []

        zp_reindex = []
        candi_reindex = []
        this_target = [] 
        this_result = []

        start2end = []

        for i in range(len(zp_candis_pair)):
            zpi,candis = zp_candis_pair[i]
            if len(candis)+len(candi_reindex) > max_pair and len(candi_reindex) > 0:
                # generate some outputs
                ci_s = candi_reindex[0]
                ci_e = candi_reindex[-1]+1
                zpi_s = zp_reindex[0]
                zpi_e = zp_reindex[-1]+1

                this_batch = {}
                this_batch["zp_reindex"] = numpy.array(zp_reindex,dtype="int32")-zp_reindex[0]
                this_batch["candi_reindex"] = numpy.array(candi_reindex,dtype="int32")-candi_reindex[0]
                this_batch["target"] = numpy.array(this_target,dtype="int32")
                this_batch["result"] = numpy.array(this_result,dtype="int32")

                this_batch["zp_post"] = self.zp_post_vec[zpi_s:zpi_e]
                this_batch["zp_pre"] = self.zp_pre_vec[zpi_s:zpi_e]
                this_batch["zp_post_mask"] = self.zp_post_vec_mask[zpi_s:zpi_e]
                this_batch["zp_pre_mask"] = self.zp_pre_vec_mask[zpi_s:zpi_e]
                this_batch["candi"] = self.candi_vec[ci_s:ci_e]
                this_batch["candi_mask"] = self.candi_vec_mask[ci_s:ci_e]
                this_batch["fl"] = self.ifl_vec[ci_s:ci_e]
                
                this_batch["start2end"] = start2end

                self.data_batch.append(this_batch)
               
                zp_reindex = []
                candi_reindex = []
                this_target = [] 
                this_result = []
                start2end = []

            start = len(this_result)
            end = start
            for candii,res in candis:
                zp_reindex.append(zpi)
                candi_reindex.append(candii)
                this_target.append([res])
                this_result.append(res)
                end += 1
            start2end.append((start,end))

        if len(candi_reindex) > 0:
            ci_s = candi_reindex[0]
            ci_e = candi_reindex[-1]+1
            zpi_s = zp_reindex[0]
            zpi_e = zp_reindex[-1]+1

            this_batch = {}
            this_batch["zp_reindex"] = numpy.array(zp_reindex,dtype="int32")-zp_reindex[0]
            this_batch["candi_reindex"] = numpy.array(candi_reindex,dtype="int32")-candi_reindex[0]
            this_batch["target"] = numpy.array(this_target,dtype="int32")
            this_batch["result"] = numpy.array(this_result,dtype="int32")

            this_batch["zp_post"] = self.zp_post_vec[zpi_s:zpi_e]
            this_batch["zp_pre"] = self.zp_pre_vec[zpi_s:zpi_e]
            this_batch["zp_post_mask"] = self.zp_post_vec_mask[zpi_s:zpi_e]
            this_batch["zp_pre_mask"] = self.zp_pre_vec_mask[zpi_s:zpi_e]
            this_batch["candi"] = self.candi_vec[ci_s:ci_e]
            this_batch["candi_mask"] = self.candi_vec_mask[ci_s:ci_e]
            this_batch["fl"] = self.ifl_vec[ci_s:ci_e]

            this_batch["start2end"] = start2end

            self.data_batch.append(this_batch)

    def generate_data(self,shuffle=False):
        if shuffle:
            random.shuffle(self.data_batch) 
        
        estimate_time = 0.0
        done_num = 0
        total_num = len(self.data_batch)

        for data in self.data_batch:
            start_time = timeit.default_timer()
            done_num += 1
            yield data

            end_time = timeit.default_timer()
            estimate_time += (end_time-start_time)
            EST = total_num*estimate_time/float(done_num)
            info = "Total use %.3f seconds for %d/%d -- EST:%f , Left:%f"%(end_time-start_time,done_num,total_num,EST,EST-estimate_time)
            sys.stderr.write(info+"\r")
        print >> sys.stderr

if __name__ == "__main__":
    train_generater = DataGnerater("train",nnargs["max_pair"])
    test_generater = DataGnerater("test",0)
