#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy
import random
import cPickle
sys.setrecursionlimit(1000000)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.transforms as T
import torch.optim as optim

from conf import *
import utils
from data_generater import *
from net import *

def get_predict(data,t):
    predict = []
    thresh = t
    for result,output in data:
        max_index = -1
        dra_t = 1.0/len(output)
        if t == -1:
            thresh = dra_t
        for i in range(len(output)):
            if output[i] >= thresh:
                max_index = i
        predict.append(result[max_index])
    return predict

def get_predict_max(data):
    predict = []
    for result,output in data:
        max_index = -1
        max_pro = 0.0
        for i in range(len(output)):
            if output[i] > max_pro:
                max_index = i
                max_pro = output[i]
        predict.append(result[max_index])
    return predict
 
def get_evaluate(data):
    best_result = {}
    best_result["hits"] = 0

    # best first
    predict = get_predict_max(data)
    result = evaluate(predict)
    best_result = result

    print >> sys.stderr, "Best Hits",best_result["hits"]
    print "Best Hits",best_result["hits"]
    print "R",best_result["r"],"P",best_result["p"],"F",best_result["f"]

    return best_result

    best_result["hits"] = 0
    res_t = [-1]
    # nearest first
    for t in res_t:
        predict = get_predict(data,t)
        result = evaluate(predict)
        if result["hits"] > best_result["hits"]:
            best_result = result
            best_result["t"] = t 

    print >> sys.stderr, "Nearest Hits",best_result["hits"]
    print "Nearest Hits",best_result["hits"],"thresh",best_result["t"]
    print "R",best_result["r"],"P",best_result["p"],"F",best_result["f"]

def evaluate(predict):
    result = {}
    result["hits"] = sum(predict)
    p = sum(predict)/float(len(predict))
    r = sum(predict)/1713.0
    f = 0.0 if (p == 0 or r == 0) else (2.0/(1.0/p+1.0/r))
    result["r"] = r
    result["p"] = p
    result["f"] = f
    return result

def get_performance(test_generater,model):
    predict = []
    HITS = 0
    for data in test_generater.generate_data():
        #zp
        zp_reindex = autograd.Variable(torch.from_numpy(data["zp_reindex"]).type(torch.cuda.LongTensor))
        zp_pre = autograd.Variable(torch.from_numpy(data["zp_pre"]).type(torch.cuda.LongTensor))
        zp_pre_mask = autograd.Variable(torch.from_numpy(data["zp_pre_mask"]).type(torch.cuda.FloatTensor))
        zp_post = autograd.Variable(torch.from_numpy(data["zp_post"]).type(torch.cuda.LongTensor))
        zp_post_mask = autograd.Variable(torch.from_numpy(data["zp_post_mask"]).type(torch.cuda.FloatTensor))
        #np
        candi_reindex = autograd.Variable(torch.from_numpy(data["candi_reindex"]).type(torch.cuda.LongTensor))
        candi = autograd.Variable(torch.from_numpy(data["candi"]).type(torch.cuda.LongTensor))
        candi_mask = autograd.Variable(torch.from_numpy(data["candi_mask"]).type(torch.cuda.FloatTensor))
        
        feature = autograd.Variable(torch.from_numpy(data["fl"]).type(torch.cuda.FloatTensor))

        zp_pre = torch.transpose(zp_pre,0,1)
        mask_zp_pre = torch.transpose(zp_pre_mask,0,1)
        hidden_zp_pre = model.initHidden()
        hiddens_zp_pre = []
        for i in range(len(mask_zp_pre)):
            hidden_zp_pre = model.forward_zp_pre(zp_pre[i],hidden_zp_pre,dropout=nnargs["dropout"])*torch.transpose(mask_zp_pre[i:i+1],0,1)
            hiddens_zp_pre.append(hidden_zp_pre)
        hiddens_zp_pre = torch.cat(hiddens_zp_pre,1)
        hiddens_zp_pre = hiddens_zp_pre.view(-1,len(mask_zp_pre),nnargs["hidden_dimention"])
        pre_A = model.get_attention_pre(hiddens_zp_pre)
        pre_A = F.softmax(pre_A,1)
 
        average_results_pre = torch.matmul(torch.transpose(hiddens_zp_pre,1,2),pre_A)
        zp_pre_attention = torch.sum(average_results_pre,2)
        zp_pre_representation = zp_pre_attention[zp_reindex]
    
        zp_post = torch.transpose(zp_post,0,1)
        mask_zp_post = torch.transpose(zp_post_mask,0,1)
        hidden_zp_post = model.initHidden()
        hiddens_zp_post = []
        for i in range(len(mask_zp_post)):
            hidden_zp_post = model.forward_zp_post(zp_post[i],hidden_zp_post,dropout=nnargs["dropout"])*torch.transpose(mask_zp_post[i:i+1],0,1)
            hiddens_zp_post.append(hidden_zp_post)
        hiddens_zp_post = torch.cat(hiddens_zp_post,1)
        hiddens_zp_post = hiddens_zp_post.view(-1,len(mask_zp_post),nnargs["hidden_dimention"])
        post_A = model.get_attention_post(hiddens_zp_post)
        post_A = F.softmax(post_A,1)
 
        average_results_post = torch.matmul(torch.transpose(hiddens_zp_post,1,2),post_A)
        zp_post_attention = torch.sum(average_results_post,2)
        zp_post_representation = zp_post_attention[zp_reindex]
 
        candi = torch.transpose(candi,0,1)
        mask_candi = torch.transpose(candi_mask,0,1)
        hidden_candi = model.initHidden()
        for i in range(len(mask_candi)):
            hidden_candi = model.forward_np(candi[i],hidden_candi)*torch.transpose(mask_candi[i:i+1],0,1)
        candi_representation = hidden_candi[candi_reindex]

        output = model.generate_score(zp_pre_representation,zp_post_representation,candi_representation,feature)
        output = torch.squeeze(output)

        for s,e in data["start2end"]:
            if s == e:
                continue
            output_softmax = F.softmax(output[s:e]).data.cpu().numpy()
            predict.append((data["result"][s:e],output_softmax))

    br = get_evaluate(predict)
    print
    sys.stdout.flush()
    return br

def net_copy(net,copy_from_net):
    mcp = list(net.parameters())
    mp = list(copy_from_net.parameters())
    n = min(len(mcp),len(mp))
    for i in range(0, n): 
        mcp[i].data[:] = mp[i].data[:]

torch.cuda.set_device(args.gpu)
def main():
    test_generater = DataGnerater("test",256)
    embedding_matrix = numpy.load(args.data + "embedding.npy")
    print "Building torch model"

    model = Network(nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix,nnargs["hidden_dimention"],1).cuda()
    best_model_ = torch.load("./model/model.pretrain.best")
    net_copy(model,best_model_)
    
    get_performance(test_generater,model)


if __name__ == "__main__":
    main()
