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
from torch.optim import lr_scheduler

from conf import *
import utils
from data_generater import *
from net import *

print >> sys.stderr, "PID", os.getpid()

torch.cuda.set_device(args.gpu)

def net_copy(net,copy_from_net):
    mcp = list(net.parameters())
    mp = list(copy_from_net.parameters())
    n = len(mcp)
    for i in range(0, n):
        mcp[i].data[:] = mp[i].data[:]

def get_predict(data,t):
    predict = []
    for result,output in data:
        max_index = -1
        for i in range(len(output)):
            if output[i] > t:
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
    res_t = [0.05,0.1,0.15,0.2,0.25,0.3]
    best_result = {}
    best_result["hits"] = 0

    # best first
    predict = get_predict_max(data)
    result = evaluate(predict)
    if result["hits"] > best_result["hits"]:
        best_result = result
        best_result["t"] = -1

    # nearest first
    for t in res_t:
        predict = get_predict(data,t)
        result = evaluate(predict)
        if result["hits"] > best_result["hits"]:
            best_result = result
            best_result["t"] = t

    print >> sys.stderr, "Hits",best_result["hits"]
    print "Hits",best_result["hits"]
    print "Best thresh",best_result["t"]
    print "R",best_result["r"],"P",best_result["p"],"F",best_result["f"]
    print
    return best_result

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

MAX = 2

def main():
    train_generater = DataGnerater("train",nnargs["batch_size"])
    test_generater = DataGnerater("test",256)

    embedding_matrix = numpy.load(args.data + "embedding.npy")
    print "Building torch model"

    model = Network(nnargs["embedding_size"],nnargs["embedding_dimention"],embedding_matrix,nnargs["hidden_dimention"],1,nnargs["attention"]).cuda()

    this_lr = nnargs["lr"]

    optimizer = optim.Adam(model.parameters(), lr=this_lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    best_result = {}
    best_result["hits"] = 0
    best_model = None
     
    for echo in range(nnargs["epoch"]):
        cost = 0.0
        scheduler.step()
        print >> sys.stderr, "Begin epoch",echo

        for data in train_generater.generate_data(shuffle=True):

            output = model.forward(data,dropout=nnargs["dropout"])

            target = autograd.Variable(torch.from_numpy(data["result"]).type(torch.cuda.FloatTensor))
            loss = []
            for s,e in data["start2end"]:
                if s == e:
                    continue
                this_loss = -1.0*torch.log(torch.sum(target[s:e]*F.softmax(output[s:e]))+1e-10)+\
                        -1.0*torch.log(torch.sum((1-target[s:e])*(1-F.softmax(output[s:e])))+1e-10)
                loss.append(this_loss)
            
            optimizer.zero_grad()
            loss = torch.sum(torch.cat(loss))
            cost += loss.data[0]
            loss.backward()
            optimizer.step()
        print >> sys.stderr, "End epoch",echo,"Cost:", cost
    
        predict = []
        for data in test_generater.generate_data():
            output = model.forward(data)

            for s,e in data["start2end"]:
                if s == e:
                    continue
                output_softmax = F.softmax(output[s:e]).data.cpu().numpy()
                predict.append((data["result"][s:e],output_softmax))

        print "Result for epoch",echo 
        result = get_evaluate(predict)
        if result["hits"] > best_result["hits"]:
            best_result = result
            best_result["epoch"] = echo 
            best_model = model 
            torch.save(best_model, "./model/model.pretrain.best")
        sys.stdout.flush()

    print "Best Result on epoch", best_result["epoch"]
    print "Hits",best_result["hits"]
    print "R",best_result["r"],"P",best_result["p"],"F",best_result["f"]
 
if __name__ == "__main__":
    main()
