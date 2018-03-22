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
from buildTree import get_info_from_file
from buildTree import get_info_from_file_system
import utils
import get_feature
#import word2vec

random.seed(args.random_seed)

import cPickle
sys.setrecursionlimit(1000000)

MAX = 2

def get_sentence(zp_sentence_index,zp_index,nodes_info):
    #返回只包含zp_index位置的ZP的句子(去掉其他ZP)
    nl,wl = nodes_info[zp_sentence_index]
    return_words = []
    for i in range(len(wl)):
        this_word = wl[i].word
        if i == zp_index:
            return_words.append("**pro**")
        else:
            if not (this_word == "*pro*"): 
                return_words.append(this_word)
    return " ".join(return_words)

def get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result):
    nl,wl = nodes_info[candi_sentence_index]
    candi_word = []
    for i in range(candi_begin,candi_end+1):
        candi_word.append(wl[i].word)
    candi_word = "_".join(candi_word)

    candi_info = [str(res_result),candi_word]
    return candi_info

def setup():
    utils.mkdir(args.data)
    utils.mkdir(args.data+"train/")
    utils.mkdir(args.data+"train_reduced/")
    utils.mkdir(args.data+"test/")
    utils.mkdir(args.data+"test_reduced/")


def embedding_filtering(path):
    embedding_file = args.embedding_data
    word_dict = {}
    word_need = set()
    f = open(embedding_file)
    while True:
        line = f.readline()
        if not line:break
        line = line.strip().split(" ")
        word = line[0]
        vector = line[1:]
        vec = [float(item) for item in vector]
        word_dict[word] = vec

    paths = utils.get_file_name(path,[])

    for p in paths:
        if p.strip().endswith("DS_Store"):continue
        file_name = p.strip()
        if file_name.endswith("onf"):
            info = "Read File : %s"%file_name
            sys.stderr.write(info+"\r")

            zps,azps,candi,nodes_info = get_info_from_file(file_name,2)
            for k in nodes_info:
                nl,wl = nodes_info[k]
                for n in nl: 
                    if n.word.find("*") < 0:
                        word = n.word.strip()
                        word_need.add(word)
    print >> sys.stderr

    word_list = []
    vecs = []
    vecs.append(nnargs["embedding_dimention"]*[0.0])
    for word in word_need:
        if word in word_dict:
            vecs.append(word_dict[word]) 
            word_list.append(word)
    print >> sys.stderr, "Total words", len(vecs)
    vecs = numpy.array(vecs)

    embedding_path = args.data + "embedding.npy"
    numpy.save(embedding_path,vecs)

    f = open(args.data + "word","w")
    f.write("None\n")
    for word in word_list:
        f.write(word+"\n")

def list_vectorize(wl,words):
    il = []
    for w in wl:
        word = w.word
        if word in words:
            index = words.index(word)
        else:
            index = 0
        il.append(index) 
    return il

def generate_vector(path):
    embedding = numpy.load(args.data + "embedding.npy") 
    words = [w.strip() for w in open(args.data + "word").readlines()]

    paths = utils.get_file_name(path,[])

    total_sentence_num = 0
    vectorized_sentences = []
    zp_info = []

    HcP = [] 
    #f = open(args.data+"hcp")
    #while True:
    #    line = f.readline()
    #    if not line:break
    #    line = line.strip()
    #    HcP.append(line)
    #f.close()

    startt = timeit.default_timer()
    done_num = 0
    for p in paths:
        if p.strip().endswith("DS_Store"):continue
        done_num += 1
        file_name = p.strip()
        if file_name.endswith("onf"):
            info = "Read File : %s, %d/%d"%(file_name,done_num,len(paths))
            sys.stderr.write(info+"\r")

            if args.reduced == 1 and done_num >= 3:break

            zps,azps,candi,nodes_info = get_info_from_file(file_name,2)
            anaphorics = []
            ana_zps = []
            for (zp_sentence_index,zp_index,antecedents,coref_id) in azps:
                for (candi_sentence_index,begin_word_index,end_word_index,coref_id) in antecedents:
                    anaphorics.append((zp_sentence_index,zp_index,candi_sentence_index,begin_word_index,end_word_index))
                    ana_zps.append((zp_sentence_index,zp_index))

            si2reali = {}
            for k in nodes_info:
                nl,wl = nodes_info[k]
                vectorize_words = list_vectorize(wl,words)
                vectorized_sentences.append(vectorize_words)
                si2reali[k] = total_sentence_num
                total_sentence_num += 1

            for (sentence_index,zp_index) in zps:
                ana = 0
                if (sentence_index,zp_index) in ana_zps:
                    ana = 1
                index_in_file = si2reali[sentence_index]
                zp = (index_in_file,sentence_index,zp_index,ana)
                #zp_info.append(zp)
                zp_nl,zp_wl = nodes_info[sentence_index]

                candi_info = []
                if ana == 1:
                    for ci in range(max(0,sentence_index-2),sentence_index+1):
                        candi_sentence_index = ci  
                        candi_nl,candi_wl = nodes_info[candi_sentence_index]

                        for (candi_begin,candi_end) in candi[candi_sentence_index]:
                            if ci == sentence_index and candi_end > zp_index:
                                continue
                            res_result = 0
                            if (sentence_index,zp_index,candi_sentence_index,candi_begin,candi_end) in anaphorics:
                                res_result = 1
                            candi_index_in_file = si2reali[candi_sentence_index]

                            ifl = get_feature.get_res_feature_NN_new((sentence_index,zp_index),(candi_sentence_index,candi_begin,candi_end),zp_wl,candi_wl)
                            #ifl = get_feature.get_res_feature_NN((sentence_index,zp_index),(candi_sentence_index,candi_begin,candi_end),zp_wl,candi_wl,[],[],HcP)

                            candidate = (candi_index_in_file,candi_sentence_index,candi_begin,candi_end,res_result,ifl)
                            candi_info.append(candidate)
                zp_info.append((zp,candi_info))

    endt = timeit.default_timer()
    print >> sys.stderr
    print >> sys.stderr, "Total use %.3f seconds for Data Generating"%(endt-startt)
    vectorized_sentences = numpy.array(vectorized_sentences)
    return zp_info,vectorized_sentences

def generate_vector_data():

    DATA = args.raw_data

    train_data_path = args.data + "train/"
    test_data_path = args.data + "test/"
    if args.reduced == 1:
        train_data_path = args.data + "train_reduced/"
        test_data_path = args.data + "test_reduced/"

    train_zp_info, train_vectorized_sentences = generate_vector(DATA+"train/")
    test_zp_info, test_vectorized_sentences = generate_vector(DATA+"test/")

    train_vec_path = train_data_path + "sen.npy"
    numpy.save(train_vec_path,train_vectorized_sentences)
    save_f = file(train_data_path + "zp_info", 'wb')
    cPickle.dump(train_zp_info, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
    save_f.close()

    test_vec_path = test_data_path + "sen.npy"
    numpy.save(test_vec_path,test_vectorized_sentences)
    save_f = file(test_data_path + "zp_info", 'wb')
    cPickle.dump(test_zp_info, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
    save_f.close()

def generate_input_data():
     
    DATA = args.raw_data

    train_data_path = args.data + "train/"
    test_data_path = args.data + "test/"
    if args.reduced == 1:
        train_data_path = args.data + "train_reduced/"
        test_data_path = args.data + "test_reduced/"

    generate_vec(train_data_path)
    generate_vec(test_data_path)

def generate_vec(data_path):

    zp_candi_target = []
    zp_vec_index = 0
    candi_vec_index = 0

    zp_prefixs = []
    zp_prefixs_mask = []
    zp_postfixs = []
    zp_postfixs_mask = []
    candi_vecs = []
    candi_vecs_mask = []
    ifl_vecs = []
    
    read_f = file(data_path + "zp_info","rb")
    zp_info_test = cPickle.load(read_f)
    read_f.close()

    vectorized_sentences = numpy.load(data_path + "sen.npy")
    for zp,candi_info in zp_info_test:
        index_in_file,sentence_index,zp_index,ana = zp        
        if ana == 1:
            word_embedding_indexs = vectorized_sentences[index_in_file]
            max_index = len(word_embedding_indexs)

            prefix = word_embedding_indexs[max(0,zp_index-10):zp_index]
            prefix_mask = (10-len(prefix))*[0] + len(prefix)*[1]
            prefix = (10-len(prefix))*[0] + prefix

            zp_prefixs.append(prefix)
            zp_prefixs_mask.append(prefix_mask)

            postfix = word_embedding_indexs[zp_index+1:min(zp_index+11,max_index)]
            postfix_mask = (len(postfix)*[1] + (10-len(postfix))*[0])[::-1]
            postfix = (postfix + (10-len(postfix))*[0])[::-1]
    
            zp_postfixs.append(postfix)
            zp_postfixs_mask.append(postfix_mask)
            
            candi_vec_index_inside = []
            for candi_index_in_file,candi_sentence_index,candi_begin,candi_end,res_result,ifl in candi_info:
                candi_word_embedding_indexs = vectorized_sentences[candi_index_in_file] 
                candi_vec = candi_word_embedding_indexs[candi_begin:candi_end+1]
                if len(candi_vec) >= 8:
                    candi_vec = candi_vec[-8:]
                candi_mask = (8-len(candi_vec))*[0] + len(candi_vec)*[1]
                candi_vec = (8-len(candi_vec))*[0] + candi_vec 

                candi_vecs.append(candi_vec)
                candi_vecs_mask.append(candi_mask)

                ifl_vecs.append(ifl)
                
                candi_vec_index_inside.append((candi_vec_index,res_result))

                candi_vec_index += 1

            zp_candi_target.append((zp_vec_index,candi_vec_index_inside)) 

            zp_vec_index += 1
    # save                
    #zp_candi_target: (zp_index,[(candi_index,res),(...)])
    save_f = file(data_path + "zp_candi_pair_info", 'wb')
    cPickle.dump(zp_candi_target, save_f, protocol=cPickle.HIGHEST_PROTOCOL)
    save_f.close()
    
    zp_prefixs = numpy.array(zp_prefixs,dtype='int32')
    numpy.save(data_path+"zp_pre.npy",zp_prefixs)
    zp_prefixs_mask = numpy.array(zp_prefixs_mask,dtype='int32')
    numpy.save(data_path+"zp_pre_mask.npy",zp_prefixs_mask)
    zp_postfixs = numpy.array(zp_postfixs,dtype='int32')
    numpy.save(data_path+"zp_post.npy",zp_postfixs)
    zp_postfixs_mask = numpy.array(zp_postfixs_mask,dtype='int32')
    numpy.save(data_path+"zp_post_mask.npy",zp_postfixs_mask)
    candi_vecs = numpy.array(candi_vecs,dtype='int32')
    numpy.save(data_path+"candi_vec.npy",candi_vecs)
    candi_vecs_mask = numpy.array(candi_vecs_mask,dtype='int32')
    numpy.save(data_path+"candi_vec_mask.npy",candi_vecs_mask)

    assert len(ifl_vecs) == len(candi_vecs)

    ifl_vecs = numpy.array(ifl_vecs,dtype='float')
    numpy.save(data_path+"ifl_vec.npy",ifl_vecs)
  

if __name__ == "__main__":
    setup()
    # get filtered embedding file
    #embedding_filtering(args.raw_data)
    generate_vector_data()
    generate_input_data()
