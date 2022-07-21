#! /usr/bin/python3
# -*- coding: utf-8 -*-

import nltk
import os
import sys
from nltk.corpus import stopwords
import numpy as np
import collections
from pred_opinion import adjective
lemma = nltk.WordNetLemmatizer()
cachedstopwords = stopwords.words("english")


def mult_token(review):
    final = []
    sent_text = nltk.sent_tokenize(review)
    # print('sent_text', sent_text)
    for sentence in sent_text:
        tokenized_text = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokenized_text)
        # print('tagged', tagged)
        final.append(tagged)
        # print('final', final)
    return final


def transaction(arr):
    tmp = []
    bit = []
    h, w, n = 0, 0, len(arr)
    for i in range(0, n):
        bit.append(0)
        m, w = len(arr[i]), 0
        for j in range(0, m):
            if arr[i][j][1] == "NN" or arr[i][j][1] == "NNS" or arr[i][j][1] == "NNP" or arr[i][j][1] == "NNPS":
                if w == 0:
                    tmp.append([])
                tmp[h].append(str(arr[i][j][0]))
                w += 1
                bit[i] += 1
        if w >= 1:
            h += 1
    return tmp, bit


def cntadj(arr):
    tmp = []
    bit = []
    h, w, n = 0, 0, len(arr)
    for i in range(0, n):
        bit.append(0)
        m, w = len(arr[i]), 0
        for j in range(0, m):
            if arr[i][j][1] == "JJ" or arr[i][j][1] == "JJS" or arr[i][j][1] == "JJR":
                if w == 0:
                    tmp.append([])
                tmp[h].append(str(arr[i][j][0]))
                w += 1
                bit[i] += 1
        if w >= 1:
            h += 1
    return tmp, bit


def rem_stop_word(arr, bit):
    tmp = []
    h, w, fl, n = 0, 0, 0, len(arr)
    for i in range(0, n):
        while bit[fl] == 0:
            fl += 1
        m, w = len(arr[i]), 0
        for j in range(0, m):
            if arr[i][j] not in cachedstopwords:
                if w == 0:
                    tmp.append([])
                tmp[h].append(str(arr[i][j]))
                w += 1
            else:
                bit[fl] -= 1
        if w >= 1:
            h += 1
        fl += 1
    return tmp, bit


def lemm(arr):
    n = len(arr)
    for i in range(0, n):
        m = len(arr[i])
        for j in range(0, m):
            arr[i][j] = lemma.lemmatize(arr[i][j])
    return arr


def convert1d(arr):
    ll = len(transaction)
    tmp = []
    for i in range(0, ll):
        rr = len(transaction[i])
        for j in range(0, rr):
            tmp.append(transaction[i][j])
    return tmp


def freqone(seed, arr):
    tmp = []
    for var in seed:
        if arr.count(var) >= support:
            tmp.append(var)
    return tmp


def createdct(arr):
    dct = {}
    dct2 = {}
    ll = len(arr)
    for i in range(ll):
        dct[arr[i]] = i+1
        dct2[i+1] = arr[i]
    return dct, dct2


def crtscndmat(i):
    mat = [[0 for x in range(i+1)] for y in range(i+1)]
    return mat


def freq2(rev, dct, dct2):
    tmp, ll = [], len(rev)
    ans = []
    ans.append([])
    pair = crtscndmat(len(dct))
    for i in range(ll):
        rr = len(rev[i])
        for j in range(rr):
            if rev[i][j] in dct:
                tmp.append(dct[rev[i][j]])
        tt = len(tmp)
        for y in range(tt-1):
            for z in range(y+1, tt):
                pair[tmp[y]][tmp[z]] += 1
                pair[tmp[z]][tmp[y]] += 1
        del tmp[:]
    ll, num = len(dct)+1, 0
    for i in range(1, ll):
        for j in range(i+1, ll):
            if pair[i][j] >= support:
                ans[num].append((dct2[i], dct2[j]))
                ans.append([])
                num += 1
    return ans


def usefuladj(feature, featcnt, adject, adjcnt, frstfreq):
    ll, rr, j = len(featcnt), len(feature), 0
    tmp = [0 for x in range(ll)]
    for i in range(rr):
        fl = 0
        for f in feature[i]:
            if f in frstfreq:
                fl = 1
                break
        if fl == 1:
            while featcnt[j] == 0:
                j += 1
            tmp[j] = fl
            j += 1
    for i in range(ll):
        tmp[i] = tmp[i]*adjcnt[i]
    # print(adjcnt)
    # print(tmp)
    return tmp


def foreachfeat(feature, featcnt, frstfreq):
    tmp = []
    tmp.append([])
    num, j = 0, 0
    rr = len(feature)
    for i in range(rr):
        fl = 0
        while featcnt[j] == 0:
            tmp.append([])
            num += 1
            j += 1
        for f in feature[i]:
            if f in frstfreq:
                tmp[num].append(f)
                fl = 1
        tmp.append([])
        num += 1
        if fl == 0:
            tmp.append([])
            num += 1
        j += 1
    if len(tmp[len(tmp)-1]) == 0:
        del tmp[len(tmp)-1]
    return tmp


#############################################
'''TOKENIZATION AND NOUNS'''
# review = "Amazon web services or AWS is the cutting edge public cloud provider and this is the best public cloud service I have ever seen. We are using AWS for a very long time and really like the overall features and output, using this cloud we can not run our web application as well the mobile application seamlessly. AWS provides many options related to the servers and instances, I can choose EC2, RDS, Lamda, Redshift, and many more, these all are great and works seamlessly. I can also opt for the on-demand servers which are costly but only I need to pay when I use these servers, on another hand, I have the option to go for the reserve servers which saves my cost but I need to make some volume and time commitment to the AWS. There are many other features like I can opt for the savings plan or spot instances which are really good. The integration of this platform is good but seamless, though I get great technical support from AWS all the time but the cost of business support and the premium support is very costly. The dashboard or the console is very easy to use and I can see all my usage and reports, I can see the recommendation to reduce the cost, and also I have all the visibility in this dashboard. The servers are auto-scaled up and down and this is great. AWS helps us to run seamlessly web application and provide great business services. hence my overall experience using AWS is very good. "
# review += "Amazon Web Services or AWS is the best public cloud at this recent time. We at my current organization use AWS for all our cloud-related needs and really like the overall flexibility of this public cloud platform. I can opt for the on-demand servers, I can go for the reserve-like pricing or simply I can go for the saving plans, AWS provides all the flexibility to its users. There is a feature in which I need to pay only for those servers which I am using and if I am not using that server I don't need to pay anything to them, this is really a great feature. The user interface is very good of this platform and I can see and view all the reports/usages pattern using the dashboard and even I can export the usage report using the dashboard. The technical support is amazing though there are very heavy price we need to pay for the business support but the service is very good. The cost of this public cloud is very high but I can reduce the cost using the reserve feature of EDP feature using these features I can get heavy discounts. AWS also has many business services and infrastructure service which is very easy to use and effective as well, I can use the S3 service for the emails and also servers for the emails, I can use this for the data transfer as well, this public cloud platform really works great and fast, the security of this cloud is amazing, I don't have to worry about the security of all our data. Hence my overall experience using this cloud is very good. "
# review += "We have started using Amazon web services from the year of 2021 July to create the Healthcare Product. Before July 2021, All our applications for PI was in On premise server right from Database, Online, Batch services and other Data ware house services. Healthcare customer was planning to leverage all the FWA applications to sell as services which implies this can be used by other small scale customer. In order to have time to market and better service capabilities, We have planned to go with Amazon web services to host of Spaces/Servers to EC2, Database to Dynamo DB from Mongo DB, Queue services to SQS, Triggers to API gate way and drive space to S3 Object store to store all our data/imaging repositories enablement. Currently our product is in Business testing with better approach of all services in to one shape with less time to market for integration/release and deployment. "
# review += "Amazon Web Service is by far the most reliable yet economical cloud platform that has a plethora of options for individuals, and enterprises alike. It is highly scalable and the modular architecture ensures that work doesn't suffer or stop because a particular service was not opted for at the beginning. All of this is backed by one of the most robust networks globally with dedicated servers and infrastructure to ensure maximum uptime, something that most modern businesses need in the web era. "

review = review.lower()
pos_review = mult_token(review)

feature, featcnt = transaction(pos_review)
feature, featcnt = rem_stop_word(feature, featcnt)
# feature = lemm(feature)

#############################################

#############################################
'''PROCESS FOR ADJECTIVES'''
adject, adjcnt = cntadj(pos_review)
adject, adjcnt = rem_stop_word(adject, adjcnt)
# adject = lemm(adject)
# print("Adjectives : ")
# print(adject, adjcnt)

seed_file = "seed_list.csv"

sentence_orient = []
adjective_dic = {}

for row in adject:
    cl = adjective(seed_file, row)
    cl.file_read()
    [pos, neg, adjmap] = cl.orientation()
    sentence_orient.append((pos, neg))
    cl.file_write()
    adjective_dic.update(adjmap)
#############################################

#############################################
'''APRIORI ALGO'''
transaction = feature
support = int((0.1)*len(feature))
tmp = convert1d(transaction)
lstunq = set(tmp)
frstfreq = freqone(lstunq, tmp)
dct, dct2 = createdct(frstfreq)
scndfreq = freq2(transaction, dct, dct2)

# print("first frequent : ", frstfreq)
# print("second frequent : ", scndfreq)
# print(scndfreq)
opin = usefuladj(feature, featcnt, adject, adjcnt, frstfreq)
feature_list = foreachfeat(feature, featcnt, frstfreq)
# print("frture list : ", feature_list)
# print(adjmap)

##############################################
'''
print(str(review))			#review entered by user
print(str(pos_review))		#par of speech(pos) tagging for the review for each word of each sentence
print(str(feature))			#obtaining the nouns for each sentence if it contains them
print(str(featcnt))			#obtaining frequency of nouns for each sentence

print(str(adject))			#obtaining the adjectives for each sentence if it contains them
print(str(adjcnt))			#obtaining frequency of adjectives for each sentence

print (transaction)   #nouns for each sentence
print(dct)            #dictionary (frequentfeature:key(1,2,..))
print(dct2)           #dictionary (key(1,2,..):frequentfeature)
print(frstfreq)       #list of frequent features
print(scndfreq)       #list of frequent features (2 words)
'''
# Sentence Orientation

local_ft_orientation = {}
backl = 2
frontl = 6
# print("adjective_dic : ",  adjective_dic)
stopw = ['and', ',', '.', 'or', '?', 'but']


for sentence, features in zip(pos_review, feature_list):
    sentence = [i[0] for i in sentence]
    # print(sentence)
    # print(features)
    for feature in features:
        # print(sentence)

        ft_index = sentence.index(feature)

        # go backwards 2 units or stop at stopw
        for i in range(1, backl+1):

            ind = ft_index-i

            if(ind < 0):
                break

            word = sentence[ind]
            if word in stopw:
                break

            if word in adjective_dic.keys():
                local_ft_orientation[feature] = adjective_dic[word]
                break

        if feature in local_ft_orientation:
            continue

        local_ft_orientation[feature] = 0
        # go forward 6 units , or stop at stopw

        for i in range(1, frontl+1):

            ind = ft_index+i

            if(ind >= len(sentence)):
                break

            word = sentence[ind]
            if word in stopw:
                break

            if word in adjective_dic.keys():
                local_ft_orientation[feature] = adjective_dic[word]
                break


print("feature list : ", feature_list)
for i in local_ft_orientation.keys():
    if(local_ft_orientation[i] != 0):
        print(i, local_ft_orientation[i])

# print("feature orientation : ", local_ft_orientation)

sentence_orientation = []

for index, op in enumerate(sentence_orient):
    net_orientation = (op[0]-op[1])
    if(net_orientation < 0):
        sentence_orientation.append(-1)
    elif(net_orientation > 0):
        sentence_orientation.append(1)
    else:
        for feature in feature_list[index]:
            # if feature in local_ft_orientaion:
            net_orientation += local_ft_orientation[feature]
        if(net_orientation < 0):
            sentence_orientation.append(-1)
        elif(net_orientation > 0):
            sentence_orientation.append(1)
        else:
            sentence_orientation.append(0)


print("sentence_orientation : ", sentence_orientation)

orientationSum = sum(sentence_orientation)
if orientationSum > 0:
    review_orientation = 1
elif orientationSum < 0:
    review_orientation = -1
else:
    review_orientation = 0

print("Review Orientation : ", review_orientation)
