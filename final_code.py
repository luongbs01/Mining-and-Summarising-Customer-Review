import nltk
import json
import csv
from nltk.corpus import stopwords
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

def lemm_review(arr):
    tmp = []
    bit = []
    h, w, n = 0, 0, len(arr)
    for i in range(0, n):
        bit.append(0)
        m, w = len(arr[i]), 0
        for j in range(0, m):
            if w == 0:
                tmp.append([])
            tmp[h].append(lemma.lemmatize(str(arr[i][j][0])))
            w += 1
            bit[i] += 1
        if w >= 1:
            h += 1
    return tmp

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


def lemm(arr, type):
    n = len(arr)
    for i in range(0, n):
        m = len(arr[i])
        for j in range(0, m):
            arr[i][j] = lemma.lemmatize(arr[i][j], type)
    return arr


def convert1d(arr):
    ll = len(transaction)
    tmp = []
    for i in range(0, ll):
        rr = len(transaction[i])
        for j in range(0, rr):
            tmp.append(transaction[i][j])
    return tmp


def freqone(seed, arr, support):
    tmp = []
    for var in seed:
        if arr.count(var) >= support:
            tmp.append(var)
    return tmp


#############################################
'''TOKENIZATION AND NOUNS'''
fn = 'ec-cloud.json'
f = open('oneline/' + fn, encoding="utf8")
data = json.load(f)
name = data['name']
review = data['oneline']
review = review.lower()
pos_review = mult_token(review)
# print('pos_review', pos_review)

feature, featcnt = transaction(pos_review)
feature, featcnt = rem_stop_word(feature, featcnt)
feature = lemm(feature, 'n')
# print('feature', feature)
# print('featcnt', featcnt)
#############################################


#############################################
'''PROCESS FOR ADJECTIVES'''
adject, adjcnt = cntadj(pos_review)
adject, adjcnt = rem_stop_word(adject, adjcnt)
adject = lemm(adject, 'a')
# print("Adjectives : ", adject)
# print('adjcnt', adjcnt)

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
# print('adject_dict', adjective_dic)
# print('sentence_orient', sentence_orient)
#############################################

#############################################
'''APRIORI ALGO'''
transaction = feature
print('feature length', len(feature))
tmp = convert1d(transaction)
lstunq = set(tmp)
sp = 0
if len(feature) < 50: sp = 2
elif len(feature) >= 50 and len(feature) < 1000: sp = int(0.03 * len(feature)) + 1
else: sp = int(0.01 * len(feature))
print('support', sp)
frstfreq = freqone(lstunq, tmp, sp)


# Remove noun not a property
to_rm = []
f = open('noun_not_a_property.csv', 'r')
csv_read = csv.reader(f)
for row in csv_read:
    to_rm.append(row[0])
service = name.lower().split()
for word in service:
    to_rm.append(lemma.lemmatize(word))
for i in to_rm:
    try:
        frstfreq.remove(i)
    except:
        continue

print("first frequent : ", frstfreq)
# print(len(feature))
# print(len(featcnt))
##############################################

# Sentence Orientation

local_ft_orientation = {}
positive={}
negative={}
positive_sentence={}
negative_sentence={}
backl = 2
frontl = 6

review_after_lemm = lemm_review(pos_review)
for feature in frstfreq:
    # print(sentence)
    local_ft_orientation[feature] = 0
    positive[feature] = 0
    negative[feature] = 0
    positive_sentence[feature] = []
    negative_sentence[feature] = []
    for sentence in review_after_lemm:
        try:
            # print('feature in features:', feature)
            # print('sentence', sentence)
            ft_index = sentence.index(feature)
        except:
            # print('false')
            continue
        # go backwards 2 units or stop at stopw
        for i in range(1, backl+1):

            ind = ft_index-i

            if(ind < 0):
                break

            word = sentence[ind]
            if word in cachedstopwords:
                break

            if word in adjective_dic.keys():
                local_ft_orientation[feature] += adjective_dic[word]
                if adjective_dic[word] > 0: 
                    positive[feature] += 1 
                    positive_sentence.setdefault(feature, []).append(' '.join(sentence))
                    # positive_sentence[feature].append[sentence]
                elif adjective_dic[word] < 0: 
                    negative[feature] += 1
                    negative_sentence.setdefault(feature, []).append(' '.join(sentence))
                    # negative_sentence[feature].append[sentence]
                break

        # if feature in local_ft_orientation:
        #     continue

        # local_ft_orientation[feature] = 0
        # go forward 6 units , or stop at stopw

        for i in range(1, frontl+1):

            ind = ft_index+i

            if(ind >= len(sentence)):
                break

            word = sentence[ind]
            if word in cachedstopwords:
                break

            if word in adjective_dic.keys():
                local_ft_orientation[feature] += adjective_dic[word]
                if adjective_dic[word] > 0: 
                    positive[feature] += 1 
                    positive_sentence.setdefault(feature, []).append(' '.join(sentence))
                elif adjective_dic[word] < 0: 
                    negative[feature] += 1
                    negative_sentence.setdefault(feature, []).append(' '.join(sentence))
                break

output = {'name': name, 'features': []}
for i in local_ft_orientation.keys():
    if(positive[i] > 0 or negative[i] > 0):
        item = {'name': i, 'positive': {'count': positive[i], 'reviews': []}, 'negative': {'count': negative[i], 'reviews': []}}
        print(i, local_ft_orientation[i])
        print(i, positive[i])
        for sentence in positive_sentence[i]:
            # print(sentence)
            item['positive']['reviews'].append(sentence.replace(' .', '').replace('wa', 'was'))
        print(i, negative[i])
        for sentence in negative_sentence[i]:
            # print(sentence)
            item['negative']['reviews'].append(sentence.replace(' .', '').replace('wa', 'was'))
        print('\n')
        output['features'].append(item)

print("local_ft_orientation", local_ft_orientation)


file = open('output/test' + fn, 'w')
file.write(f'{json.dumps(output)}')
file.close()
