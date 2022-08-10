import csv
import nltk
lemma = nltk.WordNetLemmatizer()

list = []
f = open('noun_not_a_property.csv', 'r')
csv_read = csv.reader(f)
for row in csv_read:
    list.append(row[0])
service = 'Alibaba Cloud - International'.lower().split()
for word in service:
    list.append(lemma.lemmatize(word))
print(list)