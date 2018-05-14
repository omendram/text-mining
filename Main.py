import nltk
import math
import numpy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


lines = []
lines2 =[]
count = 0
with open("text.txt",encoding = 'latin-1') as f:
    for line in f:
        lines.append(line)
        count = count +1
        
print(count)
b = "!@#$.,?-/_%^&*()+=\":;"
for line in lines:
    for char in b:
        line = line.replace(char,"")
    lines2.append(line)

lines = lines2
        
        

query = ["murder","death","died","killed","murdered","killed","die","murderer","shot","poisoned","body"]
queryCount = []

for a in range(len(query)):
    a = numpy.zeros(389)
    queryCount.append(a)


doculist = []
d2 =[]
count =0

'''
for j in range(100):
    document = ''
    for i in range(39):
        line = lines[i+count]
        document += str(line)
    doculist.append(document)
    count = count+39
'''

for j in range(259):
    document = ''
    doc2 = ""
    for i in range(15):
        line = lines[i+count]
        document += str(line)
        doc2 += line
    doculist.append(document)
    d2.append(doc2)
    count += 15

def wordCount(a,b):
    queryCount[a][b] += 1
    
    
check = False
qCount =0
docCounter =0
for doc in d2:
    number = 0
    result = doc.split()
    for word in result:
        qCount =0
        for w in query:
            w = w.lower()
            if word == w:
                wordCount(qCount, docCounter)
                if word == "died" and docCounter > 500:
                   check = True
            qCount +=1
  
        if check == True:
            #print(queryCount[0][779-13])
            check = False
            
        
        
    #if number > 5:
        #print(number)
        #print(doc)
        #print(docCounter)
        #number = 0
        
    docCounter +=1


    
        

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(doculist)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(doculist)
idf = vectorizer.idf_

feature_names = vectorizer.get_feature_names()


count = 0
score = 0
docWeight = []
for a in range(len(doculist)):
    a = numpy.zeros(1)
    docWeight.append(a)

for i in range(len(doculist)):
    for j in range(len(query)):
        score = score + (idf[vectorizer.vocabulary_.get(query[j])] * queryCount[j][i])
    docWeight[i] = score
    if score > 30:
        print(i)
        print (score)
        print(doculist[i])
    score = 0;


print(vectorizer.vocabulary_.get('murderess'))
print(idf[vectorizer.vocabulary_.get('murderess')])
print(feature_names[vectorizer.vocabulary_.get('murderess')])


text_file = open("Output.txt", "wb")
text_file.write(str(dict(zip(vectorizer.get_feature_names(),idf) )).encode('utf-8').strip())
text_file.close()




    



    
        

