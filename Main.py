import nltk
import math
import numpy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


lines = []
lines2 =[]
count = 0

# Opens text file and puts all the lines in a lines array
with open("text.txt",encoding = 'latin-1') as f:
    for line in f:
        lines.append(line)
        count = count +1
        
print(count)
b = "!@#$.,?-/_%^&*()+=\":;"

# Removes all the characters that are not needed
for line in lines:
    for char in b:
        line = line.replace(char,"")
    lines2.append(line)

lines = lines2
        
        
# The query that is used for tf-idf, the query is a simple one that
# uses typicle words for murders
query = ["murder","death","died","killed","murdered","killed","die","murderer"]
queryCount = []

# Query count is used for tf to hold the count the query words in a certain document
for a in range(len(query)):
    a = numpy.zeros(389)
    queryCount.append(a)


doculist = []
d2 =[]
count =0


# splits the text file into multiple documents, in this case:
# each document is 15 lines, no recursion of sentences (was not usefull)
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

# Collects the count of words
def wordCount(a,b):
    queryCount[a][b] += 1
    
    
check = False
qCount =0
docCounter =0

# tf : counts the words of the query
for doc in d2:
    number = 0
    result = doc.split()
    for word in result:
        qCount =0
        for w in query:
            w = w.lower()
            if word == w:
                wordCount(qCount, docCounter)
            qCount +=1
    docCounter +=1


    
        



#idf function using sklearn
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(doculist)
idf = vectorizer.idf_

feature_names = vectorizer.get_feature_names()


count = 0
score = 0
docWeight = []
# docWeight holds the weight of each document, the higher the weight the more "usefull"
# the document (not all info is found in these "usefull" documents), sometimes murder or
# death is not always mentioned, but usually in the area of these document a murder has
# happened

for a in range(len(doculist)):
    a = numpy.zeros(1)
    docWeight.append(a)

# Calculates the tf-idf scores for each document
for i in range(len(doculist)):
    for j in range(len(query)):
        score = score + (idf[vectorizer.vocabulary_.get(query[j])] * queryCount[j][i])
    docWeight[i] = score
    #Threshold placed by me (You can make it whatever you want)
    if score > 30:
        print(i) # prints the document number 
        print (docWeight[i]) # Score of document i
        print(doculist[i]) # put i in the doculist array, returns the text found
    score = 0;

'''
print(vectorizer.vocabulary_.get('murderess'))
print(idf[vectorizer.vocabulary_.get('murderess')])
print(feature_names[vectorizer.vocabulary_.get('murderess')])


text_file = open("Output.txt", "wb")
text_file.write(str(dict(zip(vectorizer.get_feature_names(),idf) )).encode('utf-8').strip())
text_file.close()
'''



    



    
        

