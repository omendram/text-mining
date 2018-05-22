import nltk
import math
import numpy
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from operator import itemgetter
import pprint
from nltk.corpus import wordnet
import pickle
from difflib import SequenceMatcher

printer = pprint.PrettyPrinter(indent=4)
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

        
        
# The query that is used for tf-idf, the query is a simple one that
# uses typicle words for murders
query = ["murder","death","dead","died","killed","murdered","die","murderer"]
queryCount = []

# Query count is used for tf to hold the count the query words in a certain document
for a in range(len(query)):
    a = numpy.zeros(389)
    queryCount.append(a)


doculist = []
doculist2 = []
d2 =[]
count =0


# splits the text file into multiple documents, in this case:
# each document is 15 lines, no recursion of sentences (was not usefull)
for j in range(259):
    document = ''
    document2 =''
    doc2 = ""
    for i in range(15):
        line2 = lines2[i+count] # lines without characters
        line = lines[i+count] # lines with character --> POS
        document2 += str(line2) 
        document += str(line)
        doc2 += line2
    doculist.append(document2) # document without characters
    doculist2.append(document) # documents with characters
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

docListsWithWeights = []

# Calculates the tf-idf scores for each document
for i in range(len(doculist)):
    for j in range(len(query)):
        score = score + (idf[vectorizer.vocabulary_.get(query[j])] * queryCount[j][i])
    docWeight[i] = score

    # Make a list of tuple, with doc and their weights, helps with sorting and chronological stuff
    docListsWithWeights.append((docWeight[i], doculist2[i], i)) 

    #Threshold placed by me (You can make it whatever you want)
    if score < 30 and score > 10:
        
        print(i) # prints the document number 
        print (docWeight[i]) # Score of document i
        print(doculist2[i]) # put i in the doculist array, returns the text found
    score = 0
    

#print(vectorizer.vocabulary_.get('murderess'))
#print(idf[vectorizer.vocabulary_.get('murderess')])
#print(feature_names[vectorizer.vocabulary_.get('murderess')])

#text_file = open("Output.txt", "wb")
#text_file.write(str(dict(zip(vectorizer.get_feature_names(),idf) )).encode('utf-8').strip())
#text_file.close()

print('\n ## Named Entities ## \n')

foundNamedEntities = []

# Checks whether a similar string already exists in the array
def check_similarity_metric(str, stringList):
    noMatchFound = True
    name = str
    for index, string in enumerate(stringList):
        if SequenceMatcher(None, name, string).ratio() > 0.8:
            noMatchFound = False
            break
    return noMatchFound, stringList

# Sort according to their scores
docListsWithWeights.sort(key=itemgetter(0), reverse=True)

# Show names in top 15 docs found.
for counter in range(15):
    detectedText = docListsWithWeights[counter][1]
    detectedText.lower()
    chunked_text = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(detectedText)))
    named_entities = []

    for chunk in chunked_text:
        if type(chunk) == nltk.Tree :
            temp_name = " ".join([word for word, position in chunk.leaves()])
            metric, temp_list = check_similarity_metric(temp_name, named_entities)
            if metric:
                named_entities.append(temp_name)
            else:
                named_entities = temp_list
    named_entities = list(set(named_entities))

    
    #print('People found in document no.', counter)
    #print(chunked_text)
    #print(named_entities)
    foundNamedEntities.append(named_entities)


titels = ["Dr.","Mr.","Miss","Mrs."]
entities = []
temp = ""

def namechecker(i,j,text):
    f = False
    entity = foundNamedEntities[i][j].split()
    result = text.split()
    for k in range(len(result)):
        for word in entity:
            if result[k] == word:
                if word == entity[0]:
                    for titel in titels:
                        if k-1 >= 0 and result[k-1] == titel:
                            found = titel+" " +foundNamedEntities[i][j]
                            entity = found.split()
                            f = True
                            break
                            
                if(k+1 < len(result)):
                    if (result[k+1] == "said" or result[k+1] == "said:" ):
                        if f == True:
                            entities.append(found)
                            f = False
                        else:
                            entities.append(foundNamedEntities[i][j])
    

                    
        
    
for text in doculist2:
    for i in range(len(foundNamedEntities)):
        for j in range(len(foundNamedEntities[i])):
                if(j + 1 < len(foundNamedEntities[i])):
                    namechecker(i,j,text)
    

entities = list(set(entities))
print(entities)
            








#TODO Sort out false values, sentiment, other information
all_named_entities = []

# This gives a lot of information, entities include persons, and sometimes non-person entities as well.
# Need to figure out a good way to filter them out.
for i in doculist2:
    chunked_text = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(i)))
    #NLTK Tools
    for chunk in chunked_text:
        if type(chunk) == nltk.Tree and (chunk.label() == 'PERSON') :
                name = " ".join([word for word, position in chunk.leaves()])
                metric, temp_list = check_similarity_metric(name, all_named_entities)
                if metric:
                    all_named_entities.append(name)
                else:
                    all_named_entities = temp_list

#print('Total entites found: ', len(all_named_entities))
#print(all_named_entities)

#print(len(all_named_entities))foundNamedEntities[i][j]
#for names in all_named_entities:
#    print(names)
#printer.pprint(named_entities)
