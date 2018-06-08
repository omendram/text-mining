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
from nltk.corpus import wordnet
from textblob.classifiers import NaiveBayesClassifier
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from nltk.grammar import DependencyGrammar
import re
import gensim
from wordcloud import  WordCloud



printer = pprint.PrettyPrinter(indent=4)
lines = []
lines2 =[]
count = 0

# Opens text file and puts all the lines in a lines array
with open("text.txt",encoding = 'latin-1') as f:
    for line in f:
        lines.append(line)
        count = count +1


# Removes all the characters that are not needed        
b = "!@#$.,?-/_%^&*()+=\":;"
for line in lines:
    for char in b:
        line = line.replace(char,"")
    lines2.append(line)
        
        
# The query that is used for tf-idf
query = ["murder","death","dead","died","killed","murdered","die","murderer"]


# Query count holds the term frequencies for each word in the query

queryCount = []
for a in range(len(query)):
    a = numpy.zeros(259)
    queryCount.append(a)


doculist = []
doculist2 = []
d2 =[]
count =0


# splits the text file into multiple documents, in this case:
# each document is 15 lines
# We created 2 collections, each have the same subsections of the novel.
# However, one has the characters removed, the other keeps them.

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

# Collects the amount of words in a document
def wordCount(a,b):
    queryCount[a][b] += 1
    
    
check = False
qCount =0
docCounter =0

# tf : counts the frequencies of the query words in each document
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
# the document 

for a in range(len(doculist)):
    a = numpy.zeros(1)
    docWeight.append(a)

docListsWithWeights = []

# Calculates the tf-idf scores for each document
for i in range(len(doculist)):
    for j in range(len(query)):
        score = score + (idf[vectorizer.vocabulary_.get(query[j])] * queryCount[j][i])
    docWeight[i] = score
    docListsWithWeights.append((docWeight[i], doculist2[i], i)) 
    if score < 30 and score > 10:
        continue
    score = 0
    

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
file = open('docs.txt', 'a')

# Show names in top 15 docs found.
for counter in range(15):
    detectedText = docListsWithWeights[counter][1]
    file.write(detectedText)
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
    foundNamedEntities.append(named_entities)


titels = ["Dr.","Mr.","Mrs."]
entities = []
temp = ""
filterList = ["said","ate","drank","said:","thought","cried","slept","walked","ran","tottered"];

# Checks wether the entity is human and it does anything in the story
def namechecker(i,j,textNr):
    text = doculist2[textNr]
    f = False
    tmp = ""
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
                    if (result[k+1] in filterList):
                        if f == True:
                            entities.append(found)
                            f = False
                            
                        else:
                            entities.append(foundNamedEntities[i][j])
    


# For all the entities found, check if they are human and if they are relevant    
for k in range(len(doculist2)):
    for i in range(len(foundNamedEntities)):
        for j in range(len(foundNamedEntities[i])):
                if(j + 1 < len(foundNamedEntities[i])):
                    namechecker(i,j,k)
    


#Returns a list with the names in order of last thing done
entityOrder1 = []
for entity in entities:
    if entity not in entityOrder1:
        entityOrder1.insert(0,entity)
    else:
        entityOrder1 = list(filter(lambda a: a != entity,entityOrder1))
        entityOrder1.insert(0,entity)
        

#Clusters the entities together by similarity in name.
nameslist = []
tmp = []
for word in entityOrder1:
    if word in tmp:
        continue
    words = word.split()
    tmp = []
    for x in words:
        for y in entityOrder1:
            if x in y and x not in titels and x != "Miss":
                if("Mrs." in word):
                    tmp.append(word)
                    break
                if("Mrs." in y and "Mrs." not in x):
                    break
                tmp.append(y)
            tmp = list(set(tmp))
    nameslist.append(tmp)

nameslistTemp = []
tmplist = []
x = 0
for i in range(len(nameslist)):
    if i in tmplist:
        continue
    for name in nameslist[i]:
        name = name.split()
        for word in name:
            y = 0
            for j in range(len(nameslist)):
                if nameslist[i] == nameslist[j] and x != y:
                    tmplist.append(j)
                for word2 in nameslist[j]:
                    if "Mrs." in name or "Mrs." in word2:
                        break

                    if word in word2 and word2 not in nameslist[i] and word not in titels and word != "Miss" and nameslist[i] != nameslist[j]:
                        nameslist[i].append(word2)
                        tmplist.append(j)
            y = y +1
    nameslistTemp.append(nameslist[i])
    x = x + 1
    
nameslist = nameslistTemp



######### From here add comments because not sure whats happening
def getPOS_Tags(sentences):
    sentences = nltk.sent_tokenize(sentences)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    return sentences


sentences = getPOS_Tags(" ".join(lines))

grammar = "PEOPLE: {<NNP>+<VBD>}"
cp = nltk.RegexpParser(grammar)
result_pos = []

for sentence in sentences:
    result = cp.parse(sentence)

    if (result.label() == 'PEOPLE'):
        result_pos.append(result)
    else:
        for item in result:
            if type(item) == nltk.Tree and item.label() == 'PEOPLE':
                result_pos.append(item)

# Sort documents according to tfidf scores
docListsWithWeights.sort(key=itemgetter(0), reverse=True)
                          
train = []

# Training Naive Bayes Classifier
with open("murdertrain.txt",encoding = 'latin-1') as f:
    for line in f:
        train.append((line.strip(),'murder'))

with open("pos-train.txt",encoding = 'latin-1') as f:
    for line in f:
        train.append((line.strip(),'notMurder'))

cl = NaiveBayesClassifier(train)
counter = 0
flat_names_list = [item for sublist in nameslist for item in sublist]

# check if meanining-ful sentences cover entities
def checkEntities(sentence):
    exists = False

    for name in flat_names_list:
        if sentence.find(name) >= 0:
            exists = True

    return exists

# Bayes Classifier on relevant document sentences
all_murders = []
for doc_found in docListsWithWeights:
    if doc_found[0] > 7.0:            
        sentences = doc_found[1]
        sentences = sentences.replace('Mrs.', 'Mrs')
        sentences = sentences.replace('Mr.', 'Mrs')
        sentences = sentences.replace('Dr.', 'Dr')
        counter += 1
        sentences =  sentences.split('.')
        

        for sentence in sentences:
            if (cl.classify(sentence) == 'murder'):
                if checkEntities(sentence):
                    all_murders.append(sentence)

tagged_results = []

# APPLY POS on meaning-ful sentences
for no, murder in enumerate(all_murders):
    symbols = '!@#$.?-/_%^&*()+=\":'

    for char in symbols:
        murder = murder.replace(char,"")

    for salutation in ['Mr ', 'Mrs ', 'Miss ', 'Dr ']:
        murder = murder.replace(salutation, salutation.strip())
    
    if len(murder)> 4:
        tags = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(murder)))    
        tagged_results.append(tags)



# resolve co-reference
def splitEnds(inputStr):
    sentence = inputStr.split()
    sentenceSplits = []
    answer = ''

    for idx, word in enumerate(sentence):
        answer = answer + ' ' + word

        if idx > 1:
            if word == 'and' or word[-1] == ',' or word == ',' or idx == len(sentence)-1:
                if len(answer.replace('and', '').strip()) > 2:
                    sentenceSplits.append(answer)
                answer = ''

    return sentenceSplits;


# Resolve co-reference
def constructResults(result):
    data = {}
    currentKey = ''
    str_refs = ''
    result = result.leaves()
    entityLabels = ['NNP', 'NNS']

    for i in range(len(result)):
        items = result[i]

        if items[1] in entityLabels:
            if i > 1 and result[i-1] in entityLabels:
                currentKey =  currentKey + result[i-1]
                continue
            else:
                currentKey = items[0]
        else:
            if len(currentKey) > 2:
                data[currentKey] = items[0]
                str_refs = str_refs + ' ' + items[0]

    if len(data.keys()) == 1:
        for key in data:
            data[key] = str_refs.strip()

    if len(data.keys()) > 1:
        events = splitEnds(str_refs)
        events_length = len(events)
        data_length = len(data)

        if events_length > data_length:
            temp = []
            diff = events_length - data_length
            
            for idx,sent in enumerate(events):
                if idx >= diff:
                    temp.append(" ".join(events[0:diff]) + ' ' + sent)
            events = temp
            events_length = data_length = 1

        if events_length == data_length:
            for idx, key in enumerate(data):
                data[key] = events[idx].strip()
    return data


# Collect data for each entity
full_results = {}
for res in tagged_results:
    if type(res) == nltk.tree.Tree:
        value = constructResults(res)
        
        for key in value:
            if key in full_results.keys():
                full_results[key] = full_results[key] + ' ' + value[key]
            else:
                full_results[key] = value[key]
   
# Show results            
print(full_results)
