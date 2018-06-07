from textblob.classifiers import NaiveBayesClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

train = []

with open("murdertrain.txt",encoding = 'latin-1') as f:
    for line in f:
        train.append((line.strip(),'murder'))

with open("pos-train.txt",encoding = 'latin-1') as f:
    for line in f:
        train.append((line.strip(),'notMurder'))

cl = NaiveBayesClassifier(train)

data = [
	('they found him lying on his bed with his head split open', 'murder'),
	('he was shot to death', 'murder'),
	('they were killed while traveling', 'murder'),
	('He was anxious to learn about his uncles death', 'murder'),
	('He was killed in action', 'murder'),
	('He was stabbed during a bar fight', 'murder'),
	('John was poisoned', 'murder'),
	('His skull was fractured and he died', 'murder'),
	('She was killed in her sleep', 'murder'),
	('She was choked to death', 'murder'),
	('The villagers were murdered', 'murder'),
	('His parents were murdered and robbed', 'murder'),
	('He has murdered a few people', 'murder'),
	('The man was strangled to death', 'murder'),
	('The woman died of asphyxiation', 'murder'),
	('Megan was shot in the head', 'murder'),
	('She lost her life during a robber', 'murder'),
	('Some men were poisoned of cynide during the dinner', 'murder'),
	('She pushed him under the bus', 'murder'),
	('I killed the girl with a hatchet.', 'murder'),
	('We went to kill him.', 'murder'), 
	('He was hit by car and died.', 'murder'),
	('They killed him and stole his money', 'murder'),
	('Due to the wounds he suffered from the hit and run he died.', 'murder'),
	('We got up and bludgeoned him to death', 'murder'),
	('Karen didnt do anything, she never killed anyone', 'notMurder'),
	('Please wait outside of the house', 'notMurder'),
	('A song can make or ruin a person’s day if they let it get to them', 'notMurder'),
	('Yeah, I think its a good environment for learning English', 'notMurder'),
	('I love eating toasted cheese and tuna sandwiches', 'notMurder'),
	('I want to buy a onesie… but know it won’t suit me', 'notMurder'),
	('Italy is my favorite country; in fact, I plan to spend two weeks there next year', 'notMurder'),
	('The river stole the gods', 'notMurder'),
	('I am happy to take your donation; any amount will be greatly appreciated', 'notMurder'),
	('If the Easter Bunny and the Tooth Fairy had babies would they take your teeth and leave chocolate for you?', 'notMurder'),
	('She borrowed the book from him many years ago and hasnt yet returned it', 'notMurder'),
	('The mysterious diary records the voice', 'notMurder'),
	('If Purple People Eaters are real where do they find purple people to eat?', 'notMurder'),
	('The sky is clear; the stars are twinkling', 'notMurder'),
	('The stranger officiates the meal', 'notMurder'),
	('She was too short to see over the fence', 'notMurder'),
	('Rock music approaches at high velocity', 'notMurder'),
	('He said he was not there yesterday however, many people saw him there', 'notMurder'),
	('Sometimes, all you need to do is completely make an ass of yourself and laugh it off to realise that life isn’t so bad after all', 'notMurder'),
	('We have a lot of rain in June', 'notMurder'),
	('If you like tuna and tomato sauce- try combining the two Its really not as bad as it sounds', 'notMurder'),
	('Christmas is coming', 'notMurder'),
	('I want more detailed information', 'notMurder'),
	('Mary plays the piano', 'notMurder'),
	('Sometimes it is better to just walk away from things and go back to them later when you are in a better frame of mind', 'notMurder'),
	('The memory we used to share is no longer coherent', 'notMurder')
]

train_x = [word[0] for word in train]
train_y = [word[1] for word in train]
test_for_svm = [word[0] for word in data]
test_labels = []

for i in range(51):
	if i < 25:
		test_labels.append('murder')
	else:
		test_labels.append('notMurder')

text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])
text_clf_svm.fit(train_x, train_y)

print(text_clf_svm.score(test_for_svm, test_labels))
print(cl.accuracy(data))