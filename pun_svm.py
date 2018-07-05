
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import untangle 
import nltk 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from numpy import asarray 
from numpy import zeros

homo1_data = "/home/star/passion/puns/semeval2017_task7/data/test/subtask1-homographic-test.xml"

hetero1_data = "/home/star/passion/puns/semeval2017_task7/data/test/subtask1-heterographic-test.xml"

homo1_result = "/home/star/passion/puns/semeval2017_task7/data/test/subtask1-homographic-test.gold"

input_data = []
result_data = []
sentenceList= []
obj = untangle.parse(homo1_data)
# storing the homo_pun subtask1 results in a list 
with open(homo1_result) as file:
    for line in file:
        line = line.strip()
        temp = nltk.word_tokenize(line)
        result_data.append(temp[1])
# result_data
#print len(result_data) 


#storing the homo_pun subtask1 data in the form of list of lists 
i = 0
while ( i < len(obj.corpus)):
    sublist = []
    j = 0
    while( j < len(obj.corpus.text[i]) ):
        data = obj.corpus.text[i].word[j].cdata
        sublist.append(data.encode('utf-8'))
        j = j+1
    input_data.append(sublist)
    i = i+1

# but we need the sentences in list form for tokenizer function 
for item in input_data:
    string = ' '.join(item)
    sentenceList.append(string)
#print sentenceList

# checking the lengths of sequences
lengthList = []
for sublist in input_data:
    lengthList.append(len(sublist)) 

sorted_lengthList =  sorted(lengthList, reverse = True)

# converting text into list of lists of ids using Tokenizer 
# But this  process removes comma , fullstops etc , only considers text 
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentenceList)
vocab_size = len(tokenizer.word_index)+1
#print vocab_size
#print tokenizer.word_index.items()
encoded_sent = tokenizer.texts_to_sequences(sentenceList)
# here encoded_sent is in the form of list of lists of ids 
maxLength = 50 
x = pad_sequences(encoded_sent,maxlen=maxLength,padding='pre',truncating= 'post', value= 0.5)

# splitting data into train and test
fraction = 0.7
limit = int(fraction*len(x)) 
x_train = x[:limit]
x_test = x[limit:]

#splitting result into train and test 
y_train = result_data[:limit]
y_test = result_data[limit:]


def svmClassifier():
	# svm classifier starts 
	classifier = svm.SVC( kernel= 'rbf' ,C=500, gamma='auto')
	# cross validation required
	#classifier.fit(x_train, y_train)
	predictions = cross_val_predict(classifier, x_train, y_train, cv=4)
	#predictions = classifier.predict(x_test)
	#print predictions[:200]
	acc = accuracy_score(y_train, predictions)
	return acc
#print svmClassifier()

def adaBoostClassifier():
	abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
 		n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
	abc.fit(x_train,y_train)
	acc = accuracy_score(y_test, abc.predict(x_test))
	#print abc.predict(x_test)
	return acc

#print adaBoostClassifier()

def randomForestClassifier():
	rfc = RandomForestClassifier(n_estimators=15)
	rfc.fit(x_train,y_train)
	acc = accuracy_score(y_test, rfc.predict(x_test))
	#print rfc.predict(x_test)
	return acc

print randomForestClassifier()


# svm predicts all ones , no zeros at all ???
	






















