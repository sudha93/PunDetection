import untangle 
import nltk 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from numpy import asarray 
from numpy import zeros
from keras.models import Sequential
from keras.layers import Dense 
from keras.layers import Embedding 
from keras.layers import Flatten 
from keras.layers import Bidirectional
from keras.layers import SimpleRNN
from keras.layers import LSTM, GRU
#This model is RNN one , no external data used , only for homographic puns 

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
x = pad_sequences(encoded_sent,maxlen=maxLength,padding='pre',truncating= 'post', value= 0.0)

# splitting data into train and test
fraction = 0.7
limit = int(fraction*len(x)) 
x_train = x[:limit]
x_test = x[limit:]

#splitting result into train and test 
y_train = result_data[:limit]
y_test = result_data[limit:]

# loading the glove 50 dimensional embeddings into memory
# it contains 4 lakh word embeddings
f = open('/home/star/passion/puns/glove.6B/glove.6B.50d.txt')
embedding_index = {}
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32') 
    embedding_index[word] = coefs 
f.close() 
#print('Loaded %s word vectors.' % len(embedding_index))
# create a weight matrix for words in training docs
#print tokenizer.word_index.items()
embedding_matrix = zeros((vocab_size, 50))
for i, word in enumerate (tokenizer.word_index.items()):
    embedding_vector = embedding_index.get(word[0])
    if embedding_vector is not None:
        embedding_matrix[i+1] = embedding_vector 
# here each row corresponds to a word embedding 

# model construction 
model = Sequential()
model.add(Embedding(vocab_size,50,weights = [embedding_matrix], input_length=maxLength,trainable = True))
# I can use trainable = True as well , no harm 
model.add(Bidirectional(GRU(40,return_sequences = False , stateful = False)))
#model.add(Flatten())
model.add(Dense(50, activation='tanh'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# train RNN
#for epoch in range(25):
model.fit(x_train,y_train,batch_size=20 , epochs = 10)
print model.predict(x_test)
loss,score = model.evaluate(x_test, y_test, verbose=0)

print 'accuracy : ',score*100
# things that can be changed 
# hidden layer size of RNN 
# mo.of epochs 
# fraction 
# dense layer size 
# I can use a method of making ids , whcih retains commas, fullstops etc 























