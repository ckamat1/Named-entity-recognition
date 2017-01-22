from itertools import tee, islice, chain, izip
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer


'''This method reads train\test data for sentence processing'''
def readconll(file):
    lines = [line.strip() for line in open(file)]
    while lines[-1] == '':  # Remove trailing empty lines
        lines.pop()
    s = [x.split('_') for x in '_'.join(lines).split('__')]  # Quick split corpus into sentences
    return [[y.split() for y in x] for x in s]



'''This method writes data having a content that is same as test data appended with the predicted labels'''
def writeconll(file,test_sentences,predicted_labels):
    f = open(file, 'w')
    i = 0
    for x in test_sentences:
        for y in x:
            sentence = " ".join(y)
            sentence += " " + predicted_labels[i]
            f.write(sentence)
            f.write('\n')
            i = i + 1
        f.write('\n')

'''Collecting train and test sentences'''
train_sentences = readconll('eng.train')
test_sentences = readconll('eng.testa')


'''This method creates a 3-tuple iterables - prevs is an iterable which will help in getting access to prev elements.
Similarly for curr and next elements'''
def collect_features(input_list,prev=''):
    prevs, items, nexts = tee(input_list, 3)
    prevs = chain([prev], prevs)
    nexts = chain(islice(nexts, 1, None), [''])
    return izip(prevs, items, nexts)

'''This method is used to get two sentences together because we would need to access the first element of second sentence
when we encounter the end of first sentence'''
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, '')
    return izip(a, b)

'''This method creates suffixes. For example, for a word, "working", this
method will return ['king','ing','ng','g']'''
def createSuffixes(word):
    n = -4
    suffix_list = []
    while n < 0:
        s1 = word[n:]
        suffix_list.append(s1)
        n += 1
    return suffix_list

'''
This is the main method for extracting features from sentences.
It takes 2 parameters , "sentence" from which we will be extracting the features, and
'training' which is a flag to know whether we are passing training data or test.
When training is set to True, all the NER labels from the training data is collected to train our SVM classifier.
When training is set to False, NER labels are not collected because we need to extract only features for our test data.
Features collected are POS tags, words, checks if first letter of word is in uppercase, checks if all letters of word are in lowercase\uppercase
checks if a word is a digit, suffixes etc..

Return value : List of extracted features along with the classes ( Only for train data). For test, the class list is empty.
'''

def feature_engineering(sentences, training = False):
    prev = ''
    classes = []
    feature_list = []
    for sentence1,sentence2 in pairwise(sentences):
        for previous, current, next in collect_features(sentence1,prev):
            prev_word = previous[0] if previous != '' else previous
            prev_tag = previous[1] if previous != '' else previous
            curr_word = current[0]
            curr_tag = current[1]
            NER_label = current[3]
            s4, s3, s2, s1 = '', '','',''
            prev_word_isCap = str(previous[0].isupper()) if previous != '' else previous
            curr_word_isCap = str(current[0].isupper())
            prev_word_isLower = str(previous[0].islower()) if previous != '' else previous
            curr_word_isLower = str(current[0].islower())
            prev_word_initCap = str(previous[0][0].isupper()) if previous !='' else previous
            curr_word_initCap = str(current[0][0].isupper())
            prev_word_isDigit = str(previous[0].isdigit()) if previous != '' else previous
            curr_word_isDigit = str(current[0].isdigit())
            if len(current[0]) >= 7:
                suff_list = createSuffixes(current[0])
                s4 , s3 , s2, s1 = suff_list[0],suff_list[1],suff_list[2],suff_list[3]


            if training == True:
                classes.append(NER_label)
            if next == '':
                prev = [curr_word,curr_tag]
                next_word = sentence2[0][0]
                next_tag = sentence2[0][1]
                next_word_isCap = str(sentence2[0][0].isupper())
                next_word_isLower = str(sentence2[0][0].islower())
                next_word_initCap = str(sentence2[0][0][0].isupper())
                next_word_isDigit = str(sentence2[0][0].isdigit())
                feature_list.append({'prev_word=' + prev_word:1,'next_word='+next_word:1,'curr_word='+curr_word:1,
                                     'prev_tag='+prev_tag:1,'next_tag='+next_tag:1,'curr_tag='+curr_tag:1,
                                     'prev_caps='+ prev_word_isCap:1, 'curr_caps='+ curr_word_isCap:1,
                                     'next_caps=' + next_word_isCap:1,
                                    'prev_initCaps='+prev_word_initCap:1,'curr_initCaps='+curr_word_initCap:1,'next_initCaps='+next_word_initCap:1,
                                    'prev_wordDigit='+prev_word_isDigit:1,'curr_wordDigit='+curr_word_isDigit:1,'next_wordDigit='+next_word_isDigit:1,
                                    'prev_lower='+prev_word_isLower:1,'curr_lower='+curr_word_isLower:1,'next_lower='+next_word_isLower:1,
                                     'suff='+s4:1,'suff='+s3:1,'suff='+s2:1,'suff='+s1:1
                                     })
            else:
                next_word = next[0]
                next_tag = next[1]
                next_word_isCap = str(next[0].isupper())
                next_word_isLower = str(next[0].islower())
                next_word_initCap = str(next[0][0].isupper())
                next_word_isDigit = str(next[0].isdigit())
                feature_list.append(
                    {'prev_word=' + prev_word: 1, 'next_word=' + next_word: 1, 'curr_word=' + curr_word: 1,
                     'prev_tag=' + prev_tag: 1, 'next_tag=' + next_tag: 1, 'curr_tag=' + curr_tag: 1,
                     'prev_caps=' + prev_word_isCap: 1, 'curr_caps=' + curr_word_isCap: 1,
                     'next_caps=' + next_word_isCap: 1,
                     'prev_initCaps=' + prev_word_initCap: 1, 'curr_initCaps=' + curr_word_initCap: 1,
                     'next_initCaps=' + next_word_initCap: 1,
                     'prev_wordDigit=' + prev_word_isDigit: 1, 'curr_wordDigit=' + curr_word_isDigit: 1,
                     'next_wordDigit=' + next_word_isDigit: 1,
                     'prev_lower=' + prev_word_isLower: 1, 'curr_lower=' + curr_word_isLower: 1,
                     'next_lower=' + next_word_isLower: 1,
                     'suff=' + s4: 1, 'suff=' + s3: 1, 'suff=' + s2: 1, 'suff=' + s1: 1
                     })
    prev = [curr_word, curr_tag]
    for p,c,n in collect_features(sentence2,prev):
        prev_word = p[0] if p != '' else p
        prev_tag = p[1] if p != '' else p
        curr_word = c[0]
        curr_tag = c[1]
        next_word = n[0] if n != '' else n
        next_tag = n[1] if n != '' else n
        prev_word_isCap = str(p[0].isupper()) if p != '' else p
        curr_word_isCap = str(c[0].isupper())
        next_word_isCap = str(n[0].isupper()) if n != '' else n
        prev_word_isLower = str(p[0].islower()) if p != '' else p
        curr_word_isLower = str(c[0].islower())
        next_word_isLower = str(n[0].islower()) if n != '' else n
        prev_word_initCap = str(p[0][0].isupper()) if p != '' else p
        curr_word_initCap = str(c[0][0].isupper())
        next_word_initCap = str(n[0][0].isupper()) if n != '' else n
        prev_word_isDigit = str(previous[0].isdigit()) if previous != '' else previous
        curr_word_isDigit = str(current[0].isdigit())
        next_word_isDigit = str(n[0].isdigit()) if n != '' else n
        NER_label = c[3]
        if training == True:
            classes.append(NER_label)
        feature_list.append({'prev_word=' + prev_word: 1, 'next_word=' + next_word: 1, 'curr_word=' + curr_word: 1,
                             'prev_tag=' + prev_tag: 1, 'next_tag=' + next_tag: 1, 'curr_tag=' + curr_tag: 1,
                             'prev_caps=' + prev_word_isCap: 1, 'curr_caps=' + curr_word_isCap: 1,
                             'next_caps=' + next_word_isCap: 1,
                             'prev_initCaps=' + prev_word_initCap: 1, 'curr_initCaps=' + curr_word_initCap: 1,
                             'next_initCaps=' + next_word_initCap: 1,
                             'prev_wordDigit=' + prev_word_isDigit: 1, 'curr_wordDigit=' + curr_word_isDigit: 1,
                             'next_wordDigit=' + next_word_isDigit: 1,
                             'prev_lower=' + prev_word_isLower: 1, 'curr_lower=' + curr_word_isLower: 1,
                             'next_lower=' + next_word_isLower: 1,
                             'suff=' + s4: 1, 'suff=' + s3: 1, 'suff=' + s2: 1, 'suff=' + s1: 1
                             })
    return feature_list,classes

features, cls = feature_engineering(train_sentences,training = True) # we are extracting features and labels to train our svm classifier
features1 = feature_engineering(test_sentences)[0] #we are only extracting the features for test data

vectorizer = DictVectorizer(sparse = True)
X = vectorizer.fit_transform(features)
clf = svm.LinearSVC() # Using Linear SVC
clf.fit(X,cls)

predicted_labels = clf.predict(vectorizer.transform(features1)) #predicting the labels on the test data by passing its features

'''writing the data to eng.guessa by calling writeconll function in the format so that
it can be evaluated by the perl script'''
writeconll('eng.guessa',test_sentences,predicted_labels)


'''Uncomment the below 4 lines to output a file eng.guessb after testing on eng.testb'''
# testb_sentences = readconll('eng.testb')
# features2 = feature_engineering(testb_sentences)[0]
# predictedb_labels = clf.predict(vectorizer.transform(features2))
# writeconll('eng.guessb',testb_sentences,predictedb_labels)