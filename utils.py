import numpy as np
import pickle
import csv
import itertools
import nltk



def preprocess(vocabulary_size, use_existing, unknown_token, sentence_start_token, sentence_end_token):
    if use_existing:
        with open('training_data.pickle', 'rU') as f:
            X_train, Y_train, index_to_word, word_to_index = pickle.load(f)
    else:
        unknown_token = "UNKNOWN_TOKEN"
        sentence_start_token = "SENTENCE_START"
        sentence_end_token = "SENTENCE_END"


        #Reading CSV file in
        print("Reading CSV file")

        with open('data/reddit_text.csv', 'rb') as f:
            reader = csv.reader(f, skipinitialspace=True)
            reader.next()
            #split full comments into sentences
            sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
            #appent SENTENCE_START and SENTENCE_END
            sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
        print("Parsed %d sentences" % (len(sentences)))

        #Tokenize Sentences into words
        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

        #Count the word frequences
        word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
        print("Found %d unique word tokens." % len(word_freq.items()))

        #Build both index_to_word and word_to_index vectors for the most common words
        vocab = word_freq.most_common(vocabulary_size-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(unknown_token)
        word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

        print("Using vocabulary of size %d" % vocabulary_size)
        print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

        #Replace all words not in out vocabulary with unknown_token
        for i, sentence in enumerate(tokenized_sentences):
            tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sentence]

        print("\nExample sentence before preprocessing: '%s'" % sentences[0])
        print("\nExample sentence after preprocessing: '%s'" % tokenized_sentences[0])

        #Create training data
        X_train = np.asarray([[word_to_index[w] for w in sentence[:-1]] for sentence in tokenized_sentences])
        Y_train = np.asarray([[word_to_index[w] for w in sentence[1:]] for sentence in tokenized_sentences]) 
        with open('training_data.pickle', 'w') as f:
            pickle.dump([X_train, Y_train, index_to_word, word_to_index], f)

    return X_train, Y_train, index_to_word, word_to_index




def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)




