from utils import *
from RNN import *



#GRADIENT CHECK
"""
np.random.seed(10)
X_train_check, Y_train_check = preprocess(100, True)
model = RNN(100, 10, 1000)
model.gradient_check([0,1,2,3], [1,2,3,4])
"""


vocabulary_size = 7000
hidden_dim = 80
use_existing_data = True
use_existing_model = False
min_sentence_length = 5 

unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

X_train, Y_train, index_to_word, word_to_index = preprocess(vocabulary_size, use_existing_data, unknown_token, sentence_start_token, sentence_end_token)

model = RNN(vocabulary_size, hidden_dim)

#print "Expected Loss for random predictions: %f" % np.log(vocabulary_size)
#print "Actual loss: %f" % model.calculate_loss(X_train[:1000], Y_train[:1000])

model.train(use_existing_model, X_train[:30000], Y_train[:30000], nepoch=150)

lst_failures = []
for i in range(30):
    try:
        sentence=model.generate_sentence(unknown_token, sentence_start_token, sentence_end_token, index_to_word, word_to_index, min_sentence_length)
        print(sentence)
        print ""
    except Exception as e:
        print("----------------------------------------")
        print("FAILURE")
        print(e)
        print("----------------------------------------")
    
