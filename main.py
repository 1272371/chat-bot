import nltk
from nltk.stem import PorterStemmer
stemer=PorterStemmer()

import numpy as np
import tflearn as tfl
import tensorflow as tf
import random
import json
import pickle

class chatbot:

    def fix_data():
            
        with open("intents.json") as file:
            data = json.load(file)

        #print(data["intents"]) 
        words = []
        labels = []
        docs_x = []
        docs_y = []
        try:
            with open('data.pickle','rb') as f:
                words,labels,training.output=pickle.load(f)

        except:
            for intent in data["intents"]:
                for pattern in intent["patterns"]:
                    wrds = nltk.word_tokenize(pattern)
                    words.extend(wrds)
                    docs_x.append(wrds)
                    docs_y.append(intent["tag"])

                if intent["tag"] not in labels:
                    labels.append(intent["tag"])

            words = [stemer.stem(w.lower()) for w in words if w != "?"]
            words = sorted(list(set(words)))

            labels = sorted(labels)

            training = []
            output = []
            out_empty = [0 for _ in range(len(labels))]
            for x,doc in enumerate(docs_x):
                bag = []
                wrds = [stemer.stem(x) for x in doc]
                for w in words:
                    if w in wrds:
                        bag.append(1)
                    else :
                        bag.append(0)
                output_row = out_empty[:]
                output_row[labels.index(docs_y[x])]=1

                training.append(bag)
                output.append(output_row)

            training= np.array(training)
            output = np.array(output)

            with open('data.pickle','wb') as f:
                pickle.dump((words,labels,training,output),f)
        try:
            # Load a model
            tfl.model.load("model.tflearn")
        except: 
                #tf.reset_default_graph()
            tf.compat.v1.reset_default_graph()

            net = tfl.input_data(shape=[None,len(training[0])])
            net = tfl.fully_connected(net,8)
            net = tfl.fully_connected(net,8)
            net = tfl.fully_connected(net,len(output[0]),activation="softmax")
            net = tfl.regression(net)

            model = tfl.DNN(net)

            model.fit(training,output,n_epoch=99,batch_size=8,show_metric=False)
            # Manually save model
            model.save("model.tflearn")
        return model,labels,data,words

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i]=1
    return np.array(bag)

def chat():
    quit_ = set(["q","x","c","quit","exit","exit()"])
    model,labels,data,words = chatbot.fix_data()
    print("Welcome!, type your query")
    while True:
        in_text = input("You :" )
        if in_text.lower() in quit_:
            break
        result = model.predict([bag_of_words(in_text,words)])
        result_idx = np.argmax(result)
        tag = labels[result_idx]
        for tag_ in data["intents"]:
            if(tag==tag_['tag']):
                response = tag_['responses']
        print(random.choice(response))

if __name__ == "__main__":
    chat()