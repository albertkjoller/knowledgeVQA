import os
import pickle

from pprint import pprint
import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora

import pyLDAvis.gensim_models
import pickle 
import pyLDAvis

import nltk
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

class LDA:
    def __init__(self, processed_series, num_topics: int):
        self.processed_series = processed_series
        self.num_topics = num_topics
        
        self.load_data()
        self.build_lda_model()
    
    def load_data(self):
        # auxiliary functions
        def sent_to_words(sentences):
            for sentence in sentences:
                # deacc=True removes punctuations
                yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

        def remove_stopwords(texts):
            return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

        # load data
        self.data = data = self.processed_series.values.tolist()
        self.data_words = data_words = list(sent_to_words(data))

        # remove stop words
        self.data_words = data_words = remove_stopwords(data_words)
        
        # dictionary and corpus creation
        self.id2word = id2word = corpora.Dictionary(data_words)
        self.texts = texts = self.data_words
        self.corpus = [id2word.doc2bow(text) for text in texts] #Term Freq.

    def build_lda_model(self):
        # LDA model
        self.lda_model = lda_model = gensim.models.LdaMulticore(corpus=self.corpus,
                                               id2word=self.id2word,
                                               num_topics=self.num_topics)
    def print_keywords(self):
        # print topic keywords
        print("-"*100)
        print("Topic keywords:")
        print("-"*100)
        pprint(self.lda_model.print_topics())
        self.doc_lda = self.lda_model[corpus]

    def save_html(self, savename: str):
        # Visualize the topics
        pyLDAvis.enable_notebook()
        LDAvis_data_filepath = os.path.join('LDA/'+savename)

        # # this is a bit time consuming - make the if statement True
        # # if you want to execute visualization prep yourself
        if True:
            LDAvis_prepared = pyLDAvis.gensim_models.prepare(self.lda_model, self.corpus, self.id2word)

            with open(LDAvis_data_filepath, 'wb') as f:
                pickle.dump(LDAvis_prepared, f)

        # load the pre-prepared pyLDAvis data from disk
        with open(LDAvis_data_filepath, 'rb') as f:
            LDAvis_prepared = pickle.load(f)

        pyLDAvis.save_html(LDAvis_prepared, 'LDA/'+ savename +'.html')