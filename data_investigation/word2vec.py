import gensim
from nltk import ngrams, word_tokenize
from tqdm.notebook import tqdm

from nltk.corpus import stopwords
stoplist = stopwords.words('english')


class word2vec:
    def __init__(self, corpus, dim: int, savename: str):
        self.corpus = corpus
        self.dim = dim
        self.savename = savename
        
        # build model
        self.prepare_data()
        self.build_model()
    
    def get_ngrams(self, words):
        words = [word.lower() for word in words if word not in stoplist and len(word)>2]
        
        bigram=["_".join(phrases) for phrases in list(ngrams(words,2))]
        trigram=["_".join(phrases) for phrases in list(ngrams(words,3))]
        fourgram=["_".join(phrases) for phrases in list(ngrams(words,4))]
        
        return bigram, trigram, fourgram
    
    def prepare_data(self):
        words = self.corpus.words()
        sents = self.corpus.sents()
        print("Number of sentences: ",len(sents))
        print("Number of words: , ",len(words))
                
        data = []
        print("\n Loading data...")
        for sentence in tqdm(sents):
            l1 = [word.lower() for word in sentence if word not in stoplist and len(word) > 2]
            l2 , l3 , l4 = self.get_ngrams(sentence)
            data.append(l1 + l2 + l3 + l4)
            
        self.data = data
            
    def build_model(self):
        model = gensim.models.Word2Vec(self.data, vector_size=self.dim)
        model.save(f'{self.savename}.embedding')
        self.model = model = gensim.models.Word2Vec.load(f'{self.savename}.embedding')
        print("\n Model created and saved!")
        
    
    
    