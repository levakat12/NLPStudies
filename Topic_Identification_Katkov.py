from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.stem import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel
import nltk

#Depedencies
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

#stopwords from datasets
with open('datasets/english_stopwords.txt', 'r') as file:
    stop_words = set(file.read().splitlines())
    
#article to analyze
with open('Test/Boston_Celtics.txt', 'r') as file:
    articles = file.readlines()
    
#tokenization
tokens = [word_tokenize(text) for text in articles]

#making tokens lower case
lower_tokens = [[t.lower() for t in text_tokens] for text_tokens in tokens]

#making tokens only alphabetical
alpha_only = [[t for t in text_tokens if t.isalpha()] for text_tokens in lower_tokens]

#skip any stops
no_stops = [[t for t in text_tokens if t not in stop_words] for text_tokens in alpha_only]

#declaring lemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

#lematizing words for easier understanding
lemmatized = [[wordnet_lemmatizer.lemmatize(t) for t in text_tokens] for text_tokens in no_stops]

#gensim dicitonary
dictionary = Dictionary(lemmatized)

#gensim corpus
corpus = [dictionary.doc2bow(article) for article in lemmatized]

bow = Counter([item for sublist in lemmatized for item in sublist])

#tfidf block that finds the 'heaviest terms'
tfidf = TfidfModel(corpus)
doc = corpus[0]
tfidf_weights = tfidf[doc]
sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)
top_tfidf_weights = [(dictionary.get(term_id), weight) for term_id, weight in sorted_tfidf_weights[:5]]

#output
print(bow.most_common(5))
print(top_tfidf_weights)