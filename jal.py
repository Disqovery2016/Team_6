from textblob.classifiers import NaiveBayesClassifier
from textblob import TextBlob

train = [
    ('1, 8.1, 89', 'safe'),
    ('1,8.5,122 ', 'safe'),
    ('I feel very good about these beers.', 'pos'),
    ('This is my best work.', 'pos'),
    ("What an awesome view", 'pos'),
    ('I do not like this restaurant', 'neg'),
    ('I am tired of this stuff.', 'neg'),
    ("I can't deal with this", 'neg'),
    ('He is my sworn enemy!', 'neg'),
    ('My boss is horrible.', 'neg')
]
test = [
    ('The beer was good.', 'pos'),
    ('I do not enjoy my job', 'neg'),
    ("I ain't feeling dandy today.", 'neg'),
    ("I feel amazing!", 'pos'),
    ('Gary is a friend of mine.', 'pos'),
    ("I can't believe I'm doing this.", 'neg')
]

cl = NaiveBayesClassifier(train)

# Classify some text
print(cl.classify("Their burgers are amazing."))  # "pos"
print(cl.classify("I don't like their pizza."))   # "neg"

# Compute accuracy
print("Accuracy: {0}".format(cl.accuracy(test)))

