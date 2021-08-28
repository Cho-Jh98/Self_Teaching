from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

training_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

# print("\n".join(training_data.data[10].split("\n")[:30]))
# print("Target is: ", training_data.target_names[training_data.target[10]])

# count the word occurrence
count_vector = CountVectorizer()
x_train_count = count_vector.fit_transform(training_data.data)
# print(count_vector.vocabulary_)

# transform the word occurrence into tf-idf
#  TfidfVectorizer = CountVectorizer + TfidfTrnasformer
tfid_transformer = TfidfTransformer()
x_train_tfidf = tfid_transformer.fit_transform(x_train_count)
# print(x_train_tfidf)

model = MultinomialNB().fit(x_train_tfidf, training_data.target)


new = ['My favourite topic has something to do with quantum physics and quantum mechanics',
       'This has nothing to do with church or religion',
       'Software engineering is getting hotter and hotter nowadays']

# 다시 숫자로 변환해줘야 함
x_new_counts = count_vector.transform(new)
x_new_tfidf = tfid_transformer.transform(x_new_counts)

predicted = model.predict(x_new_tfidf)

print(predicted)


for doc, categories in zip(new, predicted):
    print("%r ----------> %s" % (doc, training_data.target_names[categories]))
# 'My favourite topic has something to do with quantum physics and quantum mechanics' ----------> alt.atheism : no suitable category
# 'This has nothing to do with church or religion' ----------> soc.religion.christian
# 'Software engineering is getting hotter and hotter nowadays' ----------> comp.graphics

# we need labels
#