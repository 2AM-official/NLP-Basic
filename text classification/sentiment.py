#!/bin/python

from re import S
from weakref import WeakValueDictionary


def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")

    # from sklearn.feature_extraction.text import CountVectorizer
    # sentiment.count_vect = CountVectorizer()
    # sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    # sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    from sklearn.feature_extraction.text import TfidfVectorizer
    sentiment.count_vect = TfidfVectorizer(ngram_range=(1,2), tokenizer=lemmatization)
    #sentiment.count_vect = TfidfVectorizer(ngram_range=(1,2), tokenizer=stemming)
    #sentiment.count_vect = TfidfVectorizer(ngram_range=(1,2), preprocessor=preprocess)
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)

    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment

def preprocess(text):
    import re
    # remove numbers
    text = re.sub(r'[0-9]', '', text)
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # lower case
    text = text.lower()
    return text

def lemmatization(text):
    import nltk
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(wordnet_lemmatizer.lemmatize(item))
    return stems

def stemming(text):
    import nltk
    #nltk.download('punkt')
    from nltk.stem.porter import PorterStemmer
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems


def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels


if __name__ == "__main__":
    
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    
    sentiment = read_files(tarfname)
    print("\nTraining classifier")
    import classify
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy)
    print("\nEvaluating")
    classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
    classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
