import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import os
import numpy as np

def get_file_names():
    file_names = []
    for file in os.listdir("."):
        if file.endswith(".xml"):
            file_names.append( file)
    return file_names

def read_xml_file(file_path):
    with open(file_path, 'r') as f:
        soup = BeautifulSoup(f, 'xml')
    return soup

def get_sentences(soup):
    sentences = [sentence.text for sentence in soup.find_all('sentence')][:-1] # last sentence is not needed
    return sentences

def get_document(filename):
    soup = read_xml_file(filename)
    cur_document = get_sentences(soup)
    return cur_document

def get_all_documents(filename_list):
    documents = []
    for file in filename_list:
        cur_document = get_document(file)
        documents.append(cur_document)
    return documents

def create_vectorizer():
    files = get_file_names()
    documents = get_all_documents(files)
    mega_doc = [" ".join(document) for document in documents]
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_vectorizer.fit(mega_doc)
    return tfidf_vectorizer

def get_most_inf_sentences(doc,tfidf_vectorizer,n=5): 
    tfidf_matrix = tfidf_vectorizer.transform(doc)
    cosine_similarities = cosine_similarity(tfidf_matrix,tfidf_matrix)
    average_similarities = cosine_similarities.mean(axis=1)
    most_inf_sentence_ind = np.argsort(average_similarities)[-n:]
    return [doc[i] for i in most_inf_sentence_ind]

def get_summary(filename):
    doc = get_document(filename)
    vectorizer = create_vectorizer()
    most_inf_sentences = get_most_inf_sentences(doc,vectorizer)
    return most_inf_sentences

if __name__ == "__main__":
    file_name = input("File name:")
    summary = get_summary(file_name)
    print("Summary:")
    for sentence in summary:
        print(sentence)