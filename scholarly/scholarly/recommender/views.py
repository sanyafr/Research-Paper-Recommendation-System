from django.shortcuts import render
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import gensim
import os
from scholarly.settings import BASE_DIR
import pandas as pd
import ast

model = gensim.models.Word2Vec.load(os.path.join(BASE_DIR, "word2vec.model"))
dataset = pd.read_csv(os.path.join(BASE_DIR, "embeds.csv"))


def preprocess_text(query):
    text = re.sub(r'\[[0-9]*\]', ' ', query)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    for i in range(len(sentences)):
        sentences[i] = [word for word in sentences[i]
                        if word not in stopwords.words('english')]

    for i in range(len(sentences)):
        sentences[i] = ' '.join(sentences[i])
    cleaned_text = ' '.join(sentences)
    return cleaned_text


def find_embed(query):
    query = query.split()
    embed = np.zeros(100)
    global model

    for word in query:
        try:
            embed = embed + np.array(model.wv[word]).reshape(100)
        except:
            pass
    embed = embed/len(query)
    return embed


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def recommend(embed):
    global dataset
    dataset['Score'] = 0
    dataset['Score'] = dataset['Score'].astype('float')
    for index, row in dataset.iterrows():
        cur_embed = np.array(ast.literal_eval(row.embed)).reshape(100)
        sim = cos_sim(cur_embed, embed)
        dataset['Score'][index] = sim

    result = dataset.nlargest(3, 'Score')
    result.reset_index(inplace=True, drop=True)
    return result


def home_view(request):
    if request.method == "GET":
        return render(request, 'home.html')
    if request.method == 'POST':
        query = request.POST.get('query')
        cleaned_query = preprocess_text(query)
        embed = find_embed(cleaned_query)
        papers = recommend(embed)
        print(papers)
        papers = papers.to_dict()
        return render(request, 'resp.html', {'papers': papers})


def rec_view(request):
    global dataset
    papers = dataset.head()
    papers['Score'] = 0.4
    papers['Score'] = papers['Score'].astype('float')
    papers = papers.to_dict()
    return render(request, 'resp.html', {'papers': papers})
