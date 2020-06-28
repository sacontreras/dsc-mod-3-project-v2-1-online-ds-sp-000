# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np
from IPython.core.display import HTML, Markdown
import matplotlib.pyplot as plt
from .skl_transformers import SimpleValueTransformer

import nltk
# # we only need the following downloads the first time
# # nltk.download('punkt')
# # nltk.download('stopwords')
# # nltk.download('wordnet')
from nltk.corpus import stopwords
from scipy.stats import entropy
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans



def preprocess__lcase_strip(doc):
    preprocessed_doc = doc.lower()
    preprocessed_doc = re.sub(r"[^a-z]", " ", preprocessed_doc)
    preprocessed_doc = preprocessed_doc.replace(r"\s+", " ")
    return preprocessed_doc

def preprocess__tokenize(doc):
    return nltk.word_tokenize(doc)

# manipulate stop words as necessary here
def get_stopwords():
    english_stopwords = stopwords.words('english') # comment out to not filter stop words
    if 'not' in english_stopwords:
        english_stopwords.remove('not')
    return english_stopwords

english_stopwords = get_stopwords()
def preprocess__filter_stopwords(doc, is_list=False):
    tokenized_doc = preprocess__tokenize(doc) if not is_list else doc
    tokenized_doc = [word for word in tokenized_doc if word not in english_stopwords]
    return tokenized_doc, ' '.join(tokenized_doc)

def preprocess_all(doc):
    preprocessed_doc = preprocess__lcase_strip(doc)
    tokenized_doc = preprocess__tokenize(preprocessed_doc)
    return preprocess__filter_stopwords(tokenized_doc, is_list=True)[1]

def tfidf_vocab_to_idx_map(tfidf_vocab):
    idx_map = {}
    for term, feat_idx in tfidf_vocab.items():
        if feat_idx in idx_map:
            current_mapping = idx_map[feat_idx]
            print(f"***WARNING!!!*** feature index {feat_idx} is currently mapped to term '{current_mapping}' but will be replace with {term}")
        idx_map[feat_idx] = term
    return idx_map

def tfidf_vec_to_doc(tfidf_vec, idx_term_map, suppress_output=True):
    idf_doc = []
    for vocab_word_index, idf in enumerate(tfidf_vec):
        if idf > 0:
            word = idx_term_map[vocab_word_index]
            if not suppress_output:
                print(f"\tidf of '{word}': {idf}")
            idf_doc.append((word, idf))
    return ' '.join([word_idf[0] for word_idf in idf_doc]), idf_doc

def doc_to_tfidf_fit(doc, tfidf_vectorizer, idx_term_map=None, suppress_output=True):
    if idx_term_map is None:
        idx_term_map = tfidf_vocab_to_idx_map(tfidf_vectorizer.vocabulary_)
    tfidf_vector = tfidf_vectorizer.transform([doc]).toarray()[0]
    if not suppress_output:
        print(f"fitted doc '{doc}' has idf vector: {tfidf_vector}")
    fitted_doc = tfidf_vec_to_doc(tfidf_vector, idx_term_map, suppress_output)
    return fitted_doc, tfidf_vector

def kmeans_centroids_to_docs(kmeans, idx_term_map):
    df = pd.DataFrame(columns=['centroid_idx', 'tfidf_vector', 'doc'])
    for centroid_idx, centroid_vec in enumerate(kmeans.cluster_centers_):
        centroid_doc = tfidf_vec_to_doc(centroid_vec, idx_term_map)[0]
        data = [{
            'centroid_idx': centroid_idx, 
            'tfidf_vector': centroid_vec,
            'doc': centroid_doc
        }]
        df = df.append(data, ignore_index=True, sort=False)
    df = df.set_index('centroid_idx')
    return df

def predict_kmeans_cluster_membership(doc, kmeans, tfidf_vectorizer, idx_term_map, suppress_output=True):
    centroid_idx = kmeans.predict(tfidf_vectorizer.transform([doc]))[0]
    if not suppress_output:
        display(HTML(f"predicted centroid idx: {centroid_idx}"))
    centroid_vector = kmeans.cluster_centers_[centroid_idx]
    cluster_doc = tfidf_vec_to_doc(centroid_vector, idx_term_map)[0]
    if not suppress_output:
        display(HTML(f"for doc '<b>{doc}</b>', cluster membership is: '<b>{cluster_doc}</b>'<p><br>"))
    return centroid_idx, centroid_vector, cluster_doc

s_all_done = "\tALL DONE!"
def _tfidf_fit_corpus_from_feat(df, feat):
    corpus = list(df[feat].unique()) # note that we use the raw (un-preprocessed) feature since we pass in the preprocessor (preprocess_all) to the TfidfVectorizer
    display(HTML(f"there are {len(corpus)} unique documents in the <i>{feat}</i> corpus"))
    tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess_all)
    display(HTML(f"<p><br>fitting the <code>TfidfVectorizer</code> model to the corpus..."))
    tfidf = tfidf_vectorizer.fit_transform(corpus)
    display(HTML(f"<pre>{s_all_done}</pre>"))
    display(HTML(f"there are {len(tfidf_vectorizer.vocabulary_)} terms in the vocabulary after fitting <code>TF-IDF</code> model to the corpus"))
    return corpus, tfidf, tfidf_vectorizer

def _kmeans_from_tfidf(tfidf, idx_term_map, n_clusters):
    display(HTML(f"<p><br>building <code>KMeans</code> model with n_clusters=={n_clusters}..."))
    kmeans = KMeans(n_clusters=n_clusters).fit(tfidf)
    df_kmeans_clusters = kmeans_centroids_to_docs(kmeans, idx_term_map)
    display(HTML(f"<pre>{s_all_done}</pre>"))
    return kmeans, df_kmeans_clusters

def _tfidf_kmeans_classify_feature(df, feat, kmeans, tfidf_vectorizer, idx_term_map):
    # add new "class" feature
    feat_name_class = f"{feat}_tfidf_kmeans_class"
    display(HTML(f"<p><br>mapping DIRTY {feat}s to corresponding {feat_name_class}es..."))
    df[feat_name_class] = df[feat].map(
        lambda feat_val: predict_kmeans_cluster_membership(
            feat_val, 
            kmeans, 
            tfidf_vectorizer, 
            idx_term_map, 
            suppress_output=True
        )[0]
    )
    display(HTML(f"<pre>{s_all_done}</pre>"))
    return df, feat_name_class


def tfidf_fit(df, df_name, feat):
    df_copy = df.copy()

    # fit TF-IDF to the corpus
    corpus, tfidf, tfidf_vectorizer = _tfidf_fit_corpus_from_feat(df_copy, feat)

    # do this beforehand to avoid recomputing it every time, should we pass in more than one document (installer)... which we do below
    display(HTML(f"<p><br>building the idx term map..."))
    idx_term_map = tfidf_vocab_to_idx_map(tfidf_vectorizer.vocabulary_)
    display(HTML(f"<pre>{s_all_done}</pre>"))
    
    return corpus, tfidf, tfidf_vectorizer, idx_term_map
    
def tfidf_transform(df, df_name, feat, tfidf_vectorizer, idx_term_map):
    df_copy = df.copy()
    
    # for display to the reader to show the evolution from DIRTY to TF-IDF "cleaned"
    # add the result of the first step of preprocessing: coverting to lower-case
    feat_name_stripped_lcase = f"{feat}_stripped_lcase"
    df_copy[feat_name_stripped_lcase] = df_copy[feat].apply(preprocess__lcase_strip)
    # add the result of the next step of preprocessing: tokenization
    feat_name_word_tokenized = f"{feat}_word_tokenized"
    df_copy[feat_name_word_tokenized] = df_copy[feat_name_stripped_lcase].apply(preprocess__tokenize)
    # add the result of the next step of preprocessing: remove stop-words
    feat_name_word_tokenized_no_stopwords = f"{feat}_word_tokenized_no_stopwords"
    df_copy[feat_name_word_tokenized_no_stopwords] = df_copy[feat_name_word_tokenized].apply(
        lambda feat_word_tokenized: preprocess__filter_stopwords(feat_word_tokenized, is_list=True)[0]
    )
    
    feat_name_after_tfidf = f"{feat}_after_tfidf"

    # now fit docs to tf-idf vectors
    display(HTML(f"<p><br>mapping DIRTY <i>{feat}</i> documents to <code>TF-IDF</code> vectors..."))
    df_copy[feat_name_after_tfidf] = df_copy[feat].apply(
        lambda _feat: doc_to_tfidf_fit(_feat, tfidf_vectorizer, idx_term_map)[0][0]
    )
    
    # replace feat with feat_name_after_tfidf
    df_copy = df_copy.drop(feat, axis=1)
    df_copy[feat] = df_copy[feat_name_after_tfidf]
    
    # clean up df_copy
    df_copy = df_copy.drop(
        [
            feat_name_stripped_lcase,
            feat_name_word_tokenized,
            feat_name_word_tokenized_no_stopwords,
            feat_name_after_tfidf
        ], 
        axis=1
    )
    
    return df_copy
    

def tfidf_kmeans_classify_feature__fit(df, df_name, feat, mean_cluster_size=None, verbosity=1):
    """
    IMPORTANT!  Set mean_cluster_size only if you want to OVERRIDE the default beahvior to base KMeans n_clusters on entropy of TF-IDF doc distribution.
        
        *** IN GENERAL, THIS IS A BAD IDEA UNLESS YOU HAVE AN EXPLICIT REASON FOR DOING SO! ***

    Other notes:
        nan values MUST be dealt with beforehand!
    """

    df_copy = df.copy()

    # fit TF-IDF to the corpus
    corpus, tfidf, tfidf_vectorizer = _tfidf_fit_corpus_from_feat(df_copy, feat)

    # for display to the reader to show the evolution from DIRTY to TF-IDF "cleaned"
    # add the result of the first step of preprocessing: coverting to lower-case
    feat_name_stripped_lcase = f"{feat}_stripped_lcase"
    df_copy[feat_name_stripped_lcase] = df_copy[feat].apply(preprocess__lcase_strip)
    # add the result of the next step of preprocessing: tokenization
    feat_name_word_tokenized = f"{feat}_word_tokenized"
    df_copy[feat_name_word_tokenized] = df_copy[feat_name_stripped_lcase].apply(preprocess__tokenize)
    # add the result of the next step of preprocessing: remove stop-words
    feat_name_word_tokenized_no_stopwords = f"{feat}_word_tokenized_no_stopwords"
    df_copy[feat_name_word_tokenized_no_stopwords] = df_copy[feat_name_word_tokenized].apply(
        lambda feat_word_tokenized: preprocess__filter_stopwords(feat_word_tokenized, is_list=True)[0]
    )

    # do this beforehand to avoid recomputing it every time, should we pass in more than one document (installer)... which we do below
    display(HTML(f"<p><br>building the idx term map..."))
    idx_term_map = tfidf_vocab_to_idx_map(tfidf_vectorizer.vocabulary_)
    display(HTML(f"<pre>{s_all_done}</pre>"))
    feat_name_after_tfidf = f"{feat}_after_tfidf"

    # now fit docs to tf-idf vectors
    display(HTML(f"<p><br>fitting DIRTY <i>{feat}</i> documents to <code>TF-IDF</code> vectors..."))
    df_copy[feat_name_after_tfidf] = df_copy[feat].apply(
        lambda _feat: doc_to_tfidf_fit(_feat, tfidf_vectorizer, idx_term_map)[0][0]
    )
    display(HTML(f"<pre>{s_all_done}</pre>"))
    if verbosity > 1:
        cols_for_this_feat = [feat, feat_name_stripped_lcase, feat_name_word_tokenized, feat_name_word_tokenized_no_stopwords, feat_name_after_tfidf]
        display(HTML(f"<h3>First few rows of {df_name} TF-IDF DataFrame (verbosity>1)</h3>"))
        display(HTML(df_copy[cols_for_this_feat].head(10).to_html()))

    # THIS PART IS KEY!  Entropy is the basis for setting the proper cluster size and hence the proper n_clusters parameter to build the KMeans model!
    dist_normalized = df_copy[feat_name_after_tfidf].value_counts(normalize=True)
    _entropy = entropy(dist_normalized, base=2)
    display(HTML(f"<p><br>info: mean_cluster_size=={mean_cluster_size}; calculated entropy: {_entropy}"))
    if mean_cluster_size is None:
        mean_cluster_size = _entropy
        display(HTML(f"<p><br>set mean_cluster_size={mean_cluster_size}"))
        
    # build KMeans model
    n_clusters = int(len(corpus)/mean_cluster_size) # 8 is default n_clusters value for KMeans
    kmeans, df_kmeans_clusters = _kmeans_from_tfidf(tfidf, idx_term_map, n_clusters)

    # clean up df_copy
    df_copy = df_copy.drop(
        [
            feat_name_stripped_lcase,
            feat_name_word_tokenized,
            feat_name_word_tokenized_no_stopwords,
            feat_name_after_tfidf
        ], 
        axis=1
    )

    return df_copy, corpus, tfidf, tfidf_vectorizer, idx_term_map, kmeans, df_kmeans_clusters

def tfidf_kmeans_classify_feature__transform(df, df_name, feat, tfidf_vectorizer, idx_term_map, kmeans, df_kmeans_clusters, verbosity=1, display_max_rows=25):
    """
    Other notes:
        nan values MUST be dealt with beforehand!
    """

    df_copy = df.copy()

    # finally, fit docs to the new tfidf kmeans class - this adds new f"{feat}_tfidf_kmeans_class" feature
    #   note that this is what should be use to classify, for example, the test/validation date set
    #   a new model should NOT be built for the test/validation date set
    df_copy, feat_name_class = _tfidf_kmeans_classify_feature(
        df_copy, 
        feat, 
        kmeans, 
        tfidf_vectorizer, 
        idx_term_map
    )

    if verbosity > 0:
        display(HTML(f"<h3><i>{feat}</i> to <i>{feat_name_class}</i> Mapping:</h3>"))
        display(HTML(df_copy[[feat, feat_name_class]].to_html(notebook=True, justify='left', max_rows=display_max_rows)))

        display(HTML(f"<p><br>building distribution plot of {feat_name_class}..."))
        display(HTML(f"<h3><i>{feat_name_class}</i> Distribution:</h3>"))
        plt.figure(figsize=(15,6))
        df_copy[feat_name_class].hist(bins=len(df_kmeans_clusters))
        plt.show()


    display(HTML(f"<p><br>computing <i>frequency</i> of {feat_name_class}..."))
    # there may be some classes from the training set which do not occur in test/validation
    #   so reset to the class having median frequency in X
    df_kmeans_clusters_copy = df_kmeans_clusters.copy()
    df_kmeans_clusters_copy = df_kmeans_clusters_copy.reset_index()
    df_kmeans_clusters_copy['frequency'] = 0
    for _class, class_freq in df_copy[feat_name_class].value_counts().items():
        df_kmeans_clusters_copy['frequency'] = np.where(
            df_kmeans_clusters_copy['centroid_idx']==_class, 
            class_freq, 
            df_kmeans_clusters_copy['frequency']
        )

    df_kmeans_clusters_copy = df_kmeans_clusters_copy.set_index('centroid_idx')
    display(HTML(f"<pre>{s_all_done}</pre>"))
    if verbosity > 0:
        df_mapped_classes = df_kmeans_clusters_copy[df_kmeans_clusters_copy.frequency>0]
        df_unmapped_classes = df_kmeans_clusters_copy[df_kmeans_clusters_copy.frequency==0]
        display(HTML(f"<h3><code>{len(df_mapped_classes)} KMeans</code> Cluster Centroids (<i>{feat_name_class}</i>) from fit occur in X, ordered by <i>frequency</i>:</h3>"))
        display(HTML(df_kmeans_clusters_copy[df_kmeans_clusters_copy.frequency>0].sort_values(by='frequency', ascending=False).to_html(notebook=True, justify='left', max_rows=display_max_rows))) 
        if len(df_unmapped_classes) > 0:
            display(HTML(f"<h3><code>{len(df_unmapped_classes)} KMeans</code> Cluster Centroids from fit do not ocur in X</h3>"))

    return df_copy, df_kmeans_clusters_copy, feat_name_class