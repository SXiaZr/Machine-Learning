# Machine Learning Project 2: Machine Learning for Science

`Teammates`: Yan Fu, Shengzhao Xia, Runzhe Liu

For this project, we joined the lab in Chair of Economics and Management of Innovation. 

In this project, we did:

1. Data collection: Web scraping from [Web of Science](http://apps.webofknowledge.com/WOS_GeneralSearch_input.do?product=WOS&search_mode=GeneralSearch&SID=F41mtBBV1mNZKmygFN7&preferencesSaved=) and get over 350,000 paper records;
2. Name disambiguation: Disambiguation of different authors using same name; 
3. Classification: Convert keywords to numerical representations and use supervised machine learning methods (eg. logistic regression, neural network) to predict if one researcher would be dismissed or not.

To reproduce our result one will need to:

1. Install relevant libraries on their computers:
- NLP and machine learing library: [`gensim`](https://radimrehurek.com/gensim/)(conda install -c conda-forge gensim),[`sklearn`](https://scikit-learn.org/stable/)(pip install -U scikit-learn);
- Deep learning labrary: [`Tensorflow`](https://www.tensorflow.org)(pip install tensorflow), [`keras`](https://keras.io)(pip install keras).
2. Download [GoogleNews-vectors-negative300.bin.gz](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) and put it into *data fold* to do word2vec conversion, we did not include it in our submission because of its large size.
3. Download [wiki-news-300d-1M-subword.vec](https://fasttext.cc/docs/en/english-vectors.html) and put it into *data fold* to do FastText conversion, we did not include it in our submission because of its large size.

Below we will introduce files and functions in our repository.
And because we **cannot** upload the whole dataset (the dismissed author list is sensitive to some extent; and the whole dataset is very large), we comment out the code where origin data is used and save and load the intermediate output of preprocessing part.

## Auxiliary modules

### `Scrapping.ipynb`, `scraper_WOS.py`, `parsing.py`, `DataCleaning.ipynb`:
These files include codes used for web scraping and basic data cleaning, and they are *NOT* reproducible because we delete the username and password needed for apply request from WOS. 

### `Undis_List_Generation_for_FastText.ipynb`:

Undis_List_Generation_for_FastText.ipynb is written to create a list of undismissed researchers extracted from database *author_df_nodup_aff.csv*. The list is used for the generation of feature by FastText. 

This file is *NOT* reproducible because the data *author_df_nodup_aff.csv* cannot be shared for the sake of privacy.  

## Working notebook
### `Disambiguation.ipynb`: 
This notebook is written for name disambiguation. We use kNN+LOF outlier dectection in it and distinguish different researchers sharing same name by their subjects. 

### `classification.ipynb`:
This notebook is written for author classification. The main objective of classification is to build a model based on researchers’ publications to predict whether he/she would get the sack after the repression of the coup. 

### The folder `Data`:
* #### Data obtained after data pre-processing:
`author_explict_dict.pickle` contains paper data of each author in dismissed list.

`not_dis_author_dict.pickle` contains paper data of each author in undismissed list in "set 1".

`total_no_dis_pub_2.csv` and `no_dis_auth_vec_2.csv` contain publications from undismissed author and vectors from FASTTEXT in "Set 2".

* #### Feature extracted by Word2vec:
`X_keyword_vector.pickle` and `y_keyword_vector.pickle` contain features extracted by Word2vec method and corresponding labels.

* #### Feature extracted by Bag-of-words:
`X_bag_of_words.pickle` and `y_bag_of_words.pickle` contain features extracted by Bag-of-words method and corresponding labels.

* #### Feature extracted by FastTest:
`X_set1_fasttext.pickle`, `y_set1_fasttext.pickle`, `X_set2_fasttext.pickle`, `y_set2_fasttext.pickle` contain X and y to train our model in the FASTTEXT part.
