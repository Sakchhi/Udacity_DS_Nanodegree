import sys
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords, wordnet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report


def load_data(database_filepath):
    """
    Read the data to be modeled from sqlite database
    :param database_filepath: Path of database, including database name
    :return:
        Messages to be used as features for modeling
        Binary values for target features
        Unique target classes in the data
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response_etl', engine)
    X = df.message
    Y = df.iloc[:, 4:]
    categories = Y.columns.tolist()
    return X, Y, categories


def tokenize(text):
    """
    Clean, tokenize and lemmatize text
    :param text: Text to tokenize
    :return: list of tokens
    """
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    # tagged = nltk.pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # for word, tag in tagged:
    #     wntag = get_wordnet_pos(tag)
    #     if wntag is None:  # not supply tag in case of None
    #         lemma = lemmatizer.lemmatize(word)
    #     else:
    #         lemma = lemmatizer.lemmatize(word, pos=wntag)
    #     clean_tokens.append(lemma)
    return clean_tokens

def get_wordnet_pos(treebank_tag):
    """
    Convert Treebank tags to Wordnet tags
    :param treebank_tag: Treebank tag
    :return: Wordnet tag or None
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def build_model():
    """
    Pipeline model with grid search to tune parameters
    :return: Model
    """
    pipeline = Pipeline([
        ('vec', CountVectorizer(strip_accents='unicode', tokenizer=tokenize,
                                max_features=5000, stop_words=stopwords.words('english'))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])
    parameters = {
        'vec__max_features': [2000, 5000, 7000],
        'vec__max_df': [0.8, 0.9],
        'tfidf__smooth_idf': [True, False]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predict targets for test data and evaluate model performance through classification report
    :param model: Trained model
    :param X_test: Features for test data
    :param Y_test: Actual targets for test data
    :param category_names: Categories in data
    :return: None
    """
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=Y_test.columns)
    print(classification_report(np.hstack(Y_test.values),np.hstack(Y_pred.values)))


def save_model(model, model_filepath):
    """
    Pickle trained model
    :param model: Trained model
    :param model_filepath: Location to save pickled model
    :return: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()