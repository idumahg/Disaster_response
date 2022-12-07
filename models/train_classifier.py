"""Code to implement the ETL Pipeline."""
import sys
from sqlalchemy.engine import create_engine
import pandas as pd
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    Load data from SQL database.

    Parameters:
        database_filepath: SQL database filepath

    Returns:
        X: feature variables
        y: target variables
    """
    df = pd.read_sql_table("DisasterResponseTable",
                           'sqlite:///' + database_filepath)
    X = df.message
    y = df[list(set(df.columns) - set(df[['id', 'message', 'genre',
                                          'original']]))]
    category_names = y.columns

    return X, y, category_names


def tokenize(text):
    """
    Tokenize each of the text data in feature df.

    Parameters:
        text: text data

    Returns:
        clean_tokens: the clean token
    """
    tokens = word_tokenize(text)
    tokens = [tok for tok in tokens if tok not in stop_words]
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build the machine learning pipeline.

    Returns:
        cv: trained model
    """
    clf = RandomForestClassifier()

    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('multi', MultiOutputClassifier(clf))])

    parameters = {
        'multi__estimator__n_estimators': [100, 200, 300],
        'multi__estimator__min_samples_split': [2, 3, 5]}

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=5,
                      verbose=3, cv=3)

    return cv


def evaluate_model(model, x_test, y_test, category_names):
    """
    Evaluate model on test set.

    Parameters:
        model: trained model
        x_test: feature test data
        y_test: target test data
        category_names: names of columns in y_test

    Returns:
        None
    """
    y_pred = model.predict(x_test)
    y_pred = pd.DataFrame(y_pred, columns=category_names)

    print(classification_report(y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save trained model.

    Parameters:
        model: trained model
        model_filepath: directory to save the model

    Return:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """Script to run."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
