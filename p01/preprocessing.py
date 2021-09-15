import spacy # spacy version == 3.1.2, wrote in requirements.txt


class EnglishPreprocess:
    """
    Preprocess English Texts using spacy
    It tokenize text based mostly on space ' ' and punctuation marks
    's is treated as an individual token
    Then, it'll delete stopwords and punctuation marks then return
    lemmatized data

    usage
    ep = EnglishPreprocess(text)
    preprocessed_text = ep.get_preprocessed_data()

    if you haven't installed en_core_web_sm
    run this command on Terminal

    $ python -m spacy download en_core_web_sm
    """
    def __init__(self, text):
        self.nlp = spacy.load('en_core_web_sm')
        self.spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
        self.text = self.nlp(text)

    def is_token_allowed(self, token):
        """
        check if the token is punctuation mark or stopword
        return True only if the token is not punctuation mark and stopword
        """
        if token.is_stop or token.is_punct:
            return False
        return True

    def preprocess_token(self, token):
        """
        return lowercased lemmatized token
        """
        return token.lemma_.strip().lower()

    def get_preprocessed_data(self):
        filtered_tokens = [self.preprocess_token(token) for token in self.text if self.is_token_allowed(token)]
        return filtered_tokens