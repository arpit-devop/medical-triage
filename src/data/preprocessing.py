import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

class DataPreprocessor:
    def __init__(self, max_seq_length=128, max_words=10000):
        self.max_seq_length = max_seq_length
        self.max_words = max_words
        self.lstm_tokenizer = Tokenizer(num_words=self.max_words, oov_token="<OOV>")
        # Lazy-loaded: only instantiated when prepare_bert_data() is called
        self._bert_tokenizer = None

    @property
    def bert_tokenizer(self):
        """Lazy-load the BioBERT tokenizer only when it is actually needed."""
        if self._bert_tokenizer is None:
            from transformers import AutoTokenizer
            self._bert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        return self._bert_tokenizer

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove stopwords
        text = " ".join([word for word in text.split() if word not in STOPWORDS])
        return text

    def prepare_lstm_data(self, texts, labels=None):
        cleaned_texts = [self.clean_text(t) for t in texts]
        self.lstm_tokenizer.fit_on_texts(cleaned_texts)
        sequences = self.lstm_tokenizer.texts_to_sequences(cleaned_texts)
        padded = pad_sequences(sequences, maxlen=self.max_seq_length, padding='post', truncating='post')
        
        if labels is not None:
            # Need to one-hot encode or label encode labels before this
            return padded, np.array(labels)
        return padded

    def prepare_bert_data(self, texts, labels=None):
        cleaned_texts = [self.clean_text(t) for t in texts]
        encodings = self.bert_tokenizer(
            cleaned_texts,
            truncation=True,
            padding=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        
        if labels is not None:
            import torch
            return encodings, torch.tensor(labels)
        return encodings

if __name__ == "__main__":
    # Test preprocessor
    test_texts = [
        "Patient presents with severe chest pain and radiating pain in the left arm.",
        "Experiencing chronic migraines and blurry vision for 3 days."
    ]
    processor = DataPreprocessor()
    lstm_inputs = processor.prepare_lstm_data(test_texts)
    print("LSTM Shape:", lstm_inputs.shape)

    # BioBERT tokenizer is only loaded if this line is reached
    bert_inputs = processor.prepare_bert_data(test_texts)
    print("BERT Input IDs Shape:", bert_inputs['input_ids'].shape)
