import pandas as pd
import re
import math
import string
from collections import defaultdict

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score

import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt
import seaborn as sns

# Descargar los recursos necesarios
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class SentimentModel:
    def __init__(self, n=3, skip=2, remove_punctuation=False, remove_stopwords=False, lemmatize=False, use_entropy=True, apply_weighting=True, entropy_threshold=0.3):
        # Asignar los mejores hiperparámetros como valores predeterminados
        self.n = n
        self.skip = skip
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.use_entropy = use_entropy
        self.apply_weighting = apply_weighting
        self.entropy_threshold = entropy_threshold
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        # Asegurarse de que 'text' es una cadena
        text = str(text)

        if self.remove_punctuation:
            punctuation = re.escape(string.punctuation) + '¡' + ',' + ';'
            text = re.sub(f'[{punctuation}]', '', text)

        tokens = word_tokenize(text)

        if self.remove_stopwords:
            stop_words = set(stopwords.words("english"))
            tokens = [word for word in tokens if word.lower() not in stop_words]

        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return " ".join(tokens)

    def get_ngrams(self, text):
        """
        Genera n-gramas o skip-gramas a partir de un texto.
        """
        tokens = word_tokenize(text)
        if self.skip == 0:
            return list(ngrams(tokens, self.n))
        else:
            return list(nltk.skipgrams(tokens, self.n, self.skip))

    def calculate_ngram_frequencies(self, texts, weight=1):
        """
        Calcula las frecuencias de los n-gramas en un conjunto de textos, ponderadas por el parámetro 'weight'.
        """
        ngram_values = defaultdict(int)
        for text in texts:
            ngrams_in_text = self.get_ngrams(text)
            for ngram in ngrams_in_text:
                ngram_values[ngram] += weight  # Aplicar ponderación
        return ngram_values

    def calculate_entropy(self, pos_freq, neg_freq):
        """
        Calcula la entropía de un n-grama dado sus frecuencias en textos positivos y negativos.
        """
        total_freq = pos_freq + neg_freq
        if total_freq == 0:
            return 0

        p_pos = pos_freq / total_freq if pos_freq > 0 else 0
        p_neg = neg_freq / total_freq if neg_freq > 0 else 0

        entropy = 0
        if p_pos > 0:
            entropy -= p_pos * math.log2(p_pos)
        if p_neg > 0:
            entropy -= p_neg * math.log2(p_neg)

        return entropy

    def apply_class_weighting(self, positive_texts, negative_texts):
        """
        Realiza la ponderación de la frecuencia de aparición de cada n-grama dependiendo de su grupo
        (positivo o negativo) y aplica un filtro de entropía si es necesario.
        """
        # Número de textos positivos y negativos
        num_positive = len(positive_texts)
        num_negative = len(negative_texts)

        # Peso que tendrá la aparición de un n-grama en cada grupo
        weight_positive = 1 / num_positive if self.apply_weighting and num_positive > 0 else 1
        weight_negative = 1 / num_negative if self.apply_weighting and num_negative > 0 else 1

        # Calcular las frecuencias de n-gramas en textos positivos y negativos, ponderadas por su tamaño
        positive_ngram_values = self.calculate_ngram_frequencies(positive_texts, weight_positive)
        negative_ngram_values = self.calculate_ngram_frequencies(negative_texts, weight_negative)

        combined_ngram_values = defaultdict(lambda: {"freq": 0, "sentiment": ""})

        # Unir los n-gramas de textos positivos y negativos
        for ngram in set(positive_ngram_values.keys()).union(set(negative_ngram_values.keys())):
            pos_freq = positive_ngram_values.get(ngram, 0)
            neg_freq = negative_ngram_values.get(ngram, 0)

            # Aplicar filtro de entropía si es necesario
            if self.use_entropy:
                entropy_value = self.calculate_entropy(pos_freq, neg_freq)
                if entropy_value >= self.entropy_threshold:
                    continue  # Ignorar n-gramas ambiguos con entropía alta

            # Sumar las frecuencias y asignar el sentimiento dominante
            combined_freq = pos_freq + neg_freq
            sentiment = 1 if pos_freq > abs(neg_freq) else 0

            combined_ngram_values[ngram]["freq"] = combined_freq
            combined_ngram_values[ngram]["sentiment"] = sentiment

        return combined_ngram_values

    def fit(self, X_train, y_train):
        """
        "Entrenar" el modelo utilizando los datos de entrenamiento. Esto ajusta las frecuencias
        de los n-gramas a partir de los textos positivos y negativos.
        """
        # Limpiar los textos de entrenamiento
        X_train_clean = X_train.apply(self.clean_text)

        # Separar los textos positivos y negativos basados en las etiquetas y_train
        positive_texts = X_train_clean[y_train == 1]
        negative_texts = X_train_clean[y_train == 0]

        # Ajustar los n-gramas para los textos de entrenamiento
        self.ngram_analysis = self.apply_class_weighting(positive_texts, negative_texts)

    def predict(self, X_test):
        """
        Predice la polaridad de un conjunto de datos preprocesado utilizando los n-gramas ajustados durante el entrenamiento.
        """
        X_test_clean = X_test.apply(self.clean_text)

        polarities = []
        for text in X_test_clean:
            ngrams_in_text = self.get_ngrams(text)

            score = 0
            for ngram in ngrams_in_text:
                if ngram in self.ngram_analysis:
                    sentiment = self.ngram_analysis[ngram]["sentiment"]
                    relative_freq = self.ngram_analysis[ngram]["freq"]

                    # Asignar puntaje basado en la frecuencia relativa y el sentimiento del n-grama
                    if sentiment == 1:
                        score += relative_freq
                    elif sentiment == 0:
                        score -= relative_freq

            # Decidir la polaridad del texto en función del puntaje
            polarity = 1 if score > 0 else 0
            polarities.append(polarity)

        return polarities
