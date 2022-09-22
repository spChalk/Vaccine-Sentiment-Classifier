# :syringe: Vaccine-Sentiment-Classifier

Vaccine Sentiment Classifier (or **VSC**) is a Deep Learning classifier trained on real world **twitter data**.

VSC distinguishes 3 types of tweets:
1. :neutral_face: Neutral
2. :angry: Anti-vax 
3. :relaxed: Pro-vax

The main aim of the project, is to showcase the multitude of ways (from shallow to deep learning) one can use NLP in order to extract sentiment from a given set.

## Structure
The project is divided into 4 different implementations. 

Each of them includes a `.ipynb` notebook and its corresponding documentation.

In a nutshell, the most important topics of each implementation are listed below.

### 1. VSC using Softmax Regression
- Data Cleaning, preprocessing, visualization
- [n-grams](https://en.wikipedia.org/wiki/N-gram)
- [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- Count vectorizer
- Hashing vectorizer
- Softmax regressor
- Hyperparameter tuning, using evolutionary algorithm: [GASearchCV](https://sklearn-genetic-opt.readthedocs.io/en/stable/api/gasearchcv.html)
- [Learning curves](https://en.wikipedia.org/wiki/Learning_curve) & [Classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)

### 2. VSC using Feed Forward Neural Networks (FFNN)
- Dealing with imbalanced data
- [*Optuna*](https://optuna.org/) hyperparameter tuner
- FFNN using [Bag of Words (BoW)](https://en.wikipedia.org/wiki/Bag-of-words_model), [Term Frequency–Inverse Document Frequency (TF-IDF)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), [Word embeddings](https://en.wikipedia.org/wiki/Word_embedding)
- [GloVe](https://nlp.stanford.edu/projects/glove/)
- [RoC curve](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)

### 3. VSC using Recurrent Neural Networks (RNN)
- [Bidirectional RNN](https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks)
- [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) / [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit)
- [Gradient clipping](https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem)
- [Skip Connections](https://www.analyticsvidhya.com/blog/2021/08/all-you-need-to-know-about-skip-connections/)
- [Attention mechanism](https://arxiv.org/pdf/1706.03762.pdf)

### 4. VSC using BERT
- [Huggingface BERT moidels](https://huggingface.co/docs/transformers/model_doc/bert)
