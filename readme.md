I'm using this as a place to put machine learning algorithms that I've reimplemented myself for a combination of fun and education.

# Organization

Each algorithm has its own folder and inside there is generally one file for the model, one to load data, and one to train the model. If there is a subfolder `data` included, then that means that you can run the train file and expect the algorithm to run as anticipated.

# Data

Some of these were done independently, and thus the data was probably pulled from the UCI database or some comparable open-source platform. Some of these were done with the dual purpose of being used for my university classwork. In that case, often there is no `data` subfolder for an algorithm, and/or the model training file is missing. That's probably because I wasn't comfortable putting it in there because of uncertainty in who owns tge data or because I thought including the whole solution would be a breach of academic integrity (again, mostly only a concern for algorithms that were implemented to serve the dual purpose as sufficing for my university coursework).

# Algorithms Implemented

* Support Vector Machines (svm)
* Gaussian Mixture Model (for clustering and also used as a Bayes' Classifier)
* Non-negative matrix factorization for reccomendation systems
* Hidden Markov Model
* Linear Regression
