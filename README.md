# BT3102 Project
BT3102 Optimization methods, Viterbi's algorithm implementation
Monte-carlo, Graph and flow methods

Running Final.py to obtain probabilities of possible word POS (Part of speech) tags for each word
Cleaning the data to obtain probabilities of each word's most likely POS tag
Using Viterbi algorithm:

It relies on dynamic programming to efficiently compute the most probable sequence of hidden states (POS tags).
For each word in the sentence, it calculates the most probable POS tag based on the previous word's tag (using transition probabilities) and the likelihood of the word being associated with that tag (using emission probabilities).
