import numpy as np

def create_vocab(list_of_tokens): 
    """Creates a vocabulary dict of the list of tokens used in the training data. 
    
    The dictionnary is an index dictionnary, that assign to each word an index. 
    The dictionnary is meant to be used for "one-hot encoding" or similar
    
    Returns
     - (dict) where the key is the word and the value is the index
    """
    
    vocab = {}
    
    idx = 1
    for l in list_of_tokens:
        for token in l: 
            if token not in vocab.keys(): 
                vocab[token] = idx
                idx += 1
                
    # Add the UNK word
    vocab["UNK"] = 0
    
    return vocab


def custom_encode(words: list, vocab: dict, unknown_word: str = "UNK"): 
    """Creates an encoding for the provided sentence (words)
    
    Not really an embedding, but more like a type of one-hot-encoding.
    It's basically creating a vector of size (vocab_size,) where each item (vocabulary word) is marked as 1 or 0 if it's present in the provided words.
    The number could be > 1 if the words list contains multiple times the same word.

    Args:
        words (list): list of stemmed and preprocessed words
        vocab (dict): vocabulary as a dictionnary where the key is the word and the value is the index of the word
        unknown_word (str, optional): special word to be used for unknown words. Defaults to "UNK".

    Returns:
        list: returns the encoded words as a single vector os size (len(vocab),). Returned as a list.
    """
    
    encoded_words = np.zeros(len(vocab))
    
    # Create an encoding of the words
    for word in words: 
        if word in vocab.keys():
            idx = vocab[word]
        else:
            idx = vocab[unknown_word]
        
        encoded_words[idx] = 1
    
    return encoded_words.tolist()