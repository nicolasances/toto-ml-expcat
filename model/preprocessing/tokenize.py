import re

def tokenize_description(desc, punctuation_symbols, stemmer):
    """Tokenizes a description and applies the following:
    - removes useless characters (e.g. '-')
    - stems the word
    """
    # Split descriptions into tokens
    tokens = desc.split()
    
    # Prepare to remove any punctuation in the word
    translation_table = str.maketrans('', '', ''.join(punctuation_symbols))
    
    # Filter out punctuation, pure digits, remove numeric characters from the word
    tokens = [re.sub(r'\d', '', word.lower().translate(translation_table)) for word in tokens if word not in punctuation_symbols and not word.isdigit()]
    
    # Perform stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Join the tokens, as needed by TF-IDF
    tokens = " ".join(tokens)
    
    return tokens