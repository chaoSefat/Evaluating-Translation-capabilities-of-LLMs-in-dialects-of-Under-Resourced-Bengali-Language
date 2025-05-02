import pandas as pd
import re
import json

def read_glossary_csv():
    """
    Reads the glossary CSV file and returns a dictionary mapping Bangla words to their syllabic representations.
    """
    glossary_df = pd.read_csv("data/ONUBAD/glossary.csv")
    glossary_dict = dict(zip(glossary_df["ben"].str.strip(), glossary_df["syl"].str.strip()))
    return glossary_dict

    

def tokenize_bangla(sentence):
    return re.findall(r'[\u0980-\u09FF]+', sentence)

def extract_glossary_subset(tokens, glossary_dict):
    matched = {word: glossary_dict[word] for word in tokens if word in glossary_dict}
    return matched

if __name__ == "__main__":
    glossary_dict = read_glossary_csv()
    sentence = "আমি বাংলায় গান গাই"
    tokens = tokenize_bangla(sentence)
    print(f"Tokens: {tokens}") # for debuging
    glossary_subset = extract_glossary_subset(tokens, glossary_dict)
    print(f"Glossary: {glossary_subset}")
    #print("আমি বাংলায় গান গাই")