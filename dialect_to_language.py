#### IMPORTS ####
import json
from openai import OpenAI
import os
from typing import List, Dict
from dotenv import load_dotenv 

######### CONFIGURATION ############
#########
dialect = "Sylheti"
dialect_dir = "Sylhet"
split = "Test"
language = "English"
method = "Zero-Shot"
model="gpt-4.1-mini"
#method = "Few-Shot"
######### CONFIGURATION ############
#########
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print ("Error: OpenAI API Key not provided.")

GPT_client = OpenAI(api_key=api_key)

def load_data(dialect_dir, split):
    """
    Load the data from the specified language and split.
    """
    with open(f"data/Vashantor_Json_Format/{split}/{dialect_dir} {split} Translation.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def zero_shot_prompt(dialect, language, sentence):

    sys_prompt = f"""
You are an expert translator from {dialect} dialect of Bengali to standard {language} language.
            Follow these guidelines for translation:
            1. Preserve the original meaning and intent of the {dialect} sentence
            2. Use appropriate vocabulary and grammar patterns
            3. Respect idioms and cultural expressions
            4. Maintain the tone and register of the original sentence

            Translate the {dialect} text to natural, fluent {language}. Provide the translation in {language} without any additional commentary or explanation or artifacts
            """
    user_prompt = f"""
            {dialect}: {sentence}
            {language}:
            """
    return sys_prompt, user_prompt

#print(zero_shot_prompt(dialect, language, "হে সিলেট থাকি আমার লগে দেখা করাত আইছিল"))

def OpenAI_translate(sys_prompt, user_prompt, model):
    client = GPT_client
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ], temperature=0
    )
    translation = response.choices[0].message.content.strip()
    return translation

def process_file(data: List[Dict], dialect, language, method, model):
    """
    Process the data file and translate each sentence.
    """
    translations = []
    input_sentence_key = f"{dialect_dir.lower()}_bangla_speech"


    i = 1
    for item in data:
        i = i + 1 # debugging purposes
        if i == 5:
            break
        new_item = item.copy()
        sentence = item[input_sentence_key]
        if method == "Zero-Shot":
            sys_prompt, user_prompt = zero_shot_prompt(dialect, language, sentence)
            translation = OpenAI_translate(sys_prompt, user_prompt, model)
            new_item["translation"] = translation
        elif method == "Few-Shot":
            # Implement Few-Shot logic here later to do
            pass
        else:
            raise ValueError("Invalid method specified. Use 'Zero-Shot' or 'Few-Shot'.")
        
        translations.append(new_item)
        print(f"Processed {i} sentences: {sentence} -> {new_item['translation']}")
        
        # Debugging purposes
    
    
    
    
    output_file_name = f"output/{dialect_dir}_{split}_{language}_{method}_{model}_Translation.json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, 'w', encoding='utf-8') as f:
        json.dump(translations, f, ensure_ascii=False, indent=4)
    return translations



####### Playground #######

data = load_data(dialect_dir, split)
translations = process_file(data, dialect, language, method, model)
print(f"Translations saved to output/{dialect_dir}_{split}_{language}_{method}_{model}_Translation.json")