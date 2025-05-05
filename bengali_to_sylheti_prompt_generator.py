import json
import csv
import argparse
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_fewshot_examples(json_path: str) -> List[Dict]:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        logger.info(f"Successfully loaded {len(examples)} few-shot examples from {json_path}")
        return examples
    except Exception as e:
        logger.error(f"Error loading few-shot examples: {e}")
        return []

def load_glossary(csv_path: str) -> List[Dict[str, str]]:
    glossary = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'ben' in row and 'syl' in row:
                    glossary.append({
                        'bangla': row['ben'].strip(),
                        'sylheti': row['syl'].strip()
                    })
        logger.info(f"Successfully loaded {len(glossary)} glossary entries from {csv_path}")
        return glossary
    except Exception as e:
        logger.error(f"Error loading glossary: {e}")
        return []

def tokenize_sentence(sentence: str) -> List[str]:
    # Remove punctuation and split by whitespace
    words = re.findall(r'[\u0980-\u09FF]+', sentence)
    return words

def filter_relevant_glossary(input_sentence: str, glossary: List[Dict[str, str]]) -> List[Dict[str, str]]:
    input_tokens = set(tokenize_sentence(input_sentence.lower()))
    seen = set()
    unique_entries = []

    for entry in glossary:
        bangla_word = entry['bangla'].strip().lower()
        if (bangla_word in input_tokens or bangla_word in input_sentence.lower()) and bangla_word not in seen:
            unique_entries.append(entry)
            seen.add(bangla_word)

    logger.info(f"Filtered glossary from {len(glossary)} to {len(unique_entries)} unique relevant entries")
    return unique_entries

def select_fewshot_examples(
    input_sentence: str, 
    examples: List[Dict], 
    num_examples: int = 5
) -> List[Dict]:

    if len(examples) <= num_examples:
        logger.info(f"Using all {len(examples)} available examples (requested {num_examples})")
        return examples
    
    # Simple word overlap similarity
    input_tokens = set(tokenize_sentence(input_sentence.lower()))
    
    # Calculate similarity scores
    scored_examples = []
    for example in examples:
        example_tokens = set(tokenize_sentence(example['bangla_speech'].lower()))
        # Jaccard similarity: intersection over union
        similarity = len(input_tokens.intersection(example_tokens)) / max(1, len(input_tokens.union(example_tokens)))
        scored_examples.append((similarity, example))
    
    # Sort by similarity (highest first) - using the first element in each tuple (the similarity score)
    scored_examples.sort(key=lambda x: x[0], reverse=True)
    
    # Take top examples, but also include some random ones for diversity
    top_examples = [ex for _, ex in scored_examples[:max(1, int(num_examples * 0.6))]]
    
    # Get remaining examples
    remaining = [ex for _, ex in scored_examples[max(1, int(num_examples * 0.6)):]]
    # Select random examples from remaining
    random_examples = random.sample(
        remaining, 
        min(len(remaining), num_examples - len(top_examples))
    )
    
    selected_examples = top_examples + random_examples
    random.shuffle(selected_examples)  # Shuffle to avoid bias
    
    logger.info(f"Selected {len(selected_examples)} few-shot examples")
    return selected_examples

def construct_prompt(
    input_sentence: str,
    selected_examples: List[Dict],
    relevant_glossary: List[Dict[str, str]]
) -> str:
  
    # System instructions
    system_instructions = """You are an expert translator from Bengali (Bangla) to Sylheti. 
            Follow these guidelines for translation:
            1. Preserve the original meaning and intent of the Bengali sentence
            2. Use appropriate Sylheti vocabulary and grammar patterns
            3. Respect idioms and cultural expressions specific to Sylheti
            4. Follow the few-shot examples provided below
            5. Utilize the glossary entries when appropriate
            6. Maintain the tone and register of the original sentence

            Translate the Bengali text to natural, fluent Sylheti."""

    # Format few-shot examples
    examples_text = "Few-shot examples:\n"
    for i, example in enumerate(selected_examples, 1):
        examples_text += f"{i}. Bangla: {example['bangla_speech']}\n   Sylheti: {example['sylhet_bangla_speech']}\n\n"
    
    # Format glossary entries
    glossary_text = "Glossary:\n"
    for i, entry in enumerate(relevant_glossary, 1):
        glossary_text += f"{i}. Bangla: {entry['bangla']} → Sylheti: {entry['sylheti']}\n"
    
    # Format the input
    input_text = f"Bangla: {input_sentence}\nSylheti:"
    
    # Combine all parts
    full_prompt = f"{system_instructions}\n\n{examples_text}\n{glossary_text}\n\n{input_text}"
    
    return full_prompt

def generate_prompt(
    input_sentence: str, 
    glossary_csv_path: str, 
    fewshot_json_path: str,
    num_examples: int = 5
) -> str:
    """
    Generate a complete prompt for Bengali to Sylheti translation.
    
    Args:
        input_sentence: Bengali sentence to translate
        glossary_csv_path: Path to the glossary CSV file
        fewshot_json_path: Path to the few-shot examples JSON file
        num_examples: Number of few-shot examples to include
        
    Returns:
        Complete prompt ready for an LLM
    """
    # Load data
    glossary = load_glossary(glossary_csv_path)
    fewshot_examples = load_fewshot_examples(fewshot_json_path)
    
    # Process data
    relevant_glossary = filter_relevant_glossary(input_sentence, glossary)
    selected_examples = select_fewshot_examples(input_sentence, fewshot_examples, num_examples)
    
    # Create prompt
    prompt = construct_prompt(input_sentence, selected_examples, relevant_glossary)
    
    return prompt

def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(description='Generate prompts for Bengali to Sylheti translation')
    parser.add_argument('--input', '-i', type=str, required=True, help='Bengali sentence to translate')
    parser.add_argument('--glossary', '-g', type=str, required=True, help='Path to glossary CSV file')
    parser.add_argument('--fewshot', '-f', type=str, required=True, help='Path to few-shot examples JSON file')
    parser.add_argument('--examples', '-e', type=int, default=5, help='Number of few-shot examples to include')
    parser.add_argument('--output', '-o', type=str, help='Output file for the prompt (prints to stdout if not specified)')
    
    args = parser.parse_args()
    
    prompt = generate_prompt(
        args.input, 
        args.glossary, 
        args.fewshot,
        args.examples
    )
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(prompt)
        print(f"Prompt written to {args.output}")
    else:
        print(prompt)

# Direct execution without CLI when run from VS Code
if __name__ == "__main__":
    # Configure these variables for your use case
    input_sentence = "মালিক ১৯৫৬-১৯৬৬ সালের মধ্যে করাচি, পাঞ্জাব ও রাওয়ালপিন্ডির হয়ে মোট ৪৯টি প্রথম শ্রেণির ম্যাচ খেলেন"  # "I can speak Bengali."
    glossary_csv_path = "data/ONUBAD/glossary.csv"  
    fewshot_json_path = "data/Vashantor_Json_Format/Train/Sylhet Train Translation.json"
    num_examples = 5
    
    # Generate the prompt
    output_prompt = generate_prompt(
        input_sentence,
        glossary_csv_path,
        fewshot_json_path,
        num_examples
    )
    
    # Print the prompt
    print(output_prompt)
    
    # output_file = "output_prompt.txt"
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     f.write(output_prompt)
    # print(f"Prompt written to {output_file}")