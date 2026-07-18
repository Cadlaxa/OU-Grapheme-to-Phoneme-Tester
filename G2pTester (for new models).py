import onnxruntime as ort
import numpy as np
from collections import defaultdict
import traceback
import sys
import ruamel.yaml
from pathlib import Path as P
from colorama import Fore, Style, init
import re

init(autoreset=True)

class G2pTester:
    def __init__(self, config_path='config.yaml'):
        self.lock = None  # Placeholder for thread safety if needed
        self.dict = {}
        self.grapheme_indexes = {}
        self.pred_cache = defaultdict(list)
        self.session = None
        self.config_path = config_path 
        self.split_character = False
        self.load_config()
        self.prompt_for_grapheme()

    def load_config(self):
        yaml = ruamel.yaml.YAML()
        encodings = ['utf-8', 'utf-16', 'utf-8-sig', 'shift-jis', 'iso-8859-1']
        config = None
        for encoding in encodings:
            try:
                with open(self.config_path, 'r', encoding=encoding) as f:
                    config = yaml.load(f)
                break 
            except Exception as e:
                continue 
        if config is None:
            print(Fore.RED + "Failed to load configuration file with supported encodings.")
            sys.exit(1)

        print(Fore.LIGHTGREEN_EX + "Available models:")
        for i, model in enumerate(config):
            print(Fore.CYAN + f"{i + 1}: {model['model_name']}")

        while True:
            try:
                model_index = int(input(Fore.YELLOW + "Select a model by number: \n")) - 1
                if model_index < 0 or model_index >= len(config):
                    print(Fore.RED + "Invalid selection. Please enter a valid model number.")
                else:
                    selected_model = config[model_index]
                    break
            except ValueError:
                print(Fore.RED + "Invalid input. Please enter a number.")

        # Load selected model's configuration
        self.graphemes = selected_model['graphemes']
        self.phonemes = selected_model['phonemes'][4:]  # Skip initial placeholder phonemes
        self.dict_path = selected_model['dict_path']
        self.onnx_path = selected_model['onnx_path']
        self.split_character = selected_model.get('split_character', False)
        self.load_pack()  # Load dictionary and model for the selected configuration

    def load_pack(self):
        loaded_dict = False
        loaded_model = False
        encodings = ['utf-8', 'utf-16', 'utf-8-sig', 'shift-jis', 'iso-8859-1']

        # Ensure dict_paths is always a list
        dict_paths = self.dict_path if isinstance(self.dict_path, list) else [self.dict_path]

        # Load dictionary from multiple files
        for dict_path_str in dict_paths:
            dict_path = P(dict_path_str)
            if dict_path.exists():
                for encoding in encodings:
                    try:
                        with open(dict_path, 'r', encoding=encoding) as f:
                            for line in f:
                                parts = re.split(r'\t+| {2,}', line.strip(), maxsplit=1)
                                if len(parts) >= 2:
                                    grapheme = parts[0].lower()
                                    phoneme_parts = parts[1:]
                                    phonemes = ''.join(phoneme_parts).lower()
                                    self.dict[grapheme] = phonemes.split()
                                    loaded_dict = True
                                else:
                                    print(Fore.YELLOW + f"Ignoring line: {line.strip()}")
                        break
                    except Exception as e:
                        print(Fore.YELLOW + f"Failed to load {dict_path} with encoding {encoding}. Trying next encoding.")
                        continue 
            else:
                print(Fore.RED + f"Dictionary file not found: {dict_path}")

        # Check if the ONNX model path is provided and not "null"
        if self.onnx_path != "null":
            try:
                print(Fore.GREEN + f"\nLoading ONNX model from: {self.onnx_path}")
                self.session = ort.InferenceSession(P(self.onnx_path))
                loaded_model = True
            except Exception as e:
                print(Fore.RED + f"\nFailed to load ONNX model: {e}")

        # Error handling for missing resources
        if not loaded_dict and self.onnx_path == "null":
            print(Fore.RED + "Both dictionary and ONNX model paths are null. Exiting...")
            sys.exit(1)
        elif not loaded_dict:
            print(Fore.YELLOW + "No valid dictionary files found, but ONNX model will be used.")
        elif not loaded_model:
            print(Fore.YELLOW + "No valid ONNX model loaded, but dictionary will be used.")

        # Create grapheme indexes (skip the first four graphemes)
        self.grapheme_indexes = {g: i + 4 for i, g in enumerate(self.graphemes[4:])}

    def split_input(self, input_text):
        is_roman = re.compile(r'[A-Za-z]')
        is_vowel = re.compile(r'[aeiouAEIOU]')
        result = []
        i = 0
        while i < len(input_text):
            # Check for multi-character Hiragana/Katakana graphemes
            found = False
            for length in range(3, 1, -1):  # Check for 3, 2 character graphemes
                if i + length <= len(input_text) and input_text[i:i+length] in self.dict:
                    result.append(input_text[i:i+length])
                    i += length
                    found = True
                    break
            if found:
                continue
            if is_roman.match(input_text[i]):
                if i + 1 < len(input_text):
                    if is_roman.match(input_text[i + 1]):
                        if is_vowel.match(input_text[i]) and is_vowel.match(input_text[i + 1]):
                            result.append(input_text[i])
                            i += 1
                        elif is_vowel.match(input_text[i + 1]):
                            result.append(input_text[i:i+2])
                            i += 2
                        else:
                            result.append(input_text[i])
                            i += 1
                    else:
                        result.append(input_text[i])
                        i += 1
                else:
                    result.append(input_text[i])
                    i += 1
            else:
                result.append(input_text[i])
                i += 1
        return ' '.join(result)

    def prompt_for_grapheme(self):
        print(Fore.GREEN + "Enter graphemes for phoneme prediction (type 'exit' to quit or 'change' to switch models):")
        while True:
            grapheme_input = input(Fore.YELLOW + "Grapheme: ").strip()
            if grapheme_input.lower() == 'exit':
                print(Fore.CYAN + "Exiting...")
                break
            elif grapheme_input.lower() == 'change':
                print(Fore.CYAN + "Switching model...\n")
                self.load_config()
            else:
                # Use split_input if enabled
                if self.split_character:
                    grapheme_input = self.split_input(grapheme_input)
                phonemes = self.predict(grapheme_input)
                print(Fore.GREEN + f"Phonemes for '{grapheme_input}': [{phonemes}]")

    def predict(self, input_text):
        words = input_text.strip().split()
        predicted_phonemes = []
        for word in words:
            word_lower = word.lower()
            if word_lower in self.dict:
                predicted_phonemes.append(' '.join(self.dict[word_lower]))
            else:
                cached_phoneme = self.pred_cache.get(word_lower)
                if cached_phoneme:
                    predicted_phonemes.append(' '.join(cached_phoneme))
                else:
                    predicted_phoneme = self.predict_with_model(word)
                    if predicted_phoneme:
                        predicted_phonemes.append(predicted_phoneme)
                        self.pred_cache[word_lower] = predicted_phoneme.split()  # Cache the result
                    else:
                        print(Fore.RED + f"Error in prediction for '{word}'")
                        predicted_phonemes.append("N/A")
        return '] ['.join(predicted_phonemes)

    def predict_with_model(self, word):
        # 1. Convert word characters to grapheme IDs
        # Use UNK_IDX (1) if a character isn't in the dictionary
        input_ids = [self.grapheme_indexes.get(c, 1) for c in word]
        
        # 2. Format as a 2D numpy array: shape (1, Sequence Length)
        src = np.array([input_ids], dtype=np.int32)

        try: 
            # 3. Run the modern end-to-end model (it only needs 'src' as input!)
            ort_inputs = {'src': src}
            outputs = self.session.run(None, ort_inputs)
            tgt = outputs[0] # Returns an array of predicted IDs

            # 4. Convert output IDs back to phoneme strings
            predicted_phonemes = []
            for id in tgt.flatten():
                if id > 3: # Skip special tokens (0=pad, 1=unk, 2=bos, 3=eos)
                    predicted_phonemes.append(self.phonemes[id - 4])
                    
            return ' '.join(predicted_phonemes)
            
        except Exception as e:
            print(Fore.RED + "Error in prediction:", traceback.format_exc())
            return ""

G2pTester()