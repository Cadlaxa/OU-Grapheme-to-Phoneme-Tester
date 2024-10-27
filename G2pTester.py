import onnxruntime as ort
import numpy as np
from collections import defaultdict
import traceback
import sys
import ruamel.yaml
from pathlib import Path as P
from colorama import Fore, Style, init
import re  # Import regex module

init(autoreset=True)  # Initialize colorama for automatic style reset

class G2pTester:
    def __init__(self, config_path='config.yaml'):
        self.lock = None  # Placeholder for thread safety if needed
        self.dict = {}
        self.grapheme_indexes = {}
        self.pred_cache = defaultdict(list)
        self.session = None
        self.config_path = config_path  # Store config path for reloading
        self.split_character = False  # Initialize split character option
        self.load_config()  # Initial model loading
        self.prompt_for_grapheme()  # Start grapheme loop

    def load_config(self):
        yaml = ruamel.yaml.YAML()
        encodings = ['utf-8', 'utf-16', 'utf-8-sig', 'shift-jis', 'iso-8859-1']
        config = None
        for encoding in encodings:
            try:
                with open(self.config_path, 'r', encoding=encoding) as f:
                    config = yaml.load(f)
                break  # Exit loop if successful
            except Exception as e:
                continue  # Try the next encoding if it fails

        if config is None:
            print(Fore.RED + "Failed to load configuration file with supported encodings.")
            sys.exit(1)

        # Display available models for selection
        print(Fore.LIGHTGREEN_EX + "Available models:")
        for i, model in enumerate(config):
            print(Fore.CYAN + f"{i + 1}: {model['model_name']}")

        # Model selection with error handling
        while True:
            try:
                model_index = int(input(Fore.YELLOW + "Select a model by number: \n")) - 1
                if model_index < 0 or model_index >= len(config):
                    print(Fore.RED + "Invalid selection. Please enter a valid model number.")
                else:
                    selected_model = config[model_index]
                    break  # Exit loop once a valid model is selected
            except ValueError:
                print(Fore.RED + "Invalid input. Please enter a number.")

        # Load selected model's configuration
        self.graphemes = selected_model['graphemes']
        self.phonemes = selected_model['phonemes'][4:]  # Skip initial placeholder phonemes
        self.dict_path = selected_model['dict_path']
        self.onnx_path = selected_model['onnx_path']
        self.split_character = selected_model.get('split_character', False)  # Get split character option
        self.load_pack()  # Load dictionary and model for the selected configuration

    def load_pack(self):
        loaded_dict = False  # Track if any valid dictionary is loaded
        loaded_model = False  # Track if the ONNX model is loaded
        encodings = ['utf-8', 'utf-16', 'utf-8-sig', 'shift-jis', 'iso-8859-1']

        # Ensure dict_paths is always a list
        dict_paths = self.dict_path if isinstance(self.dict_path, list) else [self.dict_path]

        # Load dictionary from multiple files
        for dict_path_str in dict_paths:
            dict_path = P(dict_path_str)  # Convert to Path object for each path
            if dict_path.exists():
                for encoding in encodings:
                    try:
                        with open(dict_path, 'r', encoding=encoding) as f:
                            for line in f:
                                parts = line.strip().split('  ')
                                if len(parts) >= 2:
                                    grapheme = parts[0].lower()
                                    phoneme_parts = parts[1:]
                                    phonemes = ''.join(phoneme_parts).replace('0', '').replace('1', '').replace('2', '').replace('3', '').lower()
                                    self.dict[grapheme] = phonemes.split()
                                    loaded_dict = True
                                else:
                                    print(Fore.YELLOW + f"Ignoring line: {line.strip()}")
                        break  # Exit encoding loop if successful
                    except Exception as e:
                        print(Fore.YELLOW + f"Failed to load {dict_path} with encoding {encoding}. Trying next encoding.")
                        continue  # Try the next encoding
            else:
                print(Fore.RED + f"Dictionary file not found: {dict_path}")

        # Check if the ONNX model path is provided and not "null"
        if self.onnx_path != "null":
            try:
                print(Fore.GREEN + f"Loading ONNX model from: {self.onnx_path}")
                self.session = ort.InferenceSession(P(self.onnx_path))
                loaded_model = True
            except Exception as e:
                print(Fore.RED + f"Failed to load ONNX model: {e}")

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
                self.load_config()  # Reload model selection
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
        word_with_dash = "-" + word  # Adding a dash for skipped phoneme workaround
        input_ids = np.array([self.grapheme_indexes.get(c, 0) for c in word_with_dash], dtype=np.int32)
        input_length = len(input_ids)

        if len(input_ids.shape) == 1:
            input_ids = np.expand_dims(input_ids, axis=0)

        t = np.ones((1,), dtype=np.int32)
        src = input_ids
        tgt = np.array([2], dtype=np.int32)
        if len(tgt.shape) == 1:
            tgt = np.expand_dims(tgt, axis=0)

        try: 
            while t[0] < input_length and len(tgt) < 48:
                input_feed = {'src': src, 'tgt': tgt, 't': t}
                outputs = self.session.run(['pred'], input_feed)
                pred = outputs[0].flatten().astype(int)
                if pred != 2:
                    new_tgt_shape = (tgt.shape[0], tgt.shape[1] + 1)
                    new_tgt = np.zeros(new_tgt_shape, dtype=np.int32)
                    new_tgt[:, :tgt.shape[1]] = tgt
                    new_tgt[:, tgt.shape[1]] = pred
                    tgt = new_tgt
                else:
                    t[0] += 1

            predicted_phonemes = [self.phonemes[id - 4] for id in tgt.flatten().astype(int) if id != 2]
            predicted_phonemes_str = ' '.join(predicted_phonemes)
            return predicted_phonemes_str
        except Exception as e:
            print(Fore.RED + "Error in prediction:", traceback.format_exc())
            return ""

# Initialize and run
G2pTester()