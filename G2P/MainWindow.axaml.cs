using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Serilog;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace G2P;

public class ModelConfig {
    public string model_name {
        get;
        set;
    } = string.Empty;
    public List < string > graphemes {
        get;
        set;
    } = new();
    public List < string > phonemes {
        get;
        set;
    } = new();
    public object dict_path {
        get;
        set;
    } = string.Empty;
    public string onnx_path {
        get;
        set;
    } = string.Empty;
    public string phones_path {
        get;
        set;
    } = string.Empty;
    public bool split_character {
        get;
        set;
    }
}

public class AppSettings {
    public string ConfigPath {
        get;
        set;
    } = string.Empty;
    public string SelectedModelName {
        get;
        set;
    } = string.Empty;
    public bool SplitCharacter {
        get;
        set;
    }
}

public enum PhonemeCategory {
    Vowel,
    Stop,
    Affricate,
    Fricative,
    Nasal,
    Semivowel,
    Liquid,
    Tap,
    Aspirate,
    Separator,
    Unknown
}

public class PhonemeTextModel {
    public string Phoneme {
        get;
        set;
    } = string.Empty;
    public string TypeName {
        get;
        set;
    } = string.Empty;
    public IBrush Foreground {
        get;
        set;
    } = Brushes.Black;
}

public class NoteBoxModel {
    public string Word {
        get;
        set;
    } = string.Empty;
    public ObservableCollection < PhonemeTextModel > Phonemes {
        get;
        set;
    } = new();
}

public class PhonemeTypeLegendModel {
    public string TypeName {
        get;
        set;
    } = string.Empty;
    public IBrush BadgeColor {
        get;
        set;
    } = Brushes.Gray;
    public int Count {
        get;
        set;
    }
}

public partial class MainWindow: Window {
    private List < ModelConfig > _configs = new();
    private ModelConfig ? _activeConfig;
    private InferenceSession ? _session;
    private bool _isAutoregressive;
    private string _configDirectory = AppContext.BaseDirectory;
    private string _currentConfigPath = string.Empty;
    private bool _isInitializing = true;

    private readonly string _settingsFilePath = Path.Combine(AppContext.BaseDirectory, "settings.json");

    private readonly Dictionary < string, List < string >> _dictionary = new();
    private Dictionary < char, int > _graphemeIndexes = new();
    private List < string > _modelPhonemes = new();
    private readonly Dictionary < string, List < string >> _predCache = new();

    private readonly Dictionary < string, string > _modelPhoneTypes = new(StringComparer.OrdinalIgnoreCase);

    private readonly ObservableCollection < NoteBoxModel > _noteCollection = new();
    private readonly ObservableCollection < PhonemeTypeLegendModel > _legendCollection = new();

    public MainWindow() {
        InitializeComponent();

        PhonemesItemsControl.ItemsSource = _noteCollection;
        PhonemeTypesLegendControl.ItemsSource = _legendCollection;

        LoadSettingsAndInitialize();
        _isInitializing = false;
    }

    private void LoadSettingsAndInitialize() {
        AppSettings ? settings = LoadSettings();
        string configToLoad = string.Empty;

        if (settings != null && !string.IsNullOrEmpty(settings.ConfigPath) && File.Exists(settings.ConfigPath)) {
            configToLoad = settings.ConfigPath;
        } else {
            string defaultConfig = Path.Combine(AppContext.BaseDirectory, "config.yaml");
            if (File.Exists(defaultConfig)) {
                configToLoad = defaultConfig;
            }
        }

        if (!string.IsNullOrEmpty(configToLoad)) {
            LoadYamlConfig(configToLoad);

            if (settings != null && !string.IsNullOrEmpty(settings.SelectedModelName)) {
                var matchedModel = _configs.FirstOrDefault(m => m.model_name == settings.SelectedModelName);
                if (matchedModel != null) {
                    ModelComboBox.SelectedItem = matchedModel;
                }
            }

            if (settings != null) {
                SplitCharacterCheckBox.IsChecked = settings.SplitCharacter;
                if (_activeConfig != null) {
                    _activeConfig.split_character = settings.SplitCharacter;
                }
            }
        } else {
            StatusLog.Text = "No config file found. Click 'Browse Config...' to select one.";
            Log.Information("No default or saved config file found.");
        }
    }

    private AppSettings ? LoadSettings() {
        if (!File.Exists(_settingsFilePath)) return null;

        try {
            string json = File.ReadAllText(_settingsFilePath, Encoding.UTF8);
            var settings = JsonSerializer.Deserialize < AppSettings > (json);
            Log.Information("Loaded user settings from {SettingsPath}", _settingsFilePath);
            return settings;
        } catch (Exception ex) {
            Log.Error(ex, "Failed to load settings from {SettingsPath}", _settingsFilePath);
            return null;
        }
    }

    private void SaveSettings() {
        if (_isInitializing) return;

        try {
            var settings = new AppSettings {
                ConfigPath = _currentConfigPath,
                    SelectedModelName = _activeConfig?.model_name ?? string.Empty,
                    SplitCharacter = SplitCharacterCheckBox.IsChecked ?? false
            };

            var options = new JsonSerializerOptions {
                WriteIndented = true
            };
            string json = JsonSerializer.Serialize(settings, options);
            File.WriteAllText(_settingsFilePath, json, Encoding.UTF8);

            Log.Debug("Saved application settings to {SettingsPath}", _settingsFilePath);
        } catch (Exception ex) {
            Log.Error(ex, "Failed to save settings to {SettingsPath}", _settingsFilePath);
        }
    }

    private async void OnBrowseConfigClicked(object ? sender, RoutedEventArgs e) {
        var topLevel = TopLevel.GetTopLevel(this);
        if (topLevel == null) return;

        var files = await topLevel.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions {
            Title = "Select G2P Configuration File",
                AllowMultiple = false,
                FileTypeFilter = new [] {
                    new FilePickerFileType("YAML Configurations") {
                            Patterns = new [] {
                                "*.yaml",
                                "*.yml"
                            }
                        },
                        new FilePickerFileType("All Files") {
                            Patterns = new [] {
                                "*"
                            }
                        }
                }
        });

        if (files.Count > 0) {
            string selectedPath = files[0].Path.LocalPath;
            LoadYamlConfig(selectedPath);
            SaveSettings();
        }
    }

    private void LoadYamlConfig(string configPath) {
        Log.Information("Loading configuration from: {ConfigPath}", configPath);
        _currentConfigPath = configPath;
        ConfigPathLabel.Text = configPath;
        _configDirectory = Path.GetDirectoryName(Path.GetFullPath(configPath)) ?? AppContext.BaseDirectory;

        if (!File.Exists(configPath)) {
            string errorMsg = $"Config file '{configPath}' not found.";
            Log.Error(errorMsg);
            StatusLog.Text = errorMsg;
            return;
        }

        try {
            var deserializer = new DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .IgnoreUnmatchedProperties()
                .Build();

            string yamlContent = File.ReadAllText(configPath, Encoding.UTF8);
            _configs = deserializer.Deserialize < List < ModelConfig >> (yamlContent) ?? new();

            Log.Information("Successfully loaded {Count} model configurations.", _configs.Count);

            ModelComboBox.ItemsSource = _configs;
            if (_configs.Count > 0 && ModelComboBox.SelectedIndex < 0) {
                ModelComboBox.SelectedIndex = 0;
            }
        } catch (Exception ex) {
            Log.Error(ex, "Failed to parse YAML configuration file.");
            StatusLog.Text = $"Failed to load config file: {ex.Message}";
        }
    }

    private void OnModelSelectionChanged(object ? sender, SelectionChangedEventArgs e) {
        if (ModelComboBox.SelectedItem is not ModelConfig selectedModel) return;

        Log.Information("Switching active model to: {ModelName}", selectedModel.model_name);
        _activeConfig = selectedModel;
        SplitCharacterCheckBox.IsChecked = _activeConfig.split_character;
        LoadModelAndDictionary();
        SaveSettings();
    }

    private string ResolveRelativePath(string relativeOrAbsolutePath) {
        if (Path.IsPathRooted(relativeOrAbsolutePath))
            return relativeOrAbsolutePath;
        return Path.Combine(_configDirectory, relativeOrAbsolutePath);
    }

    private void LoadModelAndDictionary() {
        if (_activeConfig == null) return;

        _dictionary.Clear();
        _predCache.Clear();
        _modelPhoneTypes.Clear();
        _session?.Dispose();
        _session = null;

        List < string > dictPaths = _activeConfig.dict_path
        switch {
            List < object > list => list.Select(x => x.ToString() !).ToList(),
                string str => new List < string > {
                    str
                },
                _ => new List < string > ()
        };

        bool loadedDict = false;
        foreach(var rawPath in dictPaths) {
            string resolvedPath = ResolveRelativePath(rawPath);
            if (File.Exists(resolvedPath)) {
                Log.Debug("Reading dictionary file: {Path}", resolvedPath);
                int lineCount = 0;
                foreach(var line in File.ReadLines(resolvedPath)) {
                    var parts = Regex.Split(line.Trim(), @"\t+| {2,}");
                    if (parts.Length >= 2) {
                        string grapheme = parts[0].ToLower();
                        string rawPhonemes = Regex.Replace(parts[1], @"[0-3]", "").ToLower();
                        _dictionary[grapheme] = rawPhonemes.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToList();
                        loadedDict = true;
                        lineCount++;
                    }
                }
                Log.Information("Loaded {Entries} dictionary entries from {Path}", lineCount, resolvedPath);
            } else {
                Log.Warning("Dictionary file not found: {Path}", resolvedPath);
            }
        }

        LoadPhonesTxtForActiveModel(dictPaths);

        _modelPhonemes = _activeConfig.phonemes.Skip(4).ToList();
        _graphemeIndexes = new Dictionary < char, int > ();

        var indexedGraphemes = _activeConfig.graphemes
            .Skip(4)
            .Select((g, i) => (Grapheme: g, Index: i + 4));

        foreach(var (grapheme, index) in indexedGraphemes) {
            if (!string.IsNullOrEmpty(grapheme)) {
                _graphemeIndexes.TryAdd(grapheme[0], index);
            }
        }

        bool loadedModel = false;
        string resolvedOnnxPath = ResolveRelativePath(_activeConfig.onnx_path);

        if (!string.IsNullOrEmpty(_activeConfig.onnx_path) && _activeConfig.onnx_path != "null" && File.Exists(resolvedOnnxPath)) {
            try {
                Log.Information("Loading ONNX session from: {Path}", resolvedOnnxPath);
                _session = new InferenceSession(resolvedOnnxPath);
                var inputNames = _session.InputMetadata.Keys.ToList();

                _isAutoregressive = inputNames.Contains("tgt") && inputNames.Contains("t");
                loadedModel = true;

                string archType = _isAutoregressive ? "Autoregressive (Legacy)" : "End-to-End (New)";
                Log.Information("ONNX Session created successfully. Detected architecture: {Architecture}", archType);

                ModelTypeLabel.Text = archType;
                ModelTypeLabel.Foreground = Brushes.White;

                ModelTypeBadge.Background = _isAutoregressive ?
                    new SolidColorBrush(Color.Parse("#EF8331")) :
                    new SolidColorBrush(Color.Parse("#25B19C"));
            } catch (Exception ex) {
                Log.Error(ex, "Failed to load ONNX model from {Path}", resolvedOnnxPath);
                StatusLog.Text = $"ONNX Load Error: {ex.Message}";
            }
        } else if (_activeConfig.onnx_path != "null") {
            Log.Warning("ONNX model specified but file does not exist: {Path}", resolvedOnnxPath);
        }

        if (!loadedModel) {
            ModelTypeLabel.Text = "Dictionary Only";
            ModelTypeLabel.Foreground = new SolidColorBrush(Color.Parse("#804D1C"));
            ModelTypeBadge.Background = new SolidColorBrush(Color.Parse("#E2DAC3"));
        }

        StatusLog.Text = $"Loaded model '{_activeConfig.model_name}' (Dict: {loadedDict}, ONNX: {loadedModel}, Phones.txt: {_modelPhoneTypes.Count > 0})";

        ExecutePrediction();
    }

    private void LoadPhonesTxtForActiveModel(List < string > dictPaths) {
        string ? phonesFileToLoad = null;

        if (!string.IsNullOrEmpty(_activeConfig?.phones_path)) {
            string candidate = ResolveRelativePath(_activeConfig.phones_path);
            if (File.Exists(candidate)) phonesFileToLoad = candidate;
        }

        if (phonesFileToLoad == null && !string.IsNullOrEmpty(_activeConfig?.onnx_path) && _activeConfig.onnx_path != "null") {
            string onnxPath = ResolveRelativePath(_activeConfig.onnx_path);
            string ? dir = Path.GetDirectoryName(onnxPath);
            if (!string.IsNullOrEmpty(dir)) {
                string candidate = Path.Combine(dir, "phones.txt");
                if (File.Exists(candidate)) phonesFileToLoad = candidate;
            }
        }

        if (phonesFileToLoad == null && dictPaths.Count > 0) {
            string dictPath = ResolveRelativePath(dictPaths[0]);
            string ? dir = Path.GetDirectoryName(dictPath);
            if (!string.IsNullOrEmpty(dir)) {
                string candidate = Path.Combine(dir, "phones.txt");
                if (File.Exists(candidate)) phonesFileToLoad = candidate;
            }
        }

        if (phonesFileToLoad != null) {
            ParsePhonesTxt(phonesFileToLoad);
            Log.Information("Successfully loaded {Count} phoneme type rules from {Path}", _modelPhoneTypes.Count, phonesFileToLoad);
        } else {
            Log.Information("No phones.txt found for model '{ModelName}'. Falling back to default categorizer.", _activeConfig?.model_name);
        }
    }

    private void ParsePhonesTxt(string filePath) {
        var knownCategoryNames = new HashSet < string > (StringComparer.OrdinalIgnoreCase) {
            "vowel",
            "stop",
            "plosive",
            "affricate",
            "fricative",
            "nasal",
            "semivowel",
            "glide",
            "liquid",
            "tap",
            "flap",
            "aspirate",
            "consonant"
        };

        foreach(var rawLine in File.ReadLines(filePath)) {
            string line = rawLine.Trim();
            if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#") || line.StartsWith("//"))
                continue;

            var tokens = Regex.Split(line, @"\t+| +").Where(t => !string.IsNullOrWhiteSpace(t)).ToArray();
            if (tokens.Length < 2) continue;

            if (knownCategoryNames.Contains(tokens[0])) {
                string category = Capitalize(tokens[0]);
                for (int i = 1; i < tokens.Length; i++) {
                    _modelPhoneTypes[tokens[i].ToLower()] = category;
                }
            } else if (knownCategoryNames.Contains(tokens[1])) {
                string phoneme = tokens[0].ToLower();
                string category = Capitalize(tokens[1]);
                _modelPhoneTypes[phoneme] = category;
            } else if (tokens.Length == 2) {
                _modelPhoneTypes[tokens[0].ToLower()] = Capitalize(tokens[1]);
            }
        }
    }

    private static string Capitalize(string input) {
        if (string.IsNullOrEmpty(input)) return input;
        return char.ToUpper(input[0]) + input.Substring(1).ToLower();
    }

    private void OnInputTextChanged(object ? sender, TextChangedEventArgs e) => ExecutePrediction();

    private void OnSplitCharacterToggled(object ? sender, RoutedEventArgs e) {
        bool isEnabled = SplitCharacterCheckBox.IsChecked ?? false;
        if (_activeConfig != null) {
            _activeConfig.split_character = isEnabled;
            Log.Debug("Split character processing toggled to: {Enabled}", isEnabled);
            ExecutePrediction();
            SaveSettings();
        }
    }

    private void ExecutePrediction() {
        string rawInput = GraphemeInput.Text?.Trim() ?? string.Empty;
        _noteCollection.Clear();

        if (string.IsNullOrEmpty(rawInput)) {
            UpdateLegendCounts(new List < string > ());
            return;
        }

        string processedInput = (_activeConfig?.split_character == true) ?
            SplitInput(rawInput) :
            rawInput;

        string rawPhonemeString = Predict(processedInput);
        RenderNoteBoxes(processedInput, rawPhonemeString);
    }

    private string Predict(string inputText) {
        string[] words = inputText.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        List < string > predictedPhonemes = new();

        foreach(var word in words) {
            string wordLower = word.ToLower();
            if (_dictionary.TryGetValue(wordLower, out
                    var dictPhonemes)) {
                predictedPhonemes.Add(string.Join(" ", dictPhonemes));
            } else if (_predCache.TryGetValue(wordLower, out
                    var cachedPhonemes)) {
                predictedPhonemes.Add(string.Join(" ", cachedPhonemes));
            } else {
                string predicted = PredictWithModel(word);
                if (!string.IsNullOrEmpty(predicted)) {
                    predictedPhonemes.Add(predicted);
                    _predCache[wordLower] = predicted.Split(' ').ToList();
                } else {
                    predictedPhonemes.Add("N/A");
                }
            }
        }

        return string.Join("] [", predictedPhonemes);
    }

    private void RenderNoteBoxes(string processedInputText, string rawPhonemeString) {
        _noteCollection.Clear();
        List < string > detectedCategoryNames = new();

        string[] inputWords = processedInputText.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        string[] wordBlocks = rawPhonemeString.Split("] [", StringSplitOptions.RemoveEmptyEntries);

        for (int w = 0; w < wordBlocks.Length; w++) {
            string wordLabel = (w < inputWords.Length) ? inputWords[w] : $"Note {w + 1}";
            string block = wordBlocks[w].Replace("[", "").Replace("]", "").Trim();
            string[] phonemes = block.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            var noteModel = new NoteBoxModel {
                Word = wordLabel
            };

            foreach(var ph in phonemes) {
                string categoryName = GetPhonemeCategoryName(ph);
                detectedCategoryNames.Add(categoryName);

                IBrush textBrush = GetCategoryTextColorByName(categoryName);

                noteModel.Phonemes.Add(new PhonemeTextModel {
                    Phoneme = ph,
                        TypeName = categoryName,
                        Foreground = textBrush
                });
            }

            _noteCollection.Add(noteModel);
        }

        UpdateLegendCounts(detectedCategoryNames);
    }

    private string GetPhonemeCategoryName(string phoneme) {
        string pLower = phoneme.ToLower();

        if (_modelPhoneTypes.TryGetValue(pLower, out
                var customType)) {
            return customType;
        }

        return CategorizePhonemeFallback(pLower).ToString();
    }

    private static PhonemeCategory CategorizePhonemeFallback(string p) {
        if (Regex.IsMatch(p, @"^(a|i|u|e|o|aa|ae|ah|ao|aw|ax|ay|eh|er|ey|ih|iy|ow|oy|uh|uw|n|v|q)$"))
            return PhonemeCategory.Vowel;

        if (Regex.IsMatch(p, @"^(p|b|t|d|k|g|cl|q|tt|pp|kk|dd|bb|gg)$"))
            return PhonemeCategory.Stop;

        if (Regex.IsMatch(p, @"^(ch|j|ts|dz|tsh|dzh|c)$"))
            return PhonemeCategory.Affricate;

        if (Regex.IsMatch(p, @"^(f|v|th|dh|s|z|sh|zh|hy|x|h)$"))
            return PhonemeCategory.Fricative;

        if (Regex.IsMatch(p, @"^(m|n|ng|ny|n')$"))
            return PhonemeCategory.Nasal;

        if (Regex.IsMatch(p, @"^(w|y|j)$"))
            return PhonemeCategory.Semivowel;

        if (Regex.IsMatch(p, @"^(l|r)$"))
            return PhonemeCategory.Liquid;

        if (Regex.IsMatch(p, @"^(rr|dx|ry)$"))
            return PhonemeCategory.Tap;

        if (Regex.IsMatch(p, @"^(hh|h)$"))
            return PhonemeCategory.Aspirate;

        return PhonemeCategory.Unknown;
    }

    private static IBrush GetCategoryTextColorByName(string categoryName) {
        string lower = categoryName.ToLowerInvariant();

        if (lower.Contains("vowel"))
            return new SolidColorBrush(Color.Parse("#209988")); // Pastel Teal

        if (lower.Contains("stop") || lower.Contains("plosive"))
            return new SolidColorBrush(Color.Parse("#D9534F")); // Pastel Coral Red

        if (lower.Contains("affricate"))
            return new SolidColorBrush(Color.Parse("#D97706")); // Pastel Amber

        if (lower.Contains("fricative"))
            return new SolidColorBrush(Color.Parse("#B2742E")); // Warm Orange-Brown

        if (lower.Contains("nasal"))
            return new SolidColorBrush(Color.Parse("#8E44AD")); // Pastel Purple

        if (lower.Contains("semivowel") || lower.Contains("glide"))
            return new SolidColorBrush(Color.Parse("#16A085")); // Mint Turquoise

        if (lower.Contains("liquid"))
            return new SolidColorBrush(Color.Parse("#D35400")); // Deep Amber

        if (lower.Contains("tap") || lower.Contains("flap"))
            return new SolidColorBrush(Color.Parse("#C0392B")); // Soft Rose Red

        if (lower.Contains("aspirate"))
            return new SolidColorBrush(Color.Parse("#804D1C")); // Warm Ochre

        // Fallback for custom categories
        int hash = Math.Abs(categoryName.GetHashCode());
        string[] customPalette = new [] {
            "#209988",
            "#D9534F",
            "#D97706",
            "#B2742E",
            "#8E44AD",
            "#16A085",
            "#D35400",
            "#C0392B",
            "#804D1C"
        };
        return new SolidColorBrush(Color.Parse(customPalette[hash % customPalette.Length]));
    }

    private void UpdateLegendCounts(List < string > activeCategories) {
        _legendCollection.Clear();

        var defaultCategories = Enum.GetValues < PhonemeCategory > ()
            .Where(c => c != PhonemeCategory.Separator && c != PhonemeCategory.Unknown)
            .Select(c => c.ToString());

        var allCategories = defaultCategories.Union(_modelPhoneTypes.Values).Distinct();

        foreach(var catName in allCategories) {
            int count = activeCategories.Count(c => c.Equals(catName, StringComparison.OrdinalIgnoreCase));
            IBrush colorBrush = GetCategoryTextColorByName(catName);

            _legendCollection.Add(new PhonemeTypeLegendModel {
                TypeName = catName,
                    BadgeColor = colorBrush,
                    Count = count
            });
        }
    }

    private string PredictWithModel(string word) {
        if (_session == null) return string.Empty;

        try {
            return _isAutoregressive ?
                PredictAutoregressive(word) :
                PredictEndToEnd(word);
        } catch (Exception ex) {
            Log.Error(ex, "Exception during model inference for word '{Word}'", word);
            return string.Empty;
        }
    }

    private string PredictEndToEnd(string word) {
        int[] inputIds = word.Select(c => _graphemeIndexes.TryGetValue(c, out int idx) ? idx : 1).ToArray();
        var tensor = new DenseTensor < int > (inputIds, new [] {
            1,
            inputIds.Length
        });

        var inputs = new List < NamedOnnxValue > {
            NamedOnnxValue.CreateFromTensor("src", tensor)
        };

        using
        var results = _session!.Run(inputs);
        var tgt = results.First().AsTensor < int > ();

        List < string > phonemes = new();
        foreach(int id in tgt) {
            int index = id - 4;
            if (id > 3 && index >= 0 && index < _modelPhonemes.Count) {
                phonemes.Add(_modelPhonemes[index]);
            }
        }

        return string.Join(" ", phonemes);
    }

    private string PredictAutoregressive(string word) {
        string wordWithDash = "-" + word;
        int[] inputIds = wordWithDash.Select(c => _graphemeIndexes.TryGetValue(c, out int idx) ? idx : 0).ToArray();

        var srcTensor = new DenseTensor < int > (inputIds, new [] {
            1,
            inputIds.Length
        });
        var tgtList = new List < int > {
            2
        };
        var tVal = new DenseTensor < int > (new [] {
            1
        }, new [] {
            1
        });

        int inputLength = inputIds.Length;

        while (tVal[0] < inputLength && tgtList.Count < 48) {
            var tgtTensor = new DenseTensor < int > (tgtList.ToArray(), new [] {
                1,
                tgtList.Count
            });
            var inputs = new List < NamedOnnxValue > {
                NamedOnnxValue.CreateFromTensor("src", srcTensor),
                NamedOnnxValue.CreateFromTensor("tgt", tgtTensor),
                NamedOnnxValue.CreateFromTensor("t", tVal)
            };

            using
            var results = _session!.Run(inputs);
            var predTensor = results.First().AsTensor < int > ();
            int pred = predTensor.First();

            if (pred != 2) {
                tgtList.Add(pred);
            } else {
                tVal[0]++;
            }
        }

        List < string > phonemes = new();
        foreach(int id in tgtList) {
            int index = id - 4;
            if (id != 2 && index >= 0 && index < _modelPhonemes.Count) {
                phonemes.Add(_modelPhonemes[index]);
            }
        }

        return string.Join(" ", phonemes);
    }

    private string SplitInput(string inputText) {
        List < string > result = new();
        int i = 0;
        var isRoman = new Regex(@"[A-Za-z]");
        var isVowel = new Regex(@"[aeiouAEIOU]");

        while (i < inputText.Length) {
            bool found = false;
            for (int len = 3; len >= 2; len--) {
                if (i + len <= inputText.Length) {
                    string sub = inputText.Substring(i, len);
                    if (_dictionary.ContainsKey(sub)) {
                        result.Add(sub);
                        i += len;
                        found = true;
                        break;
                    }
                }
            }

            if (found) continue;

            if (isRoman.IsMatch(inputText[i].ToString())) {
                if (i + 1 < inputText.Length && isRoman.IsMatch(inputText[i + 1].ToString())) {
                    if (isVowel.IsMatch(inputText[i].ToString()) && isVowel.IsMatch(inputText[i + 1].ToString())) {
                        result.Add(inputText[i].ToString());
                        i += 1;
                    } else if (isVowel.IsMatch(inputText[i + 1].ToString())) {
                        result.Add(inputText.Substring(i, 2));
                        i += 2;
                    } else {
                        result.Add(inputText[i].ToString());
                        i += 1;
                    }
                } else {
                    result.Add(inputText[i].ToString());
                    i += 1;
                }
            } else {
                result.Add(inputText[i].ToString());
                i += 1;
            }
        }

        return string.Join(" ", result);
    }
}