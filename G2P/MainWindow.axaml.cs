using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Serilog;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;
using LanguageDetection;

namespace G2P;

public class ModelConfig {
    public string model_name { get; set; } = string.Empty;
    public string language { get; set; } = string.Empty;
    public string merge_with { get; set; } = string.Empty;
    public List<string> graphemes { get; set; } = new();
    public List<string> phonemes { get; set; } = new();
    public object dict_path { get; set; } = string.Empty;
    public string onnx_path { get; set; } = string.Empty;
    public string phones_path { get; set; } = string.Empty;
    public bool split_character { get; set; }
}

public class AppSettings {
    public string ConfigPath { get; set; } = string.Empty;
    public string SelectedModelName { get; set; } = string.Empty;
    public bool SplitCharacter { get; set; }
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
    public string Phoneme { get; set; } = string.Empty;
    public string TypeName { get; set; } = string.Empty;
    public IBrush Foreground { get; set; } = Brushes.Black;
}

public class NoteBoxModel {
    public string Word { get; set; } = string.Empty;
    public ObservableCollection<PhonemeTextModel> Phonemes { get; set; } = new();
    public string ModelTag { get; set; } = string.Empty;
    public string SourceModelInfo { get; set; } = string.Empty;
}

public class PhonemeTypeLegendModel {
    public string TypeName { get; set; } = string.Empty;
    public IBrush BadgeColor { get; set; } = Brushes.Gray;
    public int Count { get; set; }
}

public class WordPredictionResult {
    public string Word { get; set; } = string.Empty;
    public List<string> Phonemes { get; set; } = new();
    public string SourceModelName { get; set; } = string.Empty;
    public string SourceLanguage { get; set; } = string.Empty;
    public bool IsMergedModel { get; set; }
}

public partial class MainWindow : Window {
    private List<ModelConfig> _configs = new();
    private ModelConfig? _activeConfig;
    private ModelConfig? _mergedConfig;

    private LanguageDetector? _detector;

    // Per-word manual model overrides (true = Merged Model, false = Primary Model)
    private readonly Dictionary<string, bool?> _wordModelOverrides = new(StringComparer.OrdinalIgnoreCase);

    // Primary Active Model State
    private InferenceSession? _session;
    private bool _isAutoregressive;
    private readonly Dictionary<string, List<string>> _dictionary = new();
    private Dictionary<char, int> _graphemeIndexes = new();
    private List<string> _modelPhonemes = new();

    // Merged Target Model State
    private InferenceSession? _mergedSession;
    private bool _isMergedAutoregressive;
    private readonly Dictionary<string, List<string>> _mergedDictionary = new();
    private Dictionary<char, int> _mergedGraphemeIndexes = new();
    private List<string> _mergedModelPhonemes = new();

    private string _configDirectory = AppContext.BaseDirectory;
    private string _currentConfigPath = string.Empty;
    private bool _isInitializing = true;

    private readonly string _settingsFilePath = Path.Combine(AppContext.BaseDirectory, "settings.json");
    private readonly Dictionary<string, List<string>> _predCache = new();
    private readonly Dictionary<string, string> _modelPhoneTypes = new(StringComparer.OrdinalIgnoreCase);

    private readonly ObservableCollection<NoteBoxModel> _noteCollection = new();
    private readonly ObservableCollection<PhonemeTypeLegendModel> _legendCollection = new();

    public MainWindow() {
        InitializeComponent();

        PhonemesItemsControl.ItemsSource = _noteCollection;
        PhonemeTypesLegendControl.ItemsSource = _legendCollection;

        LoadSettingsAndInitialize();
        _isInitializing = false;
    }

    private void LoadSettingsAndInitialize() {
        AppSettings? settings = LoadSettings();
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

    private AppSettings? LoadSettings() {
        if (!File.Exists(_settingsFilePath)) return null;

        try {
            string json = File.ReadAllText(_settingsFilePath, Encoding.UTF8);
            var settings = JsonSerializer.Deserialize<AppSettings>(json);
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

            var options = new JsonSerializerOptions { WriteIndented = true };
            string json = JsonSerializer.Serialize(settings, options);
            File.WriteAllText(_settingsFilePath, json, Encoding.UTF8);

            Log.Debug("Saved application settings to {SettingsPath}", _settingsFilePath);
        } catch (Exception ex) {
            Log.Error(ex, "Failed to save settings to {SettingsPath}", _settingsFilePath);
        }
    }

    private async void OnBrowseConfigClicked(object? sender, RoutedEventArgs e) {
        var topLevel = TopLevel.GetTopLevel(this);
        if (topLevel == null) return;

        var files = await topLevel.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions {
            Title = "Select G2P Configuration File",
            AllowMultiple = false,
            FileTypeFilter = new[] {
                new FilePickerFileType("YAML Configurations") { Patterns = new[] { "*.yaml", "*.yml" } },
                new FilePickerFileType("All Files") { Patterns = new[] { "*" } }
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
            _configs = deserializer.Deserialize<List<ModelConfig>>(yamlContent) ?? new();

            Log.Information("Successfully loaded {Count} model configurations.", _configs.Count);

            InitializeLanguageDetectorFromConfigs();

            ModelComboBox.ItemsSource = _configs;
            if (_configs.Count > 0 && ModelComboBox.SelectedIndex < 0) {
                ModelComboBox.SelectedIndex = 0;
            }
        } catch (Exception ex) {
            Log.Error(ex, "Failed to parse YAML configuration file.");
            StatusLog.Text = $"Failed to load config file: {ex.Message}";
        }
    }

    private void InitializeLanguageDetectorFromConfigs() {
        try {
            var isoCodes = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

            foreach (var cfg in _configs) {
                if (string.IsNullOrWhiteSpace(cfg.language)) continue;

                string iso = GetIsoLanguageCode(cfg.language);
                if (!string.IsNullOrEmpty(iso)) {
                    isoCodes.Add(iso);
                }
            }

            if (isoCodes.Count > 0) {
                _detector = new LanguageDetector();
                _detector.AddLanguages(isoCodes.ToArray());
                Log.Information("Initialized LanguageDetector with ISO codes: [{Languages}]", string.Join(", ", isoCodes));
            } else {
                _detector = null;
            }
        } catch (Exception ex) {
            Log.Error(ex, "Failed to initialize LanguageDetector. Falling back to rule-based detection.");
            _detector = null;
        }
    }

    private static string GetIsoLanguageCode(string lang) {
        if (string.IsNullOrWhiteSpace(lang)) return string.Empty;

        string clean = lang.Trim().ToLower();
        return clean switch {
            "english" or "en" or "arpa" or "en arpa+" => "en",
            "filipino" or "tagalog" or "tl" or "fili" => "tl",
            "japanese" or "ja" or "jp" => "ja",
            "spanish" or "es" => "es",
            "chinese" or "zh" => "zh",
            "korean" or "ko" => "ko",
            "german" or "de" => "de",
            "french" or "fr" => "fr",
            _ => clean.Length == 2 ? clean : string.Empty
        };
    }

    private string DetectLanguage(string word) {
        if (string.IsNullOrWhiteSpace(word)) {
            return GetIsoLanguageCode(_activeConfig?.language ?? string.Empty);
        }

        string wordLower = word.ToLower();

        bool hasEnglishFeatures = Regex.IsMatch(wordLower, @"[cfvzxq]|ea|ou|ee|oo|tion|sion|ment|ness|ight");
        if (hasEnglishFeatures) {
            return "en";
        }

        if (_detector != null && wordLower.Length >= 4) {
            try {
                string detectedIso = _detector.Detect(wordLower);
                if (!string.IsNullOrEmpty(detectedIso)) {
                    return GetIsoLanguageCode(detectedIso);
                }
            } catch {
                // Ignore ambiguous short tokens
            }
        }

        return GetIsoLanguageCode(_activeConfig?.language ?? string.Empty);
    }

    private void OnModelSelectionChanged(object? sender, SelectionChangedEventArgs e) {
        if (ModelComboBox.SelectedItem is not ModelConfig selectedModel) return;

        Log.Information("Switching active model to: {ModelName} [{Language}]", selectedModel.model_name, selectedModel.language);
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

        // Clear primary state
        _dictionary.Clear();
        _predCache.Clear();
        _modelPhoneTypes.Clear();
        _wordModelOverrides.Clear();
        _session?.Dispose();
        _session = null;

        // Clear merged state
        _mergedDictionary.Clear();
        _mergedSession?.Dispose();
        _mergedSession = null;
        _mergedConfig = null;

        // 1. Load Primary Active Model
        LoadSingleModelConfig(_activeConfig, _dictionary, ref _graphemeIndexes, ref _modelPhonemes, ref _session, ref _isAutoregressive);

        // 2. Load Secondary Merged Model (if specified in merge_with)
        if (!string.IsNullOrEmpty(_activeConfig.merge_with)) {
            _mergedConfig = _configs.FirstOrDefault(c =>
                c.model_name.Equals(_activeConfig.merge_with, StringComparison.OrdinalIgnoreCase));

            if (_mergedConfig != null) {
                Log.Information("Loading secondary merge model '{MergedName}' (Language: '{Lang}')",
                    _mergedConfig.model_name, _mergedConfig.language);

                LoadSingleModelConfig(_mergedConfig, _mergedDictionary, ref _mergedGraphemeIndexes, ref _mergedModelPhonemes, ref _mergedSession, ref _isMergedAutoregressive);
            } else {
                Log.Warning("Specified merge_with model '{MergeTarget}' not found in configuration list.", _activeConfig.merge_with);
            }
        }

        if (_session != null) {
            string archType = _isAutoregressive ? "Autoregressive" : "End-to-End";
            if (_mergedSession != null) archType += " + Merged";
            ModelTypeLabel.Text = archType;
            ModelTypeLabel.Foreground = Brushes.White;
            ModelTypeBadge.Background = _isAutoregressive ?
                new SolidColorBrush(Color.Parse("#EF8331")) :
                new SolidColorBrush(Color.Parse("#25B19C"));
        } else {
            ModelTypeLabel.Text = "Dictionary Only";
            ModelTypeLabel.Foreground = new SolidColorBrush(Color.Parse("#804D1C"));
            ModelTypeBadge.Background = new SolidColorBrush(Color.Parse("#E2DAC3"));
        }

        StatusLog.Text = $"Loaded model '{_activeConfig.model_name}' [{_activeConfig.language}] (Merged: '{_mergedConfig?.model_name ?? "None"}')";
        ExecutePrediction();
    }

    private void LoadSingleModelConfig(
        ModelConfig config,
        Dictionary<string, List<string>> targetDict,
        ref Dictionary<char, int> targetGraphemeIndexes,
        ref List<string> targetPhonemes,
        ref InferenceSession? targetSession,
        ref bool isAutoregressive) 
    {
        List<string> dictPaths = config.dict_path switch {
            List<object> list => list.Select(x => x.ToString()!).ToList(),
            string str => new List<string> { str },
            _ => new List<string>()
        };

        foreach (var rawPath in dictPaths) {
            string resolvedPath = ResolveRelativePath(rawPath);
            if (File.Exists(resolvedPath)) {
                int lineCount = 0;
                foreach (var line in File.ReadLines(resolvedPath)) {
                    var parts = Regex.Split(line.Trim(), @"\t+| {2,}");
                    if (parts.Length >= 2) {
                        string grapheme = parts[0].ToLower();
                        string rawPhonemes = Regex.Replace(parts[1], @"[0-3]", "").ToLower();
                        targetDict.TryAdd(grapheme, rawPhonemes.Split(' ', StringSplitOptions.RemoveEmptyEntries).ToList());
                        lineCount++;
                    }
                }
                Log.Information("Loaded {Entries} dictionary entries for '{ModelName}' from {Path}", lineCount, config.model_name, resolvedPath);
            } else {
                Log.Warning("Dictionary file not found: {Path}", resolvedPath);
            }
        }

        LoadPhonesTxtForActiveModel(dictPaths, config);

        targetPhonemes = config.phonemes.Skip(4).ToList();
        targetGraphemeIndexes = new Dictionary<char, int>();
        var indexedGraphemes = config.graphemes.Skip(4).Select((g, i) => (Grapheme: g, Index: i + 4));

        foreach (var (grapheme, index) in indexedGraphemes) {
            if (!string.IsNullOrEmpty(grapheme)) {
                targetGraphemeIndexes.TryAdd(grapheme[0], index);
            }
        }

        string resolvedOnnxPath = ResolveRelativePath(config.onnx_path);
        if (!string.IsNullOrEmpty(config.onnx_path) && config.onnx_path != "null" && File.Exists(resolvedOnnxPath)) {
            try {
                Log.Information("Loading ONNX session for '{ModelName}' from: {Path}", config.model_name, resolvedOnnxPath);
                targetSession = new InferenceSession(resolvedOnnxPath);
                var inputNames = targetSession.InputMetadata.Keys.ToList();
                isAutoregressive = inputNames.Contains("tgt") && inputNames.Contains("t");
            } catch (Exception ex) {
                Log.Error(ex, "Failed to load ONNX model from {Path}", resolvedOnnxPath);
            }
        }
    }

    private void LoadPhonesTxtForActiveModel(List<string> dictPaths, ModelConfig? targetConfig = null) {
        var config = targetConfig ?? _activeConfig;
        string? phonesFileToLoad = null;

        if (!string.IsNullOrEmpty(config?.phones_path)) {
            string candidate = ResolveRelativePath(config.phones_path);
            if (File.Exists(candidate)) phonesFileToLoad = candidate;
        }

        if (phonesFileToLoad == null && !string.IsNullOrEmpty(config?.onnx_path) && config.onnx_path != "null") {
            string onnxPath = ResolveRelativePath(config.onnx_path);
            string? dir = Path.GetDirectoryName(onnxPath);
            if (!string.IsNullOrEmpty(dir)) {
                string candidate = Path.Combine(dir, "phones.txt");
                if (File.Exists(candidate)) phonesFileToLoad = candidate;
            }
        }

        if (phonesFileToLoad == null && dictPaths.Count > 0) {
            string dictPath = ResolveRelativePath(dictPaths[0]);
            string? dir = Path.GetDirectoryName(dictPath);
            if (!string.IsNullOrEmpty(dir)) {
                string candidate = Path.Combine(dir, "phones.txt");
                if (File.Exists(candidate)) phonesFileToLoad = candidate;
            }
        }

        if (phonesFileToLoad != null) {
            ParsePhonesTxt(phonesFileToLoad);
            Log.Information("Loaded phoneme type rules from {Path}", phonesFileToLoad);
        }
    }

    private void ParsePhonesTxt(string filePath) {
        var knownCategoryNames = new HashSet<string>(StringComparer.OrdinalIgnoreCase) {
            "vowel", "stop", "plosive", "affricate", "fricative",
            "nasal", "semivowel", "glide", "liquid", "tap", "flap", "aspirate", "consonant"
        };

        foreach (var rawLine in File.ReadLines(filePath)) {
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

    private void OnInputTextChanged(object? sender, TextChangedEventArgs e) => ExecutePrediction();

    private void OnSplitCharacterToggled(object? sender, RoutedEventArgs e) {
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
            UpdateLegendCounts(new List<string>());
            return;
        }

        string processedInput = (_activeConfig?.split_character == true) ?
            SplitInput(rawInput) :
            rawInput;

        var predictions = PredictWords(processedInput);
        RenderNoteBoxes(predictions);
    }

    private List<WordPredictionResult> PredictWords(string inputText) {
        string[] words = inputText.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        List<WordPredictionResult> results = new();

        foreach (var word in words) {
            string wordLower = word.ToLower();
            List<string> resultTokens = new();
            string sourceModelName = _activeConfig?.model_name ?? "Primary";
            string sourceLanguage = !string.IsNullOrWhiteSpace(_activeConfig?.language) 
                ? _activeConfig.language 
                : (_activeConfig?.model_name ?? "Primary");
            bool isMerged = false;

            // Manual user override handling (Left-click / Right-click note box)
            if (_wordModelOverrides.TryGetValue(wordLower, out bool? overrideMerged) && overrideMerged.HasValue) {
                if (overrideMerged.Value && _mergedConfig != null) {
                    resultTokens = PredictFromModelConfig(_mergedConfig, wordLower, _mergedDictionary, _mergedSession, _isMergedAutoregressive, _mergedGraphemeIndexes, _mergedModelPhonemes);
                    sourceModelName = _mergedConfig.model_name;
                    sourceLanguage = !string.IsNullOrWhiteSpace(_mergedConfig.language) ? _mergedConfig.language : _mergedConfig.model_name;
                    isMerged = true;
                } else if (_activeConfig != null) {
                    resultTokens = PredictFromModelConfig(_activeConfig, wordLower, _dictionary, _session, _isAutoregressive, _graphemeIndexes, _modelPhonemes);
                    sourceModelName = _activeConfig.model_name;
                    sourceLanguage = !string.IsNullOrWhiteSpace(_activeConfig.language) ? _activeConfig.language : _activeConfig.model_name;
                    isMerged = false;
                }
            } else {
                // Auto-detection language routing
                string detectedLang = DetectLanguage(wordLower);
                string mergedLang = GetIsoLanguageCode(_mergedConfig?.language ?? string.Empty);

                bool useMergedModel = _mergedConfig != null &&
                                      !string.IsNullOrEmpty(mergedLang) &&
                                      detectedLang.Equals(mergedLang, StringComparison.OrdinalIgnoreCase);

                if (useMergedModel && _mergedConfig != null) {
                    resultTokens = PredictFromModelConfig(_mergedConfig, wordLower, _mergedDictionary, _mergedSession, _isMergedAutoregressive, _mergedGraphemeIndexes, _mergedModelPhonemes);
                    if (resultTokens.Count > 0) {
                        sourceModelName = _mergedConfig.model_name;
                        sourceLanguage = !string.IsNullOrWhiteSpace(_mergedConfig.language) ? _mergedConfig.language : _mergedConfig.model_name;
                        isMerged = true;
                    }
                }

                if (resultTokens.Count == 0 && _activeConfig != null) {
                    resultTokens = PredictFromModelConfig(_activeConfig, wordLower, _dictionary, _session, _isAutoregressive, _graphemeIndexes, _modelPhonemes);
                    sourceModelName = _activeConfig.model_name;
                    sourceLanguage = !string.IsNullOrWhiteSpace(_activeConfig.language) ? _activeConfig.language : _activeConfig.model_name;
                    isMerged = false;
                }
            }

            results.Add(new WordPredictionResult {
                Word = word,
                Phonemes = resultTokens.Count > 0 ? resultTokens : new List<string> { "N/A" },
                SourceModelName = sourceModelName,
                SourceLanguage = sourceLanguage,
                IsMergedModel = isMerged
            });
        }

        return results;
    }

    private List<string> PredictFromModelConfig(
        ModelConfig? config,
        string wordLower,
        Dictionary<string, List<string>> dictionary,
        InferenceSession? session,
        bool isAutoregressive,
        Dictionary<char, int> graphemeIndexes,
        List<string> modelPhonemes) 
    {
        if (dictionary.TryGetValue(wordLower, out var dictPhonemes)) {
            return dictPhonemes;
        }
        if (session != null) {
            string pred = PredictWithModel(wordLower, session, isAutoregressive, graphemeIndexes, modelPhonemes);
            if (!string.IsNullOrEmpty(pred)) return pred.Split(' ').ToList();
        }
        return new List<string>();
    }

    private void RenderNoteBoxes(List<WordPredictionResult> predictions) {
        _noteCollection.Clear();
        List<string> detectedCategoryNames = new();

        foreach (var pred in predictions) {
            var noteModel = new NoteBoxModel {
                Word = pred.Word,
                ModelTag = $"[{pred.SourceLanguage}]",
                SourceModelInfo = $"Active Model: {pred.SourceModelName} [{pred.SourceLanguage}] {(pred.IsMergedModel ? "(Merged)" : "(Primary)")}"
            };

            // Format phonemes into syllables and insert '|' ONLY between syllables
            var formattedPhonemeModels = FormatPhonemesWithSyllableSeparators(pred.Phonemes, ref detectedCategoryNames);

            foreach (var item in formattedPhonemeModels) {
                noteModel.Phonemes.Add(item);
            }

            _noteCollection.Add(noteModel);
        }

        UpdateLegendCounts(detectedCategoryNames);
    }

    private List<PhonemeTextModel> FormatPhonemesWithSyllableSeparators(List<string> phonemeList, ref List<string> detectedCategoryNames) {
        var result = new List<PhonemeTextModel>();
        if (phonemeList == null || phonemeList.Count == 0) return result;

        bool hasExplicitBoundaries = phonemeList.Any(p => p == "." || p == "-" || p == "|");

        List<List<string>> syllables = new();

        if (hasExplicitBoundaries) {
            List<string> currentSyllable = new();
            foreach (var ph in phonemeList) {
                if (ph == "." || ph == "-" || ph == "|") {
                    if (currentSyllable.Count > 0) {
                        syllables.Add(new List<string>(currentSyllable));
                        currentSyllable.Clear();
                    }
                } else {
                    currentSyllable.Add(ph);
                }
            }
            if (currentSyllable.Count > 0) {
                syllables.Add(currentSyllable);
            }
        } else {
            syllables = SyllabifyByVowels(phonemeList);
        }

        for (int s = 0; s < syllables.Count; s++) {
            var syllable = syllables[s];

            for (int p = 0; p < syllable.Count; p++) {
                string ph = syllable[p];
                string categoryName = GetPhonemeCategoryName(ph);
                if (ph != "N/A") detectedCategoryNames.Add(categoryName);

                IBrush textBrush = GetCategoryTextColorByName(categoryName);

                result.Add(new PhonemeTextModel {
                    Phoneme = ph,
                    TypeName = categoryName,
                    Foreground = textBrush
                });
            }

            if (s < syllables.Count - 1) {
                result.Add(new PhonemeTextModel {
                    Phoneme = "|",
                    TypeName = "Syllable Boundary",
                    Foreground = GetCategoryTextColorByName("Separator")
                });
            }
        }

        return result;
    }

    private bool IsVowelPhoneme(string ph) {
        if (string.IsNullOrWhiteSpace(ph)) return false;
        string category = GetPhonemeCategoryName(ph);
        if (category.Equals("Vowel", StringComparison.OrdinalIgnoreCase)) return true;
        return Regex.IsMatch(ph.ToLower(), @"^(a|i|u|e|o|aa|ae|ah|ao|aw|ax|ay|eh|er|ey|ih|iy|ow|oy|uh|uw|@r|iw|uy|ew)");
    }

    private static bool IsValidOnsetCluster(string c1, string c2) {
        string cluster = (c1 + c2).ToLower();
        var validClusters = new HashSet<string>(StringComparer.OrdinalIgnoreCase) {
            "pr", "br", "tr", "dr", "kr", "gr", "fr", "pl", "bl", "kl", "gl", "fl", "sl",
            "sp", "st", "sk", "sm", "sn", "sw", "tw", "dw", "kw", "gw", "ch", "sh", "th",
            "ts", "dy", "ny", "sy", "ly", "ry", "my", "py", "by", "ky", "gy"
        };
        return validClusters.Contains(cluster);
    }

    private List<List<string>> SyllabifyByVowels(List<string> phonemes) {
        var syllables = new List<List<string>>();
        if (phonemes == null || phonemes.Count == 0) return syllables;

        List<int> vowelIndices = new List<int>();
        for (int i = 0; i < phonemes.Count; i++) {
            if (IsVowelPhoneme(phonemes[i])) {
                vowelIndices.Add(i);
            }
        }

        if (vowelIndices.Count <= 1) {
            syllables.Add(new List<string>(phonemes));
            return syllables;
        }

        List<int> splitIndices = new List<int>();

        for (int v = 0; v < vowelIndices.Count - 1; v++) {
            int v1 = vowelIndices[v];
            int v2 = vowelIndices[v + 1];
            int consCount = v2 - v1 - 1;

            int splitAt;
            if (consCount <= 0) {
                splitAt = v1 + 1;
            } else if (consCount == 1) {
                splitAt = v1 + 1;
            } else if (consCount == 2) {
                string c1 = phonemes[v1 + 1];
                string c2 = phonemes[v1 + 2];

                if (c1.Equals("s", StringComparison.OrdinalIgnoreCase) && Regex.IsMatch(c2, @"^(p|t|k|b|d|g|m|n)$", RegexOptions.IgnoreCase)) {
                    splitAt = v1 + 2;
                } else if (IsValidOnsetCluster(c1, c2)) {
                    splitAt = v1 + 1;
                } else {
                    splitAt = v1 + 2;
                }
            } else {
                string cPrev = phonemes[v2 - 2];
                string cLast = phonemes[v2 - 1];
                if (IsValidOnsetCluster(cPrev, cLast)) {
                    splitAt = v2 - 2;
                } else {
                    splitAt = v2 - 1;
                }
            }

            splitIndices.Add(splitAt);
        }

        int currentStart = 0;
        foreach (int splitIdx in splitIndices) {
            int length = splitIdx - currentStart;
            if (length > 0) {
                syllables.Add(phonemes.GetRange(currentStart, length));
            }
            currentStart = splitIdx;
        }
        if (currentStart < phonemes.Count) {
            syllables.Add(phonemes.GetRange(currentStart, phonemes.Count - currentStart));
        }

        return syllables;
    }

    private void OnNoteBoxPointerPressed(object? sender, PointerPressedEventArgs e) {
        if (sender is Border border && border.DataContext is NoteBoxModel noteModel) {
            var point = e.GetCurrentPoint(border);
            string wordLower = noteModel.Word.ToLower();

            if (point.Properties.IsLeftButtonPressed) {
                if (_mergedConfig == null) {
                    StatusLog.Text = $"No merged model available to switch for '{noteModel.Word}'.";
                    return;
                }
                // Left-click: Force Merged Model
                _wordModelOverrides[wordLower] = true;
                Log.Information("User forced Merged Model ('{MergedName}') for word '{Word}'", _mergedConfig.model_name, noteModel.Word);
                StatusLog.Text = $"Forced Merged Model ('{_mergedConfig.model_name}') for '{noteModel.Word}'";
                ExecutePrediction();
            } else if (point.Properties.IsRightButtonPressed) {
                // Right-click: Revert to Primary Model
                _wordModelOverrides[wordLower] = false;
                Log.Information("User reverted to Primary Model for word '{Word}'", noteModel.Word);
                StatusLog.Text = $"Reverted '{noteModel.Word}' to Primary Model ('{_activeConfig?.model_name}')";
                ExecutePrediction();
            }
        }
    }

    private string GetPhonemeCategoryName(string phoneme) {
        string pLower = phoneme.ToLower();

        if (_modelPhoneTypes.TryGetValue(pLower, out var customType)) {
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

        if (lower.Contains("separator"))
            return new SolidColorBrush(Color.Parse("#C2B496"));

        if (lower.Contains("vowel"))
            return new SolidColorBrush(Color.Parse("#209988"));

        if (lower.Contains("stop") || lower.Contains("plosive"))
            return new SolidColorBrush(Color.Parse("#D9534F"));

        if (lower.Contains("affricate"))
            return new SolidColorBrush(Color.Parse("#D97706"));

        if (lower.Contains("fricative"))
            return new SolidColorBrush(Color.Parse("#B2742E"));

        if (lower.Contains("nasal"))
            return new SolidColorBrush(Color.Parse("#8E44AD"));

        if (lower.Contains("semivowel") || lower.Contains("glide"))
            return new SolidColorBrush(Color.Parse("#16A085"));

        if (lower.Contains("liquid"))
            return new SolidColorBrush(Color.Parse("#D35400"));

        if (lower.Contains("tap") || lower.Contains("flap"))
            return new SolidColorBrush(Color.Parse("#C0392B"));

        if (lower.Contains("aspirate"))
            return new SolidColorBrush(Color.Parse("#804D1C"));

        int hash = Math.Abs(categoryName.GetHashCode());
        string[] customPalette = new[] {
            "#209988", "#D9534F", "#D97706", "#B2742E", "#8E44AD",
            "#16A085", "#D35400", "#C0392B", "#804D1C"
        };
        return new SolidColorBrush(Color.Parse(customPalette[hash % customPalette.Length]));
    }

    private void UpdateLegendCounts(List<string> activeCategories) {
        _legendCollection.Clear();

        var defaultCategories = Enum.GetValues<PhonemeCategory>()
            .Where(c => c != PhonemeCategory.Separator && c != PhonemeCategory.Unknown)
            .Select(c => c.ToString());

        var allCategories = defaultCategories.Union(_modelPhoneTypes.Values).Distinct();

        foreach (var catName in allCategories) {
            int count = activeCategories.Count(c => c.Equals(catName, StringComparison.OrdinalIgnoreCase));
            IBrush colorBrush = GetCategoryTextColorByName(catName);

            _legendCollection.Add(new PhonemeTypeLegendModel {
                TypeName = catName,
                BadgeColor = colorBrush,
                Count = count
            });
        }
    }

    private string PredictWithModel(
        string word,
        InferenceSession? session,
        bool isAutoregressive,
        Dictionary<char, int> graphemeIndexes,
        List<string> modelPhonemes) 
    {
        if (session == null) return string.Empty;

        try {
            return isAutoregressive ?
                PredictAutoregressive(word, session, graphemeIndexes, modelPhonemes) :
                PredictEndToEnd(word, session, graphemeIndexes, modelPhonemes);
        } catch (Exception ex) {
            Log.Error(ex, "Exception during model inference for word '{Word}'", word);
            return string.Empty;
        }
    }

    private string PredictEndToEnd(
        string word,
        InferenceSession session,
        Dictionary<char, int> graphemeIndexes,
        List<string> modelPhonemes) 
    {
        int[] inputIds = word.Select(c => graphemeIndexes.TryGetValue(c, out int idx) ? idx : 1).ToArray();
        var tensor = new DenseTensor<int>(inputIds, new[] { 1, inputIds.Length });
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("src", tensor) };

        using var results = session.Run(inputs);
        var tgt = results.First().AsTensor<int>();

        List<string> phonemes = new();
        foreach (int id in tgt) {
            int index = id - 4;
            if (id > 3 && index >= 0 && index < modelPhonemes.Count) {
                phonemes.Add(modelPhonemes[index]);
            }
        }

        return string.Join(" ", phonemes);
    }

    private string PredictAutoregressive(
        string word,
        InferenceSession session,
        Dictionary<char, int> graphemeIndexes,
        List<string> modelPhonemes) 
    {
        string wordWithDash = "-" + word;
        int[] inputIds = wordWithDash.Select(c => graphemeIndexes.TryGetValue(c, out int idx) ? idx : 0).ToArray();

        var srcTensor = new DenseTensor<int>(inputIds, new[] { 1, inputIds.Length });
        var tgtList = new List<int> { 2 };
        var tVal = new DenseTensor<int>(new[] { 1 }, new[] { 1 });

        int inputLength = inputIds.Length;

        while (tVal[0] < inputLength && tgtList.Count < 48) {
            var tgtTensor = new DenseTensor<int>(tgtList.ToArray(), new[] { 1, tgtList.Count });
            var inputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("src", srcTensor),
                NamedOnnxValue.CreateFromTensor("tgt", tgtTensor),
                NamedOnnxValue.CreateFromTensor("t", tVal)
            };

            using var results = session.Run(inputs);
            var predTensor = results.First().AsTensor<int>();
            int pred = predTensor.First();

            if (pred != 2) {
                tgtList.Add(pred);
            } else {
                tVal[0]++;
            }
        }

        List<string> phonemes = new();
        foreach (int id in tgtList) {
            int index = id - 4;
            if (id != 2 && index >= 0 && index < modelPhonemes.Count) {
                phonemes.Add(modelPhonemes[index]);
            }
        }

        return string.Join(" ", phonemes);
    }

    private string SplitInput(string inputText) {
        List<string> result = new();
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