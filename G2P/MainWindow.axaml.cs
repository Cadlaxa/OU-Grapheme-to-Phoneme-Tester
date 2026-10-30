using System;
using System.Collections.Generic;
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

    public MainWindow() {
        InitializeComponent();

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
        _session?.Dispose();
        _session = null;

        // 1. Parse Dictionary Files
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

        // 2. Setup Graphemes and Phonemes Lookup (using TryAdd to avoid duplicate key crashes)
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

        // 3. Load ONNX Model and Auto-Detect Architecture
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

                // Pastel Badge Styling matching cad_pastel palette
                ModelTypeBadge.Background = _isAutoregressive ?
                    new SolidColorBrush(Color.Parse("#EF8331")) // Amber/Orange
                    :
                    new SolidColorBrush(Color.Parse("#25B19C")); // Teal
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

        StatusLog.Text = $"Loaded model '{_activeConfig.model_name}' (Dict: {loadedDict}, ONNX: {loadedModel})";

        ExecutePrediction();
    }

    private void OnInputTextChanged(object ? sender, TextChangedEventArgs e) {
        ExecutePrediction();
    }

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
        string input = GraphemeInput.Text?.Trim() ?? string.Empty;
        if (string.IsNullOrEmpty(input)) {
            PhonemeOutput.Text = string.Empty;
            return;
        }

        if (_activeConfig?.split_character == true) {
            input = SplitInput(input);
        }

        string result = Predict(input);
        PhonemeOutput.Text = $"[{result}]";
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

    private string PredictWithModel(string word) {
        if (_session == null) {
            return string.Empty;
        }

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