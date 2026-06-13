# CivitAI Model Downloader 🚀

A robust Python script for downloading AI models (LoRA, Checkpoints, Embeddings) from CivitAI with intelligent file handling, automatic ZIP extraction, and resume support.

## ✨ Features

- **🔄 Smart Download Management** - Multi-connection downloads with aria2 for maximum speed
- **📦 Automatic ZIP Processing** - Extracts and filters `.safetensors` files automatically
- **🔧 Resume Support** - Continue interrupted downloads without starting over
- **✅ File Validation** - Detects and cleans up corrupted or incomplete downloads
- **🔐 Secure Token Handling** - Supports environment variables for API authentication
- **📊 Progress Tracking** - Real-time download progress with clear status indicators
- **🎯 Intelligent File Management** - Handles naming conflicts and cleans up temporary files

## 📋 Requirements

- Python 3.6+
- [aria2](https://aria2.github.io/) - High-speed download utility
- [requests](https://pypi.org/project/requests/) - HTTP library for API calls

## 🔧 Installation

### 1. Install aria2

**Ubuntu/Debian:**
```bash
sudo apt-get install aria2
```

**macOS:**
```bash
brew install aria2
```

**Windows:**
Download from [aria2 releases](https://github.com/aria2/aria2/releases)

### 2. Install Python dependencies

```bash
pip install requests
```

### 3. Download the script

```bash
wget https://github.com/concreteshoes/CivitAI_Downloader/blob/main/download_with_aria.py
chmod +x download_with_aria.py
```

## 🔑 Authentication

Get your CivitAI API token from [CivitAI Account Settings](https://civitai.com/user/account).

Set it as an environment variable:

```bash
export CIVITAI_TOKEN="your_token_here"
```

Or add to your `~/.bashrc` or `~/.zshrc` for permanent use:

```bash
echo 'export CIVITAI_TOKEN="your_token_here"' >> ~/.bashrc
source ~/.bashrc
```

## 📖 Usage

### Basic Usage

The script is set up to download from civitai.red by default. In order to download
from main website, you need to pass the following: `--base-url https://civitai.com/api`.
Download a model using its `modelVersionId` identifier:

```bash
./download_with_aria.py -m 123456
```

### Advanced Options

```bash
# Download to specific directory
./download_with_aria.py -m 123456 -o ./models

# Use custom filename
./download_with_aria.py -m 123456 --filename "my_custom_model.safetensors"

# Force re-download (ignore existing files)
./download_with_aria.py -m 123456 --force

# Provide token via command line (not recommended for security)
./download_with_aria.py -m 123456 --token "your_token_here"
```

### Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--model-id` | `-m` | CivitAI model version ID (required) | - |
| `--output` | `-o` | Output directory | Current directory |
| `--token` | - | CivitAI API token | From environment |
| `--filename` | - | Override default filename | From API |
| `--force` | - | Force re-download | False |
| `--base-url https://civitai.com/api` | - | Download from the main site | - |

## 🎯 Examples

### Download a LoRA model
```bash
./download_with_aria.py -m 245589
```

### Download multiple models to organized folders
```bash
# Download character LoRA
./download_with_aria.py -m 245589 -o ./models/lora/characters

# Download style LoRA
./download_with_aria.py -m 234567 -o ./models/lora/styles

# Download checkpoint
./download_with_aria.py -m 345678 -o ./models/checkpoints
```

### Batch download with a simple script
```bash
#!/bin/bash
# download_batch.sh

models=(245589 234567 345678 456789)
for model_id in "${models[@]}"; do
    ./download_with_aria.py -m "$model_id" -o ./models
done
```

## 🔍 How It Works

1. **Fetches Model Info** - Queries CivitAI API for filename and metadata
2. **Validates Existing Files** - Checks if valid file already exists
3. **Downloads with aria2** - Uses 8 parallel connections for speed
4. **Processes Downloaded Files**:
   - `.safetensors` - Keeps as-is
   - `.zip` - Extracts only `.safetensors` files, removes archive
   - Other formats - Keeps as downloaded
5. **Cleanup** - Removes temporary files and failed downloads

## 📊 Status Indicators

The script uses clear emoji indicators for status:

- ✅ Success - Operation completed successfully
- ❌ Error - Operation failed
- ⚠️ Warning - Important notice
- 🔍 Info - Information message
- 📥 Download - Downloading file
- 📦 Extract - Extracting archive
- 🗑️ Cleanup - Removing temporary files
- 📁 File - File operation

## 🐛 Troubleshooting

### "No CivitAI token provided"
Set your token as an environment variable or use the `--token` argument.

### "aria2c not found"
Install aria2 using the installation instructions above.

### "Download validation failed"
The file may be corrupted. Use `--force` to re-download:
```bash
./download_with_aria.py -m 123456 --force
```

### Slow downloads
CivitAI may throttle downloads. The script uses 8 connections by default for optimal speed.

### "No safetensors files found in archive"
Some models may use different formats. The original ZIP is kept in this case.

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

MIT License - feel free to use this script in your projects.

## 🙏 Acknowledgments

- [CivitAI](https://civitai.com) for providing the API and hosting models
- [aria2](https://aria2.github.io/) for the excellent download utility
- The AI art community for creating and sharing models

## 📝 Notes

- Always respect model licenses and creator terms
- Be mindful of CivitAI's rate limits and terms of service
- Large checkpoint files (>5GB) may take significant time to download
- The script requires a stable internet connection for resume to work properly

---

**Need help?** Open an issue on GitHub or check CivitAI's documentation for model-specific questions.
