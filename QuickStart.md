# ⚡ Mindrian Reverse Saliant - Quick Start Guide

Get up and running in 5 minutes!

## 📦 Step 1: Download All Files

Create a new folder and save these files:

```
mindrian-reverse-saliant/
├── server.py              # Main MCP server
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
├── .env.example           # Environment template
├── .gitignore             # Git ignore rules
├── README.md              # Full documentation
└── QUICKSTART.md          # This file
```

## 🔧 Step 2: Setup Environment

### Windows

```cmd
REM Navigate to folder
cd mindrian-reverse-saliant

REM Create virtual environment
python -m venv venv
venv\Scripts\activate

REM Install dependencies
pip install -r requirements.txt

REM Download NLTK data
python -c "import nltk; nltk.download('stopwords')"

REM Copy environment template
copy .env.example .env
```

### Mac/Linux

```bash
# Navigate to folder
cd mindrian-reverse-saliant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords')"

# Copy environment template
cp .env.example .env
```

## 🔑 Step 3: Add API Keys

Edit `.env` file:

```bash
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxxxxxxx
SCOPUS_API_KEY=your_scopus_key_here  # Optional
```

### Get API Keys:

- **Tavily**: Sign up at https://tavily.com (Free tier available)
- **Scopus**: Register at https://dev.elsevier.com (Optional, for academic papers)

## ✅ Step 4: Test the Server

```bash
# Run the server
python server.py

# You should see:
# ============================================================
# Mindrian Reverse Saliant Discovery MCP Server
# Version 1.0.0
# ============================================================
# 
# Features:
#   ✓ LSA Structural Similarity
#   ✓ BERT Semantic Similarity
#   ✓ Sequential Thinking Integration
#   ✓ Patent Database Searches
#   ✓ Startup Activity Monitoring
#   ✓ Citation Network Analysis
```

## 🔌 Step 5: Connect to Claude Desktop

### Windows

Edit: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mindrian": {
      "command": "python",
      "args": ["C:\\Users\\YourName\\mindrian-reverse-saliant\\server.py"],
      "env": {
        "TAVILY_API_KEY": "tvly-your-key-here",
        "SCOPUS_API_KEY": "your-scopus-key"
      }
    }
  }
}
```

### Mac

Edit: `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mindrian": {
      "command": "python3",
      "args": ["/Users/yourname/mindrian-reverse-saliant/server.py"],
      "env": {
        "TAVILY_API_KEY": "tvly-your-key-here",
        "SCOPUS_API_KEY": "your-scopus-key"
      }
    }
  }
}
```

### Linux

Edit: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mindrian": {
      "command": "python3",
      "args": ["/home/yourname/mindrian-reverse-saliant/server.py"],
      "env": {
        "TAVILY_API_KEY": "tvly-your-key-here",
        "SCOPUS_API_KEY": "your-scopus-key"
      }
    }
  }
}
```

**Important:** Replace the paths with your actual full paths!

## 🚀 Step 6: Use It!

Restart Claude Desktop, then try:

```
Use Mindrian to discover innovation opportunities between 
quantum computing and drug discovery
```

Claude will:
1. ✅ Search for papers in both domains
2. ✅ Compute LSA (structural similarity)
3. ✅ Compute BERT (semantic similarity)
4. ✅ Find reverse salients (high LSA + low BERT)
5. ✅ Validate with patents, startups, citations
6. ✅ Generate innovation thesis

## 📋 Common Commands

### In Claude Desktop

```
# Full automated workflow
"Use Mindrian to analyze AI and healthcare innovations"

# Step-by-step
"Use Mindrian initialize_discovery with these domains: [...]"
"Use Mindrian collect_papers_tavily with these queries: [...]"
"Use Mindrian find_reverse_salients"

# Validation
"Use Mindrian validate_reverse_salient for RS-001"

# Reports
"Use Mindrian generate_report in markdown format"
```

## 🎯 Example Use Cases

### 1. Technology Transfer
```
Discover how quantum computing methods can solve 
pharmaceutical challenges
```

### 2. Cross-Industry Innovation
```
Find opportunities between AI and manufacturing
```

### 3. Academic Research
```
Analyze papers on climate change and urban planning 
for breakthrough intersections
```

### 4. Competitive Intelligence
```
Identify unexplored innovation spaces in fintech 
based on blockchain and traditional banking
```

## 🐛 Troubleshooting

### Server won't start
- Check Python version: `python --version` (needs 3.11+)
- Verify all dependencies: `pip list`
- Check .env file exists with API keys

### Claude can't see the tools
- Verify path in `claude_desktop_config.json` is correct
- Restart Claude Desktop completely
- Check Claude logs (Help → View Logs)

### "TAVILY_API_KEY not set"
- Ensure `.env` file exists
- Check API key format (should start with `tvly-`)
- Try setting in `claude_desktop_config.json` directly

### Slow BERT computation
- Normal for 50+ papers (5-10 minutes)
- Use `bert-base-uncased` instead of `bert-large`
- Consider GPU acceleration for large datasets

## 📚 Next Steps

- Read full [README.md](README.md) for detailed documentation
- Check [API Reference](README.md#api-reference) for all tools
- See [Usage Examples](README.md#usage-examples) for advanced use cases
- Join discussions on GitHub Issues

## 💡 Pro Tips

1. **Start small**: Test with 20-30 papers first
2. **Use specific queries**: "quantum annealing optimization" > "quantum computing"
3. **Validate top results**: Always check RS-001, RS-002, RS-003
4. **Cache results**: Use CSV export for repeated analysis
5. **Combine sources**: Tavily + Scopus = comprehensive coverage

---

**Need help?** Open an issue on GitHub or email contact@mindrian.com

**Ready to discover breakthrough innovations!** 🚀
