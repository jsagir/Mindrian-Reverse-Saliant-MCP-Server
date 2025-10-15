# 📁 Mindrian Reverse Saliant - File Structure

## Complete Project Structure

```
mindrian-reverse-saliant/
│
├── 📄 server.py                    # Main MCP server (1,200+ lines)
│   ├── FastMCP initialization
│   ├── DiscoverySession class
│   ├── ThinkingLogger class
│   ├── 12 MCP tools:
│   │   ├── initialize_discovery
│   │   ├── collect_papers_tavily
│   │   ├── collect_papers_scopus
│   │   ├── load_papers_csv
│   │   ├── clean_papers
│   │   ├── compute_lsa_similarity
│   │   ├── compute_bert_similarity
│   │   ├── find_reverse_salients
│   │   ├── validate_reverse_salient
│   │   ├── develop_innovation_thesis
│   │   ├── generate_report
│   │   └── execute_full_workflow
│   └── Main entry point
│
├── 📋 requirements.txt             # Python dependencies
│   ├── fastmcp>=2.10.0
│   ├── httpx>=0.25.0
│   ├── numpy>=1.24.0
│   ├── pandas>=2.0.0
│   ├── nltk>=3.8.0
│   ├── scikit-learn>=1.3.0
│   ├── torch>=2.0.0
│   ├── transformers>=4.30.0
│   └── matplotlib>=3.7.0
│
├── ⚙️ pyproject.toml               # Project metadata & FastMCP config
│   ├── Project info
│   ├── Dependencies
│   └── FastMCP configuration
│
├── 🔐 .env.example                 # Environment variables template
│   ├── TAVILY_API_KEY
│   └── SCOPUS_API_KEY
│
├── 🚫 .gitignore                   # Git ignore rules
│   ├── Python artifacts
│   ├── Virtual environments
│   ├── Data files
│   └── API keys
│
├── 📖 README.md                    # Complete documentation (300+ lines)
│   ├── What is Reverse Salient Discovery?
│   ├── Features overview
│   ├── Installation guide
│   ├── Usage examples
│   ├── API reference
│   ├── How it works
│   ├── Understanding results
│   └── Troubleshooting
│
├── ⚡ QUICKSTART.md                # 5-minute setup guide
│   ├── Step-by-step installation
│   ├── API key setup
│   ├── Claude Desktop integration
│   ├── First commands
│   └── Pro tips
│
├── 📜 LICENSE                      # MIT License
│
└── 📁 FILE_STRUCTURE.md            # This file
```

## Key Files Explained

### 1. server.py (Main Server)

**Purpose**: Complete MCP server implementation

**Key Components**:
- `DiscoverySession`: Manages state for each discovery session
- `ThinkingLogger`: Logs sequential thinking steps
- 12 MCP tools for the complete workflow
- Main entry point with stdio transport

**Size**: ~1,200 lines
**Language**: Python 3.11+

### 2. requirements.txt

**Purpose**: Python package dependencies

**Key Packages**:
- `fastmcp`: MCP server framework
- `torch` + `transformers`: BERT embeddings
- `scikit-learn`: LSA/TF-IDF/SVD
- `httpx`: API calls (Tavily, Scopus)
- `numpy` + `pandas`: Data processing

### 3. pyproject.toml

**Purpose**: Project metadata and configuration

**Sections**:
- `[project]`: Name, version, description, dependencies
- `[tool.fastmcp]`: FastMCP-specific settings
- `[build-system]`: Build configuration

### 4. .env.example

**Purpose**: Template for environment variables

**Variables**:
- `TAVILY_API_KEY`: Required for web search
- `SCOPUS_API_KEY`: Optional for academic papers

**Usage**: Copy to `.env` and fill in your keys

### 5. .gitignore

**Purpose**: Prevent sensitive files from being committed

**Excludes**:
- `.env` (API keys)
- `__pycache__/` (Python cache)
- `venv/` (Virtual environment)
- `*.csv`, `*.npy` (Data files)

### 6. README.md

**Purpose**: Complete documentation

**Sections**:
- Introduction and overview
- Installation instructions
- Usage examples (3 detailed examples)
- API reference for all 12 tools
- How it works (architecture diagram)
- Result interpretation
- Troubleshooting guide

### 7. QUICKSTART.md

**Purpose**: Get users running in 5 minutes

**Sections**:
- Fast installation (Windows/Mac/Linux)
- API key setup
- Claude Desktop connection
- First commands to try
- Common troubleshooting

### 8. LICENSE

**Purpose**: MIT open-source license

**Permissions**: Use, modify, distribute freely

### 9. FILE_STRUCTURE.md

**Purpose**: This document - project overview

## Installation Files You'll Create

When you set up the project, you'll also have:

```
mindrian-reverse-sallient/
├── venv/                           # Virtual environment (gitignored)
├── .env                            # Your API keys (gitignored)
├── nltk_data/                      # NLTK stopwords data
└── __pycache__/                    # Python cache (gitignored)
```

## Data Files (Optional)

If you use CSV import or save results:

```
mindrian-reverse-sallient/
├── data/
│   ├── papers.csv                  # Your input papers
│   └── scopus_export.csv           # Scopus data
│
└── outputs/
    ├── reports/
    │   └── session_20250115.json   # Discovery reports
    └── matrices/
        ├── lsa_matrix.npy          # LSA similarity
        └── bert_matrix.npy         # BERT similarity
```

## Claude Desktop Integration

Your Claude config file will reference:

**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Mac**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

## File Sizes

| File | Lines | Size |
|------|-------|------|
| server.py | 1,200+ | ~45 KB |
| README.md | 350+ | ~18 KB |
| requirements.txt | 20 | ~1 KB |
| pyproject.toml | 60 | ~2 KB |
| QUICKSTART.md | 200+ | ~8 KB |
| Others | <100 | <5 KB |
| **Total** | **~2,000** | **~80 KB** |

## Dependencies Size (After Installation)

- Virtual environment: ~500 MB (torch, transformers)
- NLTK data: ~1 MB (stopwords)
- Total disk space: ~500 MB

## Development Workflow

```
1. Edit server.py          → Modify functionality
2. Update README.md        → Document changes
3. Test locally            → python server.py
4. Update version          → pyproject.toml
5. Commit changes          → git commit
6. Push to repository      → git push
```

## Production Deployment

For production use, add:

```
mindrian-reverse-sallient/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   ├── test_lsa.py
│   ├── test_bert.py
│   └── test_validation.py
└── docs/
    ├── API.md
    └── CONTRIBUTING.md
```

---

**All files ready to copy-paste!** Just create the folder structure and paste each file's contents.
