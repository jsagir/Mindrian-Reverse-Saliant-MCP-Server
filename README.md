# 🧠 Mindrian Reverse Saliant Discovery MCP Server

> Discover breakthrough cross-domain innovation opportunities through dual similarity analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.10+-green.svg)](https://github.com/jlowin/fastmcp)

## 📋 Table of Contents

- [What is Reverse Salient Discovery?](#what-is-reverse-salient-discovery)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## 🎯 What is Reverse Salient Discovery?

A **Reverse Salient** is a breakthrough innovation opportunity where:

- **High Structural Similarity (LSA)**: Papers share similar methods and techniques
- **Low Semantic Similarity (BERT)**: Papers address different problems and domains

This HIGH differential indicates that **methods from one domain can be transferred to solve problems in another domain** - a powerful source of innovation!

### Example

```
Paper A: "Quantum annealing for supply chain optimization"
Paper B: "High-throughput drug combination screening"

LSA Similarity: 0.75 (HIGH - both use optimization, combinatorial methods)
BERT Similarity: 0.12 (LOW - different domains: logistics vs pharma)
Differential: 0.63 (HUGE!)

→ Innovation Opportunity: Apply quantum annealing to drug screening!
```

## ✨ Features

### Core Capabilities

- ✅ **LSA (Latent Semantic Analysis)**: Measures structural similarity via TF-IDF + SVD
- ✅ **BERT Embeddings**: Measures semantic similarity via contextual embeddings
- ✅ **Differential Analysis**: Identifies high LSA + low BERT pairs automatically
- ✅ **Sequential Thinking**: Transparent reasoning at every decision point

### Advanced Validation

- 🔍 **Patent Database Searches**: Google Patents, USPTO
- 🚀 **Startup Activity Monitoring**: Crunchbase, TechCrunch
- 📚 **Citation Network Analysis**: Google Scholar, arXiv

### Data Sources

- 🌐 **Tavily Web Search**: Multi-source academic paper collection
- 📖 **Scopus API**: Direct academic database access
- 📄 **CSV Import**: Load existing paper collections

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- API Keys:
  - **Tavily API** (required): [Get here](https://tavily.com)
  - **Scopus API** (optional): [Get here](https://dev.elsevier.com)

### Install Steps

```bash
# 1. Clone or download the repository
git clone https://github.com/mindrian/reverse-saliant-mcp.git
cd reverse-saliant-mcp

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -c "import nltk; nltk.download('stopwords')"

# 5. Configure environment variables
cp .env.example .env
# Edit .env and add your API keys
```

### Environment Setup

Create `.env` file:

```bash
TAVILY_API_KEY=your_tavily_api_key_here
SCOPUS_API_KEY=your_scopus_api_key_here  # Optional
```

## 🎮 Quick Start

### Option 1: Run Standalone

```bash
python server.py
```

### Option 2: Connect to Claude Desktop

Edit `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mindrian": {
      "command": "python",
      "args": ["C:\\path\\to\\server.py"],
      "env": {
        "TAVILY_API_KEY": "your_key",
        "SCOPUS_API_KEY": "your_key"
      }
    }
  }
}
```

Restart Claude Desktop, and you'll have access to all Mindrian tools!

## 📚 Usage Examples

### Example 1: Automated Full Workflow

```python
# In Claude Desktop, just ask:
"""
Use Mindrian to discover innovation opportunities between 
quantum computing and drug discovery
"""

# Behind the scenes, this executes:
result = await execute_full_workflow(
    structured_input={
        "challenge": "Quantum computing × Drug discovery innovation",
        "domains": [
            {
                "label": "Quantum Computing",
                "concepts": ["superposition", "entanglement"],
                "methods": ["quantum annealing", "VQE", "QAOA"]
            },
            {
                "label": "Drug Discovery",
                "concepts": ["molecular docking", "protein folding"],
                "methods": ["high-throughput screening", "computational chemistry"]
            }
        ]
    },
    search_queries=[
        "quantum computing optimization",
        "drug discovery screening methods",
        "quantum annealing applications",
        "combinatorial drug screening"
    ],
    validate_top_n=3
)
```

### Example 2: Step-by-Step Manual Control

```python
# Step 1: Initialize
result = await initialize_discovery({
    "challenge": "Find AI × Healthcare innovations",
    "domains": [...]
})
session_id = result["session_id"]

# Step 2: Collect papers
await collect_papers_tavily(
    session_id=session_id,
    search_queries=["machine learning healthcare", "AI medical diagnosis"]
)

# Step 3: Clean papers
await clean_papers(session_id)

# Step 4: Compute LSA (structural similarity)
await compute_lsa_similarity(session_id)

# Step 5: Compute BERT (semantic similarity)
await compute_bert_similarity(session_id)

# Step 6: Find reverse salients
rs_result = await find_reverse_salients(session_id, top_n=20)

# Step 7: Validate top opportunity
await validate_reverse_salient(
    session_id=session_id,
    reverse_salient_id="RS-001",
    check_patents=True,
    check_startups=True,
    check_citations=True
)

# Step 8: Develop innovation thesis
await develop_innovation_thesis(session_id, "RS-001")

# Step 9: Generate report
await generate_report(session_id, format="markdown")
```

### Example 3: Using CSV Data

```python
# Load papers from your own CSV file
await initialize_discovery({
    "challenge": "Analyze my research corpus",
    "domains": [...]
})

await load_papers_csv(
    session_id="...",
    csv_file_path="./data/my_papers.csv"
)

# Continue with normal workflow
```

## 🔧 API Reference

### Core Tools

| Tool | Purpose | Required Args |
|------|---------|---------------|
| `initialize_discovery` | Start new session | `structured_input` |
| `collect_papers_tavily` | Web search | `session_id`, `search_queries` |
| `collect_papers_scopus` | Scopus API | `session_id`, `search_terms` |
| `load_papers_csv` | Load CSV | `session_id`, `csv_file_path` |
| `clean_papers` | Clean text | `session_id` |
| `compute_lsa_similarity` | Structural similarity | `session_id` |
| `compute_bert_similarity` | Semantic similarity | `session_id` |
| `find_reverse_salients` | Detect opportunities | `session_id` |
| `validate_reverse_salient` | Advanced validation | `session_id`, `reverse_salient_id` |
| `develop_innovation_thesis` | Create thesis | `session_id`, `reverse_salient_id` |
| `generate_report` | Full report | `session_id` |
| `execute_full_workflow` | Automated pipeline | `structured_input`, `search_queries` |

### Input Format

```python
structured_input = {
    "challenge": "Description of innovation challenge",
    "domains": [
        {
            "label": "Domain Name",
            "concepts": ["concept1", "concept2", "concept3"],
            "methods": ["method1", "method2"],
            "problems": ["problem1", "problem2"],
            "terminology": ["term1", "term2"]
        }
    ],
    "constraints": ["constraint1", "constraint2"],
    "metadata": {
        "industry": "pharmaceutical",
        "timeline": "2-3 years"
    }
}
```

## 🧠 How It Works

### The Dual Similarity Framework

```
┌─────────────────────────────────────────────────────────┐
│                   PAPER COLLECTION                       │
│  Tavily Search → Scopus API → CSV Import → Cleaning     │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌────────────────┐      ┌────────────────┐
│ LSA ANALYSIS   │      │ BERT ANALYSIS  │
│ (Structural)   │      │ (Semantic)     │
├────────────────┤      ├────────────────┤
│ • TF-IDF       │      │ • Tokenization │
│ • SVD Topics   │      │ • Embeddings   │
│ • Similarity   │      │ • Cosine Sim   │
└────────┬───────┘      └───────┬────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌─────────────────────┐
         │ DIFFERENTIAL        │
         │ |BERT - LSA|        │
         │                     │
         │ HIGH = Innovation!  │
         └──────────┬──────────┘
                    │
         ┌──────────┴───────────┐
         │                      │
         ▼                      ▼
┌────────────────┐    ┌────────────────┐
│ VALIDATION     │    │ INNOVATION     │
│ • Patents      │    │ THESIS         │
│ • Startups     │    │ • Mechanism    │
│ • Citations    │    │ • Feasibility  │
└────────────────┘    └────────────────┘
```

### Sequential Thinking Integration

Every phase includes transparent reasoning:

```python
{
  "thought_number": 1,
  "thought": "Analyzing 2 domains. Need to: (1) Analyze characteristics, 
             (2) Plan search strategy, (3) Determine data collection.",
  "next_thought_needed": True,
  "timestamp": "2025-01-15T10:30:00"
}
```

All thinking logs are stored and included in final reports.

## 📊 Understanding Results

### Reverse Salient Example

```json
{
  "id": "RS-001",
  "rank": 1,
  "lsa_similarity": 0.72,
  "bert_similarity": 0.15,
  "differential_score": 0.57,
  "breakthrough_potential": 9,
  "interpretation": "High LSA, Low BERT (INNOVATION!)"
}
```

**Interpretation:**
- **LSA 0.72** = Papers use 72% similar methods
- **BERT 0.15** = Papers only 15% similar in meaning
- **Differential 0.57** = HUGE gap = strong innovation signal
- **Potential 9/10** = Highly promising opportunity

### Validation Results

```json
{
  "checks_performed": [
    {
      "check": "patents",
      "patents_found": 2,
      "novelty_score": 8,
      "status": "Clear"
    },
    {
      "check": "startups",
      "companies_found": 1,
      "market_maturity": "Early",
      "competition_level": "Low"
    },
    {
      "check": "citations",
      "recent_papers_found": 5,
      "research_activity": "Low",
      "novelty_indicator": "High novelty"
    }
  ],
  "overall_novelty_score": 9,
  "recommendation": "HIGH PRIORITY - Novel opportunity"
}
```

## 🐛 Troubleshooting

### Common Issues

**"TAVILY_API_KEY not set"**
- Solution: Create `.env` file with your API key

**"No papers collected"**
- Check your API key is valid
- Try broader search queries
- Verify internet connection

**"BERT computation very slow"**
- This is normal for 50+ papers
- Use `bert-base-uncased` instead of `bert-large-uncased`
- Consider running on GPU if available

**"LSA matrix all zeros"**
- Papers might be too short
- Try different `max_features` parameter
- Check cleaned papers aren't empty

### Performance Tips

- **50 papers**: ~5 minutes total
- **100 papers**: ~10 minutes total
- **500 papers**: ~60 minutes total
- Use CSV import for repeated runs
- Cache LSA/BERT matrices if rerunning

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- Original research methodology from LSA/BERT dual similarity analysis
- FastMCP framework by Anthropic
- Tavily Search API
- Elsevier Scopus API

## 📧 Contact

For questions, issues, or collaboration:
- GitHub Issues: [github.com/mindrian/reverse-saliant-mcp/issues](https://github.com/mindrian/reverse-saliant-mcp/issues)
- Email: contact@mindrian.com

---

**Built with ❤️ by Mindrian Labs**

*Discover the innovations hiding between domains.*
