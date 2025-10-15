#!/usr/bin/env python3
"""
Mindrian Reverse Saliant MCP Server
Discover cross-domain innovation opportunities through dual similarity analysis

Version: 1.0.0
Author: Mindrian Labs
"""

from fastmcp import FastMCP
from typing import Dict, Any, List, Optional
import asyncio
import json
import os
from datetime import datetime
import httpx
from collections import defaultdict
import csv
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ============================================================================
# SERVER CONFIGURATION
# ============================================================================

mcp = FastMCP(
    name="mindrian-reverse-saliant",
    instructions="""
    Mindrian Reverse Saliant Discovery MCP Server
    
    Discovers breakthrough innovation opportunities through:
    - LSA (Structural Similarity): Shared methods and techniques
    - BERT (Semantic Similarity): Shared meaning and context
    - Differential Analysis: High LSA + Low BERT = Innovation!
    - Sequential Thinking: Transparent reasoning at every step
    - Advanced Validation: Patents, startups, citations
    
    Author: Mindrian Labs
    """
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

discovery_sessions = {}

class DiscoverySession:
    """Maintains state for a discovery session."""
    
    def __init__(self, session_id: str, structured_input: Dict[str, Any]):
        self.session_id = session_id
        self.input = structured_input
        self.domains = []
        self.papers = []
        self.cleaned_papers = []
        self.lsa_matrix = None
        self.bert_matrix = None
        self.difference_matrix = None
        self.reverse_salients = []
        self.opportunities = []
        self.thinking_logs = []
        self.validation_results = {}
        self.phase = "initialized"
        self.created_at = datetime.now().isoformat()

class ThinkingLogger:
    """Logs sequential thinking steps."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.thoughts = []
        self.current_total = 5
    
    def think(self, thought: str, step: int, next_needed: bool = True, 
              is_revision: bool = False, revises_step: Optional[int] = None):
        entry = {
            "thought_number": step,
            "total_thoughts": self.current_total,
            "thought": thought,
            "next_thought_needed": next_needed,
            "is_revision": is_revision,
            "timestamp": datetime.now().isoformat()
        }
        
        if revises_step:
            entry["revises_thought"] = revises_step
        
        self.thoughts.append(entry)
        
        session = discovery_sessions.get(self.session_id)
        if session:
            session.thinking_logs.append(entry)
        
        return entry
    
    def adjust_total(self, new_total: int):
        self.current_total = new_total
    
    def get_summary(self):
        return {
            "total_steps": len(self.thoughts),
            "thoughts": self.thoughts
        }

# ============================================================================
# PHASE 1: INITIALIZATION
# ============================================================================

@mcp.tool()
async def initialize_discovery(
    structured_input: Dict[str, Any],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Initialize Mindrian Reverse Salient Discovery session.
    
    Args:
        structured_input: {
            "challenge": "Description of the innovation challenge",
            "domains": [
                {
                    "label": "Domain Name",
                    "concepts": ["concept1", "concept2"],
                    "methods": ["method1", "method2"],
                    "problems": ["problem1", "problem2"],
                    "terminology": ["term1", "term2"]
                }
            ],
            "constraints": ["constraint1", "constraint2"],
            "metadata": {...}
        }
        session_id: Optional session identifier
    
    Returns:
        Session information and analysis plan
    """
    
    if not session_id:
        session_id = f"mindrian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    session = DiscoverySession(session_id, structured_input)
    discovery_sessions[session_id] = session
    
    domains = structured_input.get("domains", [])
    session.domains = domains
    
    # Sequential Thinking: Analyze the challenge
    thinker = ThinkingLogger(session_id)
    
    thinker.think(
        f"Initializing Mindrian discovery for: {structured_input.get('challenge', 'N/A')}. "
        f"Found {len(domains)} domains. Need to: (1) Analyze domain characteristics, "
        f"(2) Plan search strategy, (3) Determine data collection approach.",
        step=1,
        next_needed=True
    )
    
    for i, domain in enumerate(domains):
        concepts_count = len(domain.get("concepts", []))
        methods_count = len(domain.get("methods", []))
        
        thinker.think(
            f"Domain {i+1} '{domain.get('label')}': {concepts_count} concepts, "
            f"{methods_count} methods. Transferability potential: "
            f"{'High' if methods_count > 3 else 'Medium' if methods_count > 1 else 'Low'}.",
            step=i+2,
            next_needed=i < len(domains) - 1
        )
    
    total_step = len(domains) + 2
    recommended_searches = len(domains) * 3
    
    thinker.think(
        f"Search strategy: Need {recommended_searches} queries - "
        f"{len(domains)} intersection, {len(domains)} isolated, {len(domains)} method-transfer. "
        f"Estimated papers to collect: ~{recommended_searches * 15}.",
        step=total_step,
        next_needed=False
    )
    
    return {
        "session_id": session_id,
        "domains_found": len(domains),
        "domain_labels": [d.get("label") for d in domains],
        "recommended_searches": recommended_searches,
        "thinking_summary": thinker.get_summary(),
        "phase": "initialized",
        "next_steps": [
            "collect_papers_tavily", 
            "collect_papers_scopus", 
            "load_papers_csv"
        ]
    }

# ============================================================================
# PHASE 2: DATA COLLECTION
# ============================================================================

@mcp.tool()
async def collect_papers_tavily(
    session_id: str,
    search_queries: List[str],
    max_results_per_query: int = 20
) -> Dict[str, Any]:
    """
    Collect papers using Tavily web search.
    
    Args:
        session_id: Discovery session ID
        search_queries: List of search queries
        max_results_per_query: Maximum results per query
    
    Returns:
        Collection results and thinking summary
    """
    
    session = discovery_sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return {"error": "TAVILY_API_KEY not set in environment"}
    
    thinker = ThinkingLogger(session_id)
    
    thinker.think(
        f"Planning {len(search_queries)} Tavily searches with {max_results_per_query} results each.",
        step=1,
        next_needed=True
    )
    
    all_papers = []
    
    async with httpx.AsyncClient() as client:
        for idx, query in enumerate(search_queries):
            thinker.think(
                f"Executing search {idx+1}/{len(search_queries)}: '{query}'",
                step=idx+2,
                next_needed=idx < len(search_queries) - 1
            )
            
            try:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": tavily_api_key,
                        "query": query,
                        "max_results": max_results_per_query,
                        "search_depth": "advanced",
                        "include_domains": [
                            "arxiv.org", "nature.com", "science.org",
                            "ieee.org", "pubmed.ncbi.nlm.nih.gov"
                        ]
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for result in data.get("results", []):
                        paper_text = f"{result.get('title', '')} {result.get('content', '')}"
                        all_papers.append({
                            "text": paper_text,
                            "title": result.get("title"),
                            "url": result.get("url"),
                            "source": "tavily",
                            "query": query
                        })
            except Exception as e:
                print(f"Error searching '{query}': {e}")
    
    thinker.think(
        f"Collection complete: {len(all_papers)} papers collected.",
        step=len(search_queries) + 2,
        next_needed=False
    )
    
    session.papers = all_papers
    session.phase = "papers_collected"
    
    return {
        "session_id": session_id,
        "papers_collected": len(all_papers),
        "queries_executed": len(search_queries),
        "thinking_summary": thinker.get_summary(),
        "phase": "papers_collected"
    }

@mcp.tool()
async def collect_papers_scopus(
    session_id: str,
    search_terms: str,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Collect papers from Elsevier Scopus API.
    
    Args:
        session_id: Discovery session ID
        search_terms: Scopus query string
        api_key: Scopus API key (or from SCOPUS_API_KEY env var)
    
    Returns:
        Collection results
    """
    
    session = discovery_sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}
    
    if not api_key:
        api_key = os.getenv("SCOPUS_API_KEY")
    
    if not api_key:
        return {"error": "SCOPUS_API_KEY not set"}
    
    search_url = 'https://api.elsevier.com/content/search/scopus'
    abstract_url = 'https://api.elsevier.com/content/abstract/doi/'
    
    headers = {
        'Accept': 'application/json',
        'X-ELS-APIKey': api_key
    }
    
    search_params = {
        'query': search_terms,
        'view': 'COMPLETE'
    }
    
    papers = []
    
    async with httpx.AsyncClient() as client:
        search_response = await client.get(search_url, headers=headers, params=search_params)
        
        if search_response.status_code == 200:
            search_data = search_response.json()
            
            if 'search-results' in search_data and 'entry' in search_data['search-results']:
                entries = search_data['search-results']['entry']
                
                for entry in entries:
                    doi = entry.get('prism:doi', None)
                    title = entry.get('dc:title', 'No title')
                    
                    if doi:
                        abstract_response = await client.get(
                            abstract_url + doi,
                            headers=headers
                        )
                        
                        if abstract_response.status_code == 200:
                            abstract_data = abstract_response.json()
                            abstract = abstract_data.get(
                                'abstracts-retrieval-response', {}
                            ).get('coredata', {}).get('dc:description', 'No abstract')
                            
                            paper_text = f"{title} {abstract}"
                            papers.append({
                                "text": paper_text,
                                "title": title,
                                "doi": doi,
                                "source": "scopus"
                            })
    
    session.papers.extend(papers)
    session.phase = "papers_collected"
    
    return {
        "session_id": session_id,
        "papers_collected": len(papers),
        "phase": "papers_collected"
    }

@mcp.tool()
async def load_papers_csv(
    session_id: str,
    csv_file_path: str
) -> Dict[str, Any]:
    """
    Load papers from CSV file.
    
    Args:
        session_id: Discovery session ID
        csv_file_path: Path to CSV file
    
    Returns:
        Loading results
    """
    
    session = discovery_sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}
    
    papers = []
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                row_text = str(row)
                
                if "No abstract available" in row_text or "NO-ABSTRACT" in row_text:
                    continue
                
                papers.append({
                    "text": row_text,
                    "source": "csv"
                })
        
        if papers:
            papers.pop(0)  # Remove header
        
        session.papers = papers
        session.phase = "papers_collected"
        
        return {
            "session_id": session_id,
            "papers_loaded": len(papers),
            "phase": "papers_collected"
        }
    
    except Exception as e:
        return {"error": f"Failed to load CSV: {e}"}

@mcp.tool()
async def clean_papers(session_id: str) -> Dict[str, Any]:
    """
    Clean collected papers (remove URLs, copyright, special chars).
    
    Args:
        session_id: Discovery session ID
    
    Returns:
        Cleaning results
    """
    
    session = discovery_sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}
    
    if not session.papers:
        return {"error": "No papers to clean"}
    
    cleaned_papers = []
    
    for paper in session.papers:
        text = paper.get("text", "")
        
        text = text.replace("[", "").replace("]", "")
        text = text.replace("'", "").replace('"', "")
        text = text.replace("Â©", "COPYRIGHT")
        text = re.sub(r'http\S*,', '', text)
        text = re.sub(r'Copyright.*', '', text)
        text = re.sub(r'COPYRIGHT.*', '', text)
        
        if text.strip():
            cleaned_papers.append(text.strip())
    
    session.cleaned_papers = cleaned_papers
    session.phase = "papers_cleaned"
    
    return {
        "session_id": session_id,
        "papers_cleaned": len(cleaned_papers),
        "phase": "papers_cleaned"
    }

# ============================================================================
# PHASE 3: LSA STRUCTURAL SIMILARITY
# ============================================================================

@mcp.tool()
async def compute_lsa_similarity(
    session_id: str,
    n_components: int = 80,
    max_features: int = 2000
) -> Dict[str, Any]:
    """
    Compute STRUCTURAL similarity using LSA (TF-IDF + SVD).
    Measures shared methods, techniques, and terminology.
    
    Args:
        session_id: Discovery session ID
        n_components: Number of SVD components (topics)
        max_features: Maximum TF-IDF features
    
    Returns:
        LSA results with thinking summary
    """
    
    session = discovery_sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}
    
    if not session.cleaned_papers:
        return {"error": "No cleaned papers"}
    
    papers = session.cleaned_papers
    
    thinker = ThinkingLogger(session_id)
    
    thinker.think(
        f"Computing LSA on {len(papers)} papers. Using TF-IDF with {max_features} features "
        f"and SVD with {n_components} components. This captures STRUCTURAL similarity.",
        step=1,
        next_needed=True
    )
    
    # Remove stopwords
    stop_words = stopwords.words('english')
    final_papers = []
    
    for paper in papers:
        tokens = paper.split(" ")
        filtered_tokens = [t for t in tokens if t not in stop_words]
        final_papers.append(" ".join(filtered_tokens))
    
    # TF-IDF
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=max_features,
        max_df=0.5,
        smooth_idf=True
    )
    
    X = vectorizer.fit_transform(final_papers)
    
    # SVD
    svd_model = TruncatedSVD(
        n_components=n_components,
        algorithm='randomized',
        n_iter=10,
        random_state=256
    )
    
    svd_model.fit(X)
    
    thinker.think(
        f"SVD complete. Extracting {n_components} latent topics.",
        step=2,
        next_needed=True
    )
    
    # Extract topics
    terms = vectorizer.get_feature_names_out()
    topics = []
    
    for i, comp in enumerate(svd_model.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:7]
        topic_terms = [t[0] for t in sorted_terms]
        topics.append(topic_terms)
    
    # Compute paper-topic matrix
    paper_topics_count = []
    
    for paper in papers:
        tokens = paper.split(" ")
        category_list = [0] * len(topics)
        
        for word in tokens:
            for i, topic in enumerate(topics):
                if word in topic:
                    category_list[i] += 1
        
        paper_topics_count.append(category_list)
    
    paper_topic_matrix = np.array(paper_topics_count).astype('float32')
    
    # Normalize
    sum_of_row = np.sum(paper_topic_matrix, axis=1)
    divide_by_row = np.nan_to_num(np.divide(1, sum_of_row), nan=0)
    result = np.einsum('i, ik -> ik', divide_by_row, paper_topic_matrix)
    
    # Compute similarity matrix
    similarity_matrix_double = np.broadcast_to(
        result,
        (result.shape[0], result.shape[0], result.shape[1])
    )
    similarity_matrix_double_dual = np.einsum('ijk -> jik', similarity_matrix_double)
    
    similarity_matrix_cats = np.abs(similarity_matrix_double - similarity_matrix_double_dual)
    sum_matrix = np.sum(similarity_matrix_cats, axis=2)
    sum_matrix = np.max(sum_matrix) - sum_matrix
    
    # Normalize to [0, 1]
    min_max_average = (np.max(sum_matrix) + np.min(sum_matrix)) / 2
    scale_factor = (np.max(sum_matrix) - np.min(sum_matrix))
    lsa_matrix = ((sum_matrix - min_max_average) / scale_factor) + 0.5
    
    lsa_mean = float(np.mean(lsa_matrix))
    lsa_std = float(np.std(lsa_matrix))
    
    thinker.think(
        f"LSA matrix computed. Mean: {lsa_mean:.3f}, StdDev: {lsa_std:.3f}. "
        f"High values = similar methods.",
        step=3,
        next_needed=False
    )
    
    session.lsa_matrix = lsa_matrix
    session.phase = "lsa_computed"
    
    return {
        "session_id": session_id,
        "matrix_shape": list(lsa_matrix.shape),
        "topics_found": len(topics),
        "top_5_topics": topics[:5],
        "mean_similarity": lsa_mean,
        "std_similarity": lsa_std,
        "thinking_summary": thinker.get_summary(),
        "interpretation": "LSA captures STRUCTURAL similarity",
        "phase": "lsa_computed"
    }

# ============================================================================
# PHASE 4: BERT SEMANTIC SIMILARITY
# ============================================================================

@mcp.tool()
async def compute_bert_similarity(
    session_id: str,
    bert_model: str = "bert-base-uncased"
) -> Dict[str, Any]:
    """
    Compute SEMANTIC similarity using BERT embeddings.
    Measures shared meaning, context, and application domains.
    
    Args:
        session_id: Discovery session ID
        bert_model: BERT model to use
    
    Returns:
        BERT results with thinking summary
    """
    
    session = discovery_sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}
    
    if not session.cleaned_papers:
        return {"error": "No cleaned papers"}
    
    papers = session.cleaned_papers
    
    thinker = ThinkingLogger(session_id)
    
    thinker.think(
        f"Computing BERT embeddings on {len(papers)} papers. "
        f"This captures SEMANTIC similarity.",
        step=1,
        next_needed=True
    )
    
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    model = BertModel.from_pretrained(bert_model)
    
    def predict_to_embedding(model, segment):
        predicted_segment = model(segment)
        embedding_segment = predicted_segment.last_hidden_state[:, 0, :]
        return embedding_segment
    
    def predict_by_segments(model, tokenized_text):
        size = tokenized_text.shape[1]
        index = 0
        segments = []
        
        while size >= 512:
            segment = tokenized_text[:, index:index+512]
            segments.append(predict_to_embedding(model, segment))
            index += 512
            size -= 512
        
        if size > 0:
            segment = tokenized_text[:, index:]
            segments.append(predict_to_embedding(model, segment))
        
        combined = torch.cat(segments, dim=0)
        return combined
    
    embeddings_list = []
    
    with torch.no_grad():
        for i, paper in enumerate(papers):
            tokenized_paper = tokenizer.tokenize(paper)
            tokenized_ids = torch.tensor(
                tokenizer.convert_tokens_to_ids(tokenized_paper)
            ).unsqueeze(0)
            
            embedding = predict_by_segments(model, tokenized_ids)
            embeddings_list.append(embedding)
            
            if (i + 1) % 10 == 0:
                print(f"Embedded {i+1}/{len(papers)} papers")
    
    thinker.think(
        f"Embeddings complete. Computing pairwise similarities...",
        step=2,
        next_needed=True
    )
    
    n_papers = len(embeddings_list)
    similarity_matrix = np.zeros((n_papers, n_papers))
    
    for i in range(n_papers):
        for j in range(n_papers):
            embed_i = embeddings_list[i]
            embed_j = embeddings_list[j]
            
            piecewise_similarities = []
            
            for k in range(embed_i.shape[0]):
                for l in range(embed_j.shape[0]):
                    sim = cosine_similarity(
                        embed_i[k].unsqueeze(0),
                        embed_j[l].unsqueeze(0)
                    )
                    piecewise_similarities.append(torch.tensor(sim))
            
            total_similarity = torch.mean(torch.cat(piecewise_similarities))
            similarity_matrix[i, j] = total_similarity.item()
        
        if (i + 1) % 10 == 0:
            print(f"Compared paper {i+1}/{n_papers}")
    
    similarity_matrix = (
        (similarity_matrix - np.mean(similarity_matrix)) / 
        np.max(np.abs(similarity_matrix))
    ) + 0.5
    
    min_max_average = (np.max(similarity_matrix) + np.min(similarity_matrix)) / 2
    scale_factor = (np.max(similarity_matrix) - np.min(similarity_matrix))
    bert_matrix = ((similarity_matrix - min_max_average) / scale_factor) + 0.5
    
    bert_mean = float(np.mean(bert_matrix))
    bert_std = float(np.std(bert_matrix))
    
    thinker.think(
        f"BERT matrix computed. Mean: {bert_mean:.3f}, StdDev: {bert_std:.3f}. "
        f"High values = similar problems/domains.",
        step=3,
        next_needed=False
    )
    
    session.bert_matrix = bert_matrix
    session.phase = "bert_computed"
    
    return {
        "session_id": session_id,
        "matrix_shape": list(bert_matrix.shape),
        "mean_similarity": bert_mean,
        "std_similarity": bert_std,
        "thinking_summary": thinker.get_summary(),
        "interpretation": "BERT captures SEMANTIC similarity",
        "phase": "bert_computed"
    }

# ============================================================================
# PHASE 5: REVERSE SALIENT DETECTION
# ============================================================================

@mcp.tool()
async def find_reverse_salients(
    session_id: str,
    top_n: int = 20
) -> Dict[str, Any]:
    """
    Find reverse salients through differential analysis.
    High LSA + Low BERT = Innovation Opportunity!
    
    Args:
        session_id: Discovery session ID
        top_n: Number of top opportunities to return
    
    Returns:
        Reverse salients with thinking summary
    """
    
    session = discovery_sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}
    
    if session.lsa_matrix is None or session.bert_matrix is None:
        return {"error": "Compute similarity matrices first"}
    
    lsa_matrix = session.lsa_matrix
    bert_matrix = session.bert_matrix
    
    thinker = ThinkingLogger(session_id)
    thinker.adjust_total(5)
    
    thinker.think(
        f"Analyzing differential: |BERT - LSA|. Looking for HIGH LSA + LOW BERT.",
        step=1,
        next_needed=True
    )
    
    difference_matrix = bert_matrix - lsa_matrix
    abs_diff_matrix = np.abs(difference_matrix)
    np.fill_diagonal(abs_diff_matrix, 0)
    
    diff_mean = float(np.mean(abs_diff_matrix[abs_diff_matrix > 0]))
    diff_max = float(np.max(abs_diff_matrix))
    
    thinker.think(
        f"Differential stats: Mean={diff_mean:.3f}, Max={diff_max:.3f}.",
        step=2,
        next_needed=True
    )
    
    reverse_salients = []
    abs_diff_copy = abs_diff_matrix.copy()
    
    high_lsa_low_bert = np.sum((lsa_matrix > 0.5) & (bert_matrix < 0.3))
    
    thinker.think(
        f"Found {high_lsa_low_bert} structural transfer opportunities.",
        step=3,
        next_needed=True
    )
    
    for rank in range(1, top_n + 1):
        flat_index = np.argmax(abs_diff_copy)
        i, j = np.unravel_index(flat_index, shape=abs_diff_copy.shape)
        max_val = abs_diff_copy[i, j]
        
        if max_val == 0:
            break
        
        rs = {
            "rank": rank,
            "id": f"RS-{rank:03d}",
            "paper_a_index": int(i),
            "paper_b_index": int(j),
            "paper_a_text": session.cleaned_papers[i][:200] + "...",
            "paper_b_text": session.cleaned_papers[j][:200] + "...",
            "lsa_similarity": float(lsa_matrix[i, j]),
            "bert_similarity": float(bert_matrix[i, j]),
            "differential_score": float(max_val),
            "interpretation": "High BERT, Low LSA" if difference_matrix[i, j] > 0 
                             else "High LSA, Low BERT (INNOVATION!)",
            "breakthrough_potential": min(10, int((max_val * 10) + (lsa_matrix[i, j] * 5)))
        }
        
        reverse_salients.append(rs)
        abs_diff_copy[i, j] = 0
        abs_diff_copy[j, i] = 0
    
    thinker.think(
        f"Identified {len(reverse_salients)} reverse salients.",
        step=4,
        next_needed=False
    )
    
    session.difference_matrix = difference_matrix
    session.reverse_salients = reverse_salients
    session.phase = "reverse_salients_found"
    
    return {
        "session_id": session_id,
        "reverse_salients_found": len(reverse_salients),
        "top_opportunities": reverse_salients[:10],
        "thinking_summary": thinker.get_summary(),
        "key_insight": "High LSA + Low BERT = methods transferable to new domains",
        "phase": "reverse_salients_found"
    }

# ============================================================================
# PHASE 6: ADVANCED VALIDATION
# ============================================================================

@mcp.tool()
async def validate_reverse_salient(
    session_id: str,
    reverse_salient_id: str,
    check_patents: bool = True,
    check_startups: bool = True,
    check_citations: bool = True
) -> Dict[str, Any]:
    """
    Advanced validation with patents, startups, and citation analysis.
    
    Args:
        session_id: Discovery session ID
        reverse_salient_id: RS ID to validate (e.g., "RS-001")
        check_patents: Search patent databases
        check_startups: Search startup activity
        check_citations: Analyze citation networks
    
    Returns:
        Validation results with novelty score
    """
    
    session = discovery_sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}
    
    rs = next((r for r in session.reverse_salients if r["id"] == reverse_salient_id), None)
    if not rs:
        return {"error": "Reverse salient not found"}
    
    thinker = ThinkingLogger(session_id)
    
    thinker.think(
        f"Beginning validation for {reverse_salient_id}.",
        step=1,
        next_needed=True
    )
    
    validation_results = {
        "reverse_salient_id": reverse_salient_id,
        "validation_timestamp": datetime.now().isoformat(),
        "checks_performed": []
    }
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    paper_a_terms = rs["paper_a_text"][:100]
    paper_b_terms = rs["paper_b_text"][:100]
    
    async with httpx.AsyncClient() as client:
        
        # Patent Search
        if check_patents and tavily_api_key:
            thinker.think("Searching patent databases...", step=2, next_needed=True)
            
            patent_queries = [
                f"site:patents.google.com {paper_a_terms} {paper_b_terms}",
                f"patent {paper_a_terms} {paper_b_terms}"
            ]
            
            patent_results = []
            
            for query in patent_queries:
                try:
                    response = await client.post(
                        "https://api.tavily.com/search",
                        json={
                            "api_key": tavily_api_key,
                            "query": query,
                            "max_results": 5,
                            "search_depth": "basic"
                        },
                        timeout=15.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        patent_results.extend([{
                            "title": r.get("title"),
                            "url": r.get("url")
                        } for r in results])
                except:
                    pass
            
            patent_analysis = {
                "check": "patents",
                "patents_found": len(patent_results),
                "sample_patents": patent_results[:5],
                "novelty_score": 10 - min(10, len(patent_results)),
                "status": "Clear" if len(patent_results) < 3 else "Crowded"
            }
            
            validation_results["checks_performed"].append(patent_analysis)
            
            thinker.think(
                f"Patent search: Found {len(patent_results)} patents. Status: {patent_analysis['status']}",
                step=3,
                next_needed=True
            )
        
        # Startup Search
        if check_startups and tavily_api_key:
            thinker.think("Searching startup activity...", step=4, next_needed=True)
            
            startup_queries = [
                f"startup {paper_a_terms} {paper_b_terms}",
                f"company {paper_a_terms} {paper_b_terms} funding"
            ]
            
            startup_results = []
            
            for query in startup_queries:
                try:
                    response = await client.post(
                        "https://api.tavily.com/search",
                        json={
                            "api_key": tavily_api_key,
                            "query": query,
                            "max_results": 5,
                            "search_depth": "basic"
                        },
                        timeout=15.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        startup_results.extend([{
                            "title": r.get("title"),
                            "url": r.get("url")
                        } for r in results])
                except:
                    pass
            
            startup_analysis = {
                "check": "startups",
                "companies_found": len(startup_results),
                "sample_companies": startup_results[:5],
                "market_maturity": "Early" if len(startup_results) < 5 else "Mature",
                "competition_level": "Low" if len(startup_results) < 5 else "High"
            }
            
            validation_results["checks_performed"].append(startup_analysis)
            
            thinker.think(
                f"Startup search: Found {len(startup_results)} companies. Maturity: {startup_analysis['market_maturity']}",
                step=5,
                next_needed=True
            )
        
        # Citation Analysis
        if check_citations and tavily_api_key:
            thinker.think("Analyzing citation networks...", step=6, next_needed=True)
            
            citation_queries = [
                f"research {paper_a_terms} {paper_b_terms} citations",
                f"arxiv {paper_a_terms} {paper_b_terms}"
            ]
            
            citation_results = []
            
            for query in citation_queries:
                try:
                    response = await client.post(
                        "https://api.tavily.com/search",
                        json={
                            "api_key": tavily_api_key,
                            "query": query,
                            "max_results": 10,
                            "search_depth": "advanced",
                            "time_range": "year"
                        },
                        timeout=15.0
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])
                        citation_results.extend([{
                            "title": r.get("title"),
                            "url": r.get("url")
                        } for r in results])
                except:
                    pass
            
            citation_analysis = {
                "check": "citations",
                "recent_papers_found": len(citation_results),
                "sample_papers": citation_results[:5],
                "research_activity": "Low" if len(citation_results) < 10 else "High",
                "novelty_indicator": "High novelty" if len(citation_results) < 10 else "Well-explored"
            }
            
            validation_results["checks_performed"].append(citation_analysis)
            
            thinker.think(
                f"Citation analysis: Found {len(citation_results)} papers. {citation_analysis['novelty_indicator']}",
                step=7,
                next_needed=False
            )
    
    # Calculate overall novelty
    overall_novelty = 0
    for check in validation_results["checks_performed"]:
        if check["check"] == "patents":
            overall_novelty += check["novelty_score"]
        elif check["check"] == "citations" and check["novelty_indicator"] == "High novelty":
            overall_novelty += 5
    
    validation_results["overall_novelty_score"] = min(10, overall_novelty)
    validation_results["recommendation"] = (
        "HIGH PRIORITY - Novel opportunity" if overall_novelty >= 7 else
        "MEDIUM PRIORITY - Some existing work" if overall_novelty >= 4 else
        "LOW PRIORITY - Crowded space"
    )
    validation_results["thinking_summary"] = thinker.get_summary()
    
    session.validation_results[reverse_salient_id] = validation_results
    
    return validation_results

# ============================================================================
# PHASE 7: INNOVATION THESIS
# ============================================================================

@mcp.tool()
async def develop_innovation_thesis(
    session_id: str,
    reverse_salient_id: str
) -> Dict[str, Any]:
    """
    Develop complete innovation thesis for a reverse salient.
    
    Args:
        session_id: Discovery session ID
        reverse_salient_id: RS ID (e.g., "RS-001")
    
    Returns:
        Innovation opportunity report
    """
    
    session = discovery_sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}
    
    rs = next((r for r in session.reverse_salients if r["id"] == reverse_salient_id), None)
    if not rs:
        return {"error": "Reverse salient not found"}
    
    i = rs["paper_a_index"]
    j = rs["paper_b_index"]
    
    paper_a_full = session.cleaned_papers[i]
    paper_b_full = session.cleaned_papers[j]
    
    validation = session.validation_results.get(reverse_salient_id, {})
    
    opportunity = {
        "id": f"INN-{reverse_salient_id.split('-')[1]}",
        "reverse_salient_id": reverse_salient_id,
        "differential_score": rs["differential_score"],
        "breakthrough_potential": rs["breakthrough_potential"],
        
        "thesis": f"""
Innovation Thesis - {reverse_salient_id}

STRUCTURAL SIMILARITY (LSA): {rs['lsa_similarity']:.2%}
→ Papers share similar METHODS and TECHNIQUES

SEMANTIC SIMILARITY (BERT): {rs['bert_similarity']:.2%}
→ Papers address DIFFERENT PROBLEMS and DOMAINS

DIFFERENTIAL SCORE: {rs['differential_score']:.3f}
→ HIGH differential = STRUCTURAL TRANSFER opportunity

BREAKTHROUGH POTENTIAL: {rs['breakthrough_potential']}/10

INNOVATION TYPE: Methods from Paper A can solve problems in Paper B's domain.
        """,
        
        "validation_summary": validation if validation else "Not validated yet",
        "recommendation": validation.get("recommendation", "Validate first") if validation else "Run validation",
        "created_at": datetime.now().isoformat()
    }
    
    session.opportunities.append(opportunity)
    
    return opportunity

# ============================================================================
# REPORTING
# ============================================================================

@mcp.tool()
async def generate_report(
    session_id: str,
    format: str = "json"
) -> Dict[str, Any]:
    """
    Generate comprehensive discovery report.
    
    Args:
        session_id: Discovery session ID
        format: "json" or "markdown"
    
    Returns:
        Complete report with all findings
    """
    
    session = discovery_sessions.get(session_id)
    if not session:
        return {"error": "Session not found"}
    
    report = {
        "session_id": session_id,
        "created_at": session.created_at,
        "challenge": session.input.get("challenge", "N/A"),
        "domains": [d.get("label") for d in session.domains],
        
        "data_summary": {
            "papers_collected": len(session.papers),
            "papers_cleaned": len(session.cleaned_papers)
        },
        
        "reverse_salients": {
            "total_found": len(session.reverse_salients),
            "top_10": session.reverse_salients[:10]
        },
        
        "validation_summary": {
            "validated_count": len(session.validation_results),
            "validations": session.validation_results
        },
        
        "thinking_logs": {
            "total_thoughts": len(session.thinking_logs),
            "logs": session.thinking_logs
        },
        
        "phase": session.phase,
        "status": "complete" if len(session.reverse_salients) > 0 else "in_progress"
    }
    
    if format == "markdown":
        md = f"""# Mindrian Reverse Salient Discovery Report

## Session: {session_id}

### Challenge
{report['challenge']}

### Domains
{', '.join(report['domains'])}

### Data Summary
- Papers: {report['data_summary']['papers_collected']} collected, {report['data_summary']['papers_cleaned']} cleaned

### Reverse Salients
Total: {report['reverse_salients']['total_found']}

### Top Opportunities
"""
        for rs in report['reverse_salients']['top_10']:
            md += f"\n**{rs['id']}**: Differential={rs['differential_score']:.3f}, Potential={rs.get('breakthrough_potential', 'N/A')}/10\n"
        
        return {"report": md}
    else:
        return report

# ============================================================================
# AUTOMATED WORKFLOW
# ============================================================================

@mcp.tool()
async def execute_full_workflow(
    structured_input: Dict[str, Any],
    search_queries: List[str],
    validate_top_n: int = 3
) -> Dict[str, Any]:
    """
    Execute complete Mindrian discovery workflow automatically.
    
    Args:
        structured_input: Challenge and domain definitions
        search_queries: Tavily search queries
        validate_top_n: Number of top RSs to validate
    
    Returns:
        Complete workflow results
    """
    
    init_result = await initialize_discovery(structured_input)
    session_id = init_result["session_id"]
    
    results = {"session_id": session_id, "steps": []}
    
    tavily_result = await collect_papers_tavily(session_id, search_queries)
    results["steps"].append({"step": "collect", "result": tavily_result})
    
    clean_result = await clean_papers(session_id)
    results["steps"].append({"step": "clean", "result": clean_result})
    
    lsa_result = await compute_lsa_similarity(session_id)
    results["steps"].append({"step": "lsa", "result": lsa_result})
    
    bert_result = await compute_bert_similarity(session_id)
    results["steps"].append({"step": "bert", "result": bert_result})
    
    rs_result = await find_reverse_salients(session_id)
    results["steps"].append({"step": "reverse_salients", "result": rs_result})
    
    validations = []
    for i in range(min(validate_top_n, len(rs_result["top_opportunities"]))):
        rs_id = rs_result["top_opportunities"][i]["id"]
        validation = await validate_reverse_salient(
            session_id=session_id,
            reverse_salient_id=rs_id
        )
        validations.append(validation)
    
    results["steps"].append({"step": "validation", "result": validations})
    
    report = await generate_report(session_id)
    results["final_report"] = report
    
    return results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Mindrian Reverse Salient Discovery MCP Server")
    print("Version 1.0.0")
    print("=" * 60)
    print("\nFeatures:")
    print("  ✓ LSA Structural Similarity")
    print("  ✓ BERT Semantic Similarity")
    print("  ✓ Sequential Thinking Integration")
    print("  ✓ Patent Database Searches")
    print("  ✓ Startup Activity Monitoring")
    print("  ✓ Citation Network Analysis")
    print("\nRequired Environment Variables:")
    print("  TAVILY_API_KEY  - For web search and validation")
    print("  SCOPUS_API_KEY  - Optional, for academic APIs")
    print("\nStarting server...")
    print("=" * 60)
    
    mcp.run(transport="stdio")
