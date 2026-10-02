"""Legacy text utilities used only by the first-generation RAG stack."""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass  # Ignore if already downloaded

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logger = logging.getLogger("cellforge.task_analysis")
    logger.setLevel(getattr(logging, log_level.upper()))

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load configuration file"""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {}

def save_config(config: Dict[str, Any], config_path: str = "config.json") -> None:
    """Save configuration to file"""
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_dataset_info(dataset_path: str) -> Dict[str, Any]:
    """Get basic information about datasets in the given path"""
    dataset_info = {
        "path": dataset_path,
        "datasets": [],
        "total_files": 0
    }

    if not os.path.exists(dataset_path):
        return dataset_info

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.h5ad', '.csv', '.txt', '.tsv')):
                dataset_info["datasets"].append({
                    "name": file,
                    "path": os.path.join(root, file),
                    "type": file.split('.')[-1]
                })
                dataset_info["total_files"] += 1

    return dataset_info

def validate_task_description(task_description: str) -> bool:
    """Validate task description format"""
    if not task_description or len(task_description.strip()) < 10:
        return False

    # Check for basic required elements
    required_elements = ["task", "input", "output"]
    task_lower = task_description.lower()

    for element in required_elements:
        if element not in task_lower:
            return False

    return True

def format_task_description(task_description: str) -> str:
    """Format and clean task description"""
    # Remove extra whitespace and normalize line breaks
    formatted = " ".join(task_description.split())
    return formatted.strip()

def get_environment_info() -> Dict[str, Any]:
    """Get current environment information"""
    import sys
    import platform

    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "working_directory": os.getcwd()
    }

class TextProcessor:
    """Text processing utilities for task analysis"""

    def __init__(self):
        self.logger = setup_logging()
        self.vectorizer = TfidfVectorizer(max_features=1000)

        # Technical terms for single-cell analysis
        self.technical_terms = [
            "transformer", "attention", "embedding", "encoder", "decoder",
            "neural network", "deep learning", "machine learning", "AI",
            "single-cell", "RNA-seq", "gene expression", "perturbation",
            "CRISPR", "knockout", "drug treatment", "cytokine", "scRNA-seq",
            "scATAC-seq", "CITE-seq", "multi-omics", "dimensionality reduction",
            "clustering", "differential expression", "trajectory analysis",
            "pseudotime", "cell type annotation", "batch correction",
            "normalization", "quality control", "feature selection"
        ]

        # Biological terms for single-cell analysis
        self.biological_terms = [
            "gene", "protein", "cell", "tissue", "organism", "pathway",
            "metabolism", "signaling", "transcription", "translation",
            "mutation", "variant", "allele", "genotype", "phenotype",
            "transcriptome", "epigenome", "chromatin", "nucleus",
            "mitochondria", "ribosome", "membrane", "cytoplasm",
            "nucleic acid", "amino acid", "peptide", "enzyme",
            "receptor", "ligand", "hormone", "cytokine", "chemokine"
        ]

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""

        # Remove extra whitespace
        cleaned = " ".join(text.split())
        return cleaned.strip()

    def extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Simple keyword extraction - can be enhanced with NLP libraries

        # Remove special characters and split into words
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'author'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]

        return list(set(keywords))

    def extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from input text using intelligent keyword extraction"""
        # Use TF-IDF to extract important technical terms from the input text
        try:
            # Split text into sentences and extract key phrases
            sentences = sent_tokenize(text)
            if not sentences:
                return []

            # Use TF-IDF to find important terms
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            feature_names = self.vectorizer.get_feature_names_out()

            # Get term importance scores
            term_scores = tfidf_matrix.sum(axis=0).A1
            important_terms = []

            # Select top terms based on TF-IDF scores
            top_indices = term_scores.argsort()[-20:][::-1]  # Top 20 terms

            # Enhanced stop words list
            stop_words = {
                'this', 'that', 'with', 'from', 'they', 'have', 'will', 'been', 'were', 'said',
                'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other', 'about',
                'many', 'then', 'them', 'these', 'some', 'what', 'into', 'more', 'very', 'when',
                'just', 'only', 'know', 'take', 'than', 'first', 'been', 'call', 'who', 'oil',
                'sit', 'now', 'find', 'down', 'day', 'did', 'get', 'come', 'made', 'may',
                'part', 'over', 'new', 'sound', 'take', 'only', 'little', 'work', 'know',
                'place', 'year', 'live', 'me', 'back', 'give', 'most', 'very', 'after',
                'thing', 'our', 'just', 'name', 'good', 'sentence', 'it\'s', 'using','every'
            }

            for idx in top_indices:
                if term_scores[idx] > 0:
                    term = feature_names[idx]
                    # Filter out common words, short terms, and specific stop words
                    if (len(term) > 3 and
                        term not in stop_words and
                        not term.isdigit() and
                        not term.startswith('_') and
                        not term.endswith('_') and
                        term.count('_') <= 2):  # Limit underscores
                        important_terms.append(term)

            # Return top 5 technical terms, prioritizing domain-specific terms
            domain_terms = [term for term in important_terms if any(tech in term.lower() for tech in
                          ['transformer', 'attention', 'embedding', 'encoder', 'decoder', 'neural',
                           'deep', 'machine', 'learning', 'AI', 'single_cell', 'RNA', 'seq',
                           'CRISPR', 'perturbation', 'expression', 'gene', 'cell', 'analysis'])]

            if domain_terms:
                return domain_terms[:5]
            else:
                return important_terms[:5]

        except Exception as e:
            # Fallback to simple keyword extraction
            return self.extract_keywords(text)[:5]

    def extract_biological_terms(self, text: str) -> List[str]:
        """Extract biological terms from input text using intelligent keyword extraction"""
        # Use TF-IDF to extract important biological terms from the input text
        try:
            # Split text into sentences and extract key phrases
            sentences = sent_tokenize(text)
            if not sentences:
                return []

            # Use TF-IDF to find important terms
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            feature_names = self.vectorizer.get_feature_names_out()

            # Get term importance scores
            term_scores = tfidf_matrix.sum(axis=0).A1
            important_terms = []

            # Select top terms based on TF-IDF scores
            top_indices = term_scores.argsort()[-20:][::-1]  # Top 20 terms

            # Enhanced stop words list (same as technical terms)
            stop_words = {
                'this', 'that', 'with', 'from', 'they', 'have', 'will', 'been', 'were', 'said',
                'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other', 'about',
                'many', 'then', 'them', 'these', 'some', 'what', 'into', 'more', 'very', 'when',
                'just', 'only', 'know', 'take', 'than', 'first', 'been', 'call', 'who', 'oil',
                'sit', 'now', 'find', 'down', 'day', 'did', 'get', 'come', 'made', 'may',
                'part', 'over', 'new', 'sound', 'take', 'only', 'little', 'work', 'know',
                'place', 'year', 'live', 'me', 'back', 'give', 'most', 'very', 'after',
                'thing', 'our', 'just', 'name', 'good', 'sentence', 'man', 'think', 'say',
                'great', 'where', 'help', 'through', 'much', 'before', 'line', 'right', 'too',
                'mean', 'old', 'any', 'same', 'tell', 'boy', 'follow', 'came', 'want', 'show',
                'also', 'around', 'form', 'three', 'small', 'set', 'put', 'end', 'does',
                'another', 'well', 'large', 'must', 'big', 'even', 'such', 'because', 'turn',
                'here', 'why', 'ask', 'went', 'men', 'read', 'need', 'land', 'different',
                'home', 'us', 'move', 'try', 'kind', 'hand', 'picture', 'again', 'change',
                'off', 'play', 'spell', 'air', 'away', 'house', 'point', 'page',
                'letter', 'mother', 'answer', 'found', 'study', 'still', 'learn', 'should',
                'America', 'world', 'high', 'every', 'near', 'add', 'food', 'between', 'own',
                'below', 'country', 'plant', 'last', 'school', 'father', 'keep', 'tree',
                'never', 'start', 'city', 'earth', 'eye', 'light', 'thought', 'head',
                'under', 'story', 'saw', 'left', 'don\'t', 'few', 'while', 'along', 'might',
                'close', 'something', 'seem', 'next', 'hard', 'open', 'example', 'begin',
                'life', 'always', 'those', 'both', 'paper', 'together', 'got', 'group',
                'often', 'run', 'important', 'until', 'children', 'side', 'feet', 'car',
                'mile', 'night', 'walk', 'white', 'sea', 'began', 'grow', 'took', 'river',
                'four', 'carry', 'state', 'once', 'book', 'hear', 'stop', 'without',
                'second', 'late', 'miss', 'idea', 'enough', 'eat', 'face', 'watch', 'far',
                'Indian', 'real', 'almost', 'let', 'above', 'girl', 'sometimes', 'mountain',
                'cut', 'young', 'talk', 'soon', 'list', 'song', 'being', 'leave', 'family',
                'it\'s', 'using'
            }

            for idx in top_indices:
                if term_scores[idx] > 0:
                    term = feature_names[idx]
                    # Filter out common words, short terms, and specific stop words
                    if (len(term) > 3 and
                        term not in stop_words and
                        not term.isdigit() and
                        not term.startswith('_') and
                        not term.endswith('_') and
                        term.count('_') <= 2):  # Limit underscores
                        important_terms.append(term)

            # Return top 5 biological terms, prioritizing domain-specific terms
            domain_terms = [term for term in important_terms if any(bio in term.lower() for bio in
                          ['gene', 'protein', 'cell', 'tissue', 'organism', 'pathway',
                           'metabolism', 'signaling', 'transcription', 'translation',
                           'mutation', 'variant', 'allele', 'genotype', 'phenotype',
                           'transcriptome', 'epigenome', 'chromatin', 'nucleus',
                           'mitochondria', 'ribosome', 'membrane', 'cytoplasm',
                           'nucleic', 'amino', 'peptide', 'enzyme', 'receptor',
                           'ligand', 'hormone', 'cytokine', 'chemokine'])]

            if domain_terms:
                return domain_terms[:5]
            else:
                return important_terms[:5]

        except Exception as e:
            # Fallback to simple keyword extraction
            return self.extract_keywords(text)[:5]

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using TF-IDF"""
        try:
            tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
            return (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]
        except:
            return 0.0

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Alias for calculate_similarity for compatibility"""
        return self.calculate_similarity(text1, text2)

    def extract_summary(self, text: str, max_length: int = 200) -> str:
        """Extract summary from text using TF-IDF sentence scoring"""
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return ""

            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            sentence_scores = tfidf_matrix.sum(axis=1).A1

            top_sentence_indices = sentence_scores.argsort()[-3:][::-1]
            top_sentence_indices.sort()

            summary = " ".join([sentences[i] for i in top_sentence_indices])
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            return summary
        except:
            return text[:max_length] + "..." if len(text) > max_length else text

    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[str]:
        """Extract key phrases from text"""
        try:
            sentences = sent_tokenize(text)
            if not sentences:
                return []

            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            feature_names = self.vectorizer.get_feature_names_out()

            phrase_scores = []
            for i, sentence in enumerate(sentences):
                sentence_vector = tfidf_matrix[i].toarray()[0]
                top_indices = sentence_vector.argsort()[-top_n:][::-1]
                for idx in top_indices:
                    if sentence_vector[idx] > 0:
                        phrase_scores.append((feature_names[idx], sentence_vector[idx]))

            phrase_counter = Counter([phrase for phrase, _ in phrase_scores])
            return [phrase for phrase, _ in phrase_counter.most_common(top_n)]
        except:
            return self.extract_keywords(text)[:top_n]

    def validate_task_format(self, task_description: str) -> bool:
        """Validate task description format"""
        return validate_task_description(task_description)
