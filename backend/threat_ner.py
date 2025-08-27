"""
Simple NER for threat entities without transformers dependency
Fallback version that works with basic dependencies
"""
import re
from typing import Dict, List

def extract_threat_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract cybersecurity entities using regex patterns
    Fallback version when transformers is not available
    """
    try:
        # Try to import transformers version
        from transformers import pipeline
        ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
        
        entities = ner_pipeline(text)
        grouped = {}

        for ent in entities:
            label = ent["entity_group"]
            word = ent["word"]

            if label not in grouped:
                grouped[label] = []
            if word not in grouped[label]:
                grouped[label].append(word)

        return grouped
        
    except Exception:
        # Fallback to regex-based extraction
        return extract_entities_with_regex(text)

def extract_entities_with_regex(text: str) -> Dict[str, List[str]]:
    """
    Extract cybersecurity entities using regex patterns
    """
    entities = {}
    
    # Define patterns for cybersecurity entities
    patterns = {
        'IP_ADDRESS': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
        'CVE': r'CVE-\d{4}-\d{4,7}',
        'MD5': r'\b[a-fA-F0-9]{32}\b',
        'SHA1': r'\b[a-fA-F0-9]{40}\b',
        'SHA256': r'\b[a-fA-F0-9]{64}\b',
        'URL': r'https?://[^\s<>"{}|\\^`\[\]]*',
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'DOMAIN': r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,}\b',
        'PORT': r'\b(?:port\s+)?(\d{1,5})\b'
    }
    
    # Extract entities using patterns
    for entity_type, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # For PORT, filter valid port numbers
            if entity_type == 'PORT':
                matches = [p for p in matches if 1 <= int(p) <= 65535]
            
            if matches:
                entities[entity_type] = list(set(matches))
    
    # Extract common threat keywords
    threat_keywords = [
        'malware', 'virus', 'trojan', 'ransomware', 'backdoor',
        'phishing', 'spam', 'botnet', 'ddos', 'injection',
        'vulnerability', 'exploit', 'payload', 'shellcode',
        'attack', 'breach', 'intrusion', 'compromise'
    ]
    
    found_keywords = []
    text_lower = text.lower()
    for keyword in threat_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    
    if found_keywords:
        entities['THREAT_KEYWORDS'] = found_keywords
    
    return entities
