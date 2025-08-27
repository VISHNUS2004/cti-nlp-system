"""
Enhanced NER Model specifically for Cybersecurity Threat Intelligence
Uses domain-specific models and custom entity types
"""
import os
import json
import torch
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    pipeline, Trainer, TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import numpy as np

class CybersecurityNER:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.custom_entities = [
            'MALWARE', 'VULNERABILITY', 'IP_ADDRESS', 'DOMAIN', 
            'FILE_HASH', 'CVE', 'ATTACK_TECHNIQUE', 'THREAT_ACTOR',
            'TOOL', 'PLATFORM', 'PROTOCOL', 'PORT'
        ]
        
    def setup_models(self):
        """Setup multiple NER models for ensemble"""
        model_configs = {
            'general': {
                'model': 'dslim/bert-base-NER',
                'description': 'General purpose NER'
            },
            'cybersecurity': {
                'model': 'microsoft/DialoGPT-medium',  # Can be fine-tuned
                'description': 'Cybersecurity focused (to be fine-tuned)'
            },
            'scientific': {
                'model': 'allenai/scibert_scivocab_uncased',
                'description': 'Scientific/technical text understanding'
            }
        }
        
        for name, config in model_configs.items():
            try:
                print(f"Loading {name} model: {config['model']}")
                
                if name == 'general':
                    # Use pre-trained NER pipeline
                    self.models[name] = pipeline(
                        "ner", 
                        model=config['model'], 
                        grouped_entities=True,
                        device=0 if torch.cuda.is_available() else -1
                    )
                else:
                    # For custom models, we'll set up tokenizers for fine-tuning
                    self.tokenizers[name] = AutoTokenizer.from_pretrained(config['model'])
                    
            except Exception as e:
                print(f"Failed to load {name} model: {e}")
    
    def extract_cyber_entities(self, text):
        """Extract cybersecurity-specific entities"""
        import re
        
        entities = {
            'IP_ADDRESS': [],
            'DOMAIN': [],
            'FILE_HASH': [],
            'CVE': [],
            'EMAIL': [],
            'URL': [],
            'PORT': []
        }
        
        # IP Address pattern (IPv4)
        ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
        entities['IP_ADDRESS'] = re.findall(ip_pattern, text)
        
        # Domain pattern
        domain_pattern = r'\b[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}\b'
        entities['DOMAIN'] = re.findall(domain_pattern, text)
        entities['DOMAIN'] = [match[0] + match[1] + match[2] for match in entities['DOMAIN']]
        
        # Hash patterns (MD5, SHA1, SHA256)
        hash_patterns = {
            'MD5': r'\b[a-fA-F0-9]{32}\b',
            'SHA1': r'\b[a-fA-F0-9]{40}\b',
            'SHA256': r'\b[a-fA-F0-9]{64}\b'
        }
        
        for hash_type, pattern in hash_patterns.items():
            hashes = re.findall(pattern, text)
            entities['FILE_HASH'].extend(hashes)
        
        # CVE pattern
        cve_pattern = r'CVE-\d{4}-\d{4,7}'
        entities['CVE'] = re.findall(cve_pattern, text, re.IGNORECASE)
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['EMAIL'] = re.findall(email_pattern, text)
        
        # URL pattern
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]*'
        entities['URL'] = re.findall(url_pattern, text)
        
        # Port pattern
        port_pattern = r'\b(?:port\s+)?(\d{1,5})\b'
        potential_ports = re.findall(port_pattern, text, re.IGNORECASE)
        entities['PORT'] = [p for p in potential_ports if 1 <= int(p) <= 65535]
        
        # Remove empty lists
        entities = {k: list(set(v)) for k, v in entities.items() if v}
        
        return entities
    
    def extract_threat_indicators(self, text):
        """Extract threat indicators using multiple approaches"""
        results = {
            'general_entities': {},
            'cyber_entities': {},
            'threat_keywords': []
        }
        
        # General NER
        if 'general' in self.models:
            try:
                general_entities = self.models['general'](text)
                grouped = {}
                for ent in general_entities:
                    label = ent['entity_group']
                    word = ent['word']
                    if label not in grouped:
                        grouped[label] = []
                    if word not in grouped[label]:
                        grouped[label].append(word)
                results['general_entities'] = grouped
            except Exception as e:
                print(f"General NER failed: {e}")
        
        # Cybersecurity-specific entities
        results['cyber_entities'] = self.extract_cyber_entities(text)
        
        # Threat keywords detection
        threat_keywords = [
            'malware', 'virus', 'trojan', 'ransomware', 'backdoor',
            'phishing', 'spam', 'botnet', 'ddos', 'injection',
            'vulnerability', 'exploit', 'payload', 'shellcode',
            'attack', 'breach', 'intrusion', 'compromise',
            'apt', 'advanced persistent threat', 'zero-day',
            'lateral movement', 'privilege escalation', 'exfiltration'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        for keyword in threat_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        results['threat_keywords'] = found_keywords
        
        return results
    
    def create_training_data_for_custom_ner(self, texts_and_labels):
        """Create training data for custom cybersecurity NER"""
        training_data = []
        
        for text, labels in texts_and_labels:
            tokens = text.split()  # Simple tokenization
            token_labels = ['O'] * len(tokens)  # Default to 'O' (Outside)
            
            # Assign labels based on the provided annotations
            for entity_text, entity_type in labels:
                # Find the entity in the text and assign BIO labels
                entity_tokens = entity_text.split()
                for i in range(len(tokens) - len(entity_tokens) + 1):
                    if tokens[i:i+len(entity_tokens)] == entity_tokens:
                        token_labels[i] = f'B-{entity_type}'
                        for j in range(1, len(entity_tokens)):
                            token_labels[i+j] = f'I-{entity_type}'
                        break
            
            training_data.append({
                'tokens': tokens,
                'labels': token_labels
            })
        
        return training_data
    
    def fine_tune_for_cybersecurity(self, training_data, output_dir):
        """Fine-tune a model for cybersecurity NER"""
        # This would implement the fine-tuning logic
        # Similar to your train_ner_transformer.py but with cyber-specific labels
        print(f"Fine-tuning model for cybersecurity NER...")
        print(f"Training samples: {len(training_data)}")
        print(f"Output directory: {output_dir}")
        
        # Implementation would go here
        # For now, we'll use the existing infrastructure
        
    def get_comprehensive_analysis(self, text):
        """Get comprehensive threat intelligence analysis"""
        analysis = {
            'text': text,
            'entities': self.extract_threat_indicators(text),
            'risk_score': 0,
            'threat_type': 'unknown',
            'recommendations': []
        }
        
        # Calculate risk score based on findings
        score = 0
        entities = analysis['entities']
        
        # Scoring based on entity types found
        scoring_weights = {
            'FILE_HASH': 30,
            'CVE': 25,
            'IP_ADDRESS': 15,
            'DOMAIN': 10,
            'URL': 10,
            'EMAIL': 5,
            'PORT': 5
        }
        
        for entity_type, entity_list in entities.get('cyber_entities', {}).items():
            score += len(entity_list) * scoring_weights.get(entity_type, 0)
        
        # Additional scoring for threat keywords
        score += len(entities.get('threat_keywords', [])) * 5
        
        analysis['risk_score'] = min(100, score)  # Cap at 100
        
        # Determine threat type based on keywords and entities
        keywords = entities.get('threat_keywords', [])
        if any(word in keywords for word in ['malware', 'virus', 'trojan', 'ransomware']):
            analysis['threat_type'] = 'malware'
        elif any(word in keywords for word in ['phishing', 'spam']):
            analysis['threat_type'] = 'social_engineering'
        elif any(word in keywords for word in ['injection', 'vulnerability', 'exploit']):
            analysis['threat_type'] = 'vulnerability_exploitation'
        elif any(word in keywords for word in ['ddos', 'botnet']):
            analysis['threat_type'] = 'network_attack'
        
        # Generate recommendations
        if entities.get('cyber_entities', {}).get('IP_ADDRESS'):
            analysis['recommendations'].append("Block identified IP addresses")
        if entities.get('cyber_entities', {}).get('DOMAIN'):
            analysis['recommendations'].append("Add domains to DNS blacklist")
        if entities.get('cyber_entities', {}).get('FILE_HASH'):
            analysis['recommendations'].append("Update antivirus signatures")
        if entities.get('cyber_entities', {}).get('CVE'):
            analysis['recommendations'].append("Apply security patches for identified CVEs")
        
        return analysis

# Usage example
def main():
    ner = CybersecurityNER()
    ner.setup_models()
    
    # Test text
    test_text = """
    The malware sample with MD5 hash d41d8cd98f00b204e9800998ecf8427e was found 
    communicating with command and control server at IP 192.168.1.100 on port 443. 
    The attack exploits CVE-2021-34527 vulnerability and sends data to malicious-domain.com.
    """
    
    # Get comprehensive analysis
    analysis = ner.get_comprehensive_analysis(test_text)
    
    print("="*60)
    print("CYBERSECURITY NER ANALYSIS")
    print("="*60)
    print(f"Risk Score: {analysis['risk_score']}/100")
    print(f"Threat Type: {analysis['threat_type']}")
    print(f"\nEntities Found:")
    for category, entities in analysis['entities'].items():
        if entities:
            print(f"  {category}: {entities}")
    print(f"\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")

if __name__ == "__main__":
    main()
