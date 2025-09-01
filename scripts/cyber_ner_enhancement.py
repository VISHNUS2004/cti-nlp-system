#!/usr/bin/env python3
"""
Cybersecurity-Specific NER Enhancement
======================================

This script fine-tunes BERT for cybersecurity Named Entity Recognition
using domain-specific threat intelligence data.

Features:
- Fine-tunes BERT on cybersecurity entities
- Custom entity types: MALWARE, THREAT_ACTOR, VULNERABILITY, IOC, etc.
- Evaluation with precision, recall, F1
- Integration with existing backend

Author: CTI-NLP Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path

# Transformers for NER
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    Trainer, TrainingArguments, DataCollatorForTokenClassification
)
from datasets import Dataset as HFDataset
import evaluate

# For evaluation
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

class CyberNERDataset:
    """Prepare cybersecurity NER dataset"""
    
    def __init__(self):
        # Define cybersecurity-specific entity types
        self.entity_types = {
            'O': 0,           # Outside
            'B-MALWARE': 1,   # Beginning of malware name
            'I-MALWARE': 2,   # Inside malware name
            'B-THREAT_ACTOR': 3,  # Beginning of threat actor
            'I-THREAT_ACTOR': 4,  # Inside threat actor
            'B-VULNERABILITY': 5, # Beginning of vulnerability
            'I-VULNERABILITY': 6, # Inside vulnerability
            'B-IOC': 7,       # Beginning of IOC (IP, domain, hash)
            'I-IOC': 8,       # Inside IOC
            'B-ATTACK_TYPE': 9,   # Beginning of attack type
            'I-ATTACK_TYPE': 10,  # Inside attack type
            'B-TOOL': 11,     # Beginning of tool/technique
            'I-TOOL': 12,     # Inside tool/technique
            'B-ORG': 13,      # Beginning of organization
            'I-ORG': 14,      # Inside organization
            'B-LOC': 15,      # Beginning of location
            'I-LOC': 16,      # Inside location
        }
        
        self.id2label = {v: k for k, v in self.entity_types.items()}
        self.label2id = self.entity_types
        
        # Cybersecurity keywords for automatic labeling
        self.cyber_keywords = {
            'MALWARE': [
                'emotet', 'trickbot', 'dridex', 'qakbot', 'zeus', 'banking trojan',
                'ransomware', 'wannacry', 'petya', 'notpetya', 'ryuk', 'sodinokibi',
                'maze', 'conti', 'lockbit', 'darkside', 'revil', 'ragnar locker',
                'backdoor', 'rat', 'remote access trojan', 'keylogger', 'spyware',
                'adware', 'rootkit', 'botnet', 'worm', 'virus', 'trojan'
            ],
            'THREAT_ACTOR': [
                'apt1', 'apt28', 'apt29', 'apt40', 'fancy bear', 'cozy bear',
                'lazarus', 'carbanak', 'fin7', 'fin8', 'wizard spider',
                'sandworm', 'turla', 'equation group', 'ocean lotus',
                'kimsuky', 'darkhydrus', 'oilrig', 'charming kitten'
            ],
            'VULNERABILITY': [
                'cve-', 'vulnerability', 'zero-day', 'buffer overflow',
                'sql injection', 'xss', 'csrf', 'rce', 'privilege escalation',
                'memory corruption', 'use-after-free', 'heap overflow'
            ],
            'ATTACK_TYPE': [
                'phishing', 'spear phishing', 'watering hole', 'drive-by download',
                'brute force', 'credential stuffing', 'password spraying',
                'ddos', 'dos', 'mitm', 'man-in-the-middle', 'session hijacking',
                'social engineering', 'pretexting', 'baiting', 'quid pro quo'
            ],
            'TOOL': [
                'metasploit', 'cobalt strike', 'empire', 'powershell empire',
                'mimikatz', 'bloodhound', 'nmap', 'wireshark', 'burp suite',
                'sqlmap', 'john the ripper', 'hashcat', 'hydra'
            ]
        }
    
    def create_synthetic_dataset(self, threat_texts, num_samples=1000):
        """Create synthetic NER dataset from threat descriptions"""
        
        import re
        
        ner_data = []
        
        for i, text in enumerate(threat_texts[:num_samples]):
            tokens = text.split()
            labels = ['O'] * len(tokens)
            
            # Automatic labeling based on keywords
            text_lower = text.lower()
            
            for entity_type, keywords in self.cyber_keywords.items():
                for keyword in keywords:
                    # Find keyword matches
                    pattern = r'\b' + re.escape(keyword) + r'\b'
                    matches = list(re.finditer(pattern, text_lower))
                    
                    for match in matches:
                        start_pos = match.start()
                        end_pos = match.end()
                        
                        # Find corresponding token positions
                        current_pos = 0
                        for token_idx, token in enumerate(tokens):
                            token_start = current_pos
                            token_end = current_pos + len(token)
                            
                            if token_start >= start_pos and token_end <= end_pos:
                                if token_idx == 0 or labels[token_idx-1] != f'I-{entity_type}':
                                    labels[token_idx] = f'B-{entity_type}'
                                else:
                                    labels[token_idx] = f'I-{entity_type}'
                            
                            current_pos = token_end + 1  # +1 for space
            
            ner_data.append({
                'tokens': tokens,
                'labels': labels
            })
        
        return ner_data
    
    def prepare_dataset_for_training(self, ner_data):
        """Prepare dataset for Hugging Face training"""
        
        # Convert to the format expected by transformers
        formatted_data = []
        
        for item in ner_data:
            tokens = item['tokens']
            labels = item['labels']
            
            # Convert labels to IDs
            label_ids = [self.label2id.get(label, 0) for label in labels]
            
            formatted_data.append({
                'tokens': tokens,
                'ner_tags': label_ids
            })
        
        return formatted_data

class CyberNERTrainer:
    """Train cybersecurity-specific NER model"""
    
    def __init__(self, model_name="distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.dataset_creator = CyberNERDataset()
        
    def tokenize_and_align_labels(self, examples):
        """Tokenize text and align labels with tokens"""
        
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            padding=True,
            max_length=512
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        
        metric = evaluate.load("seqeval")
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [self.dataset_creator.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        true_labels = [
            [self.dataset_creator.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = metric.compute(predictions=true_predictions, references=true_labels)
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    def train(self, train_texts, output_dir="models/cyber_ner"):
        """Train the cybersecurity NER model"""
        
        print("🔄 Creating synthetic NER dataset...")
        ner_data = self.dataset_creator.create_synthetic_dataset(train_texts)
        formatted_data = self.dataset_creator.prepare_dataset_for_training(ner_data)
        
        # Split into train/validation
        from sklearn.model_selection import train_test_split
        train_data, val_data = train_test_split(formatted_data, test_size=0.2, random_state=42)
        
        # Create HuggingFace datasets
        train_dataset = HFDataset.from_list(train_data)
        val_dataset = HFDataset.from_list(val_data)
        
        # Tokenize
        train_tokenized = train_dataset.map(self.tokenize_and_align_labels, batched=True)
        val_tokenized = val_dataset.map(self.tokenize_and_align_labels, batched=True)
        
        # Initialize model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.dataset_creator.entity_types),
            id2label=self.dataset_creator.id2label,
            label2id=self.dataset_creator.label2id
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        print("🔄 Training cybersecurity NER model...")
        trainer.train()
        
        # Save model
        print(f"✅ Saving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer
    
    def predict(self, texts, model_path="models/cyber_ner"):
        """Predict entities in new texts"""
        
        # Load model if not already loaded
        if self.model is None:
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        results = []
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            predictions = torch.argmax(outputs.logits, dim=2)
            
            # Convert to labels
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            predicted_labels = [self.dataset_creator.id2label[pred.item()] for pred in predictions[0]]
            
            # Extract entities
            entities = self.extract_entities(tokens, predicted_labels)
            results.append({
                'text': text,
                'entities': entities
            })
        
        return results
    
    def extract_entities(self, tokens, labels):
        """Extract entities from tokens and labels"""
        
        entities = []
        current_entity = None
        current_tokens = []
        
        for token, label in zip(tokens, labels):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            if label.startswith('B-'):
                # Save previous entity
                if current_entity:
                    entities.append({
                        'entity': current_entity,
                        'text': self.tokenizer.convert_tokens_to_string(current_tokens),
                        'tokens': current_tokens
                    })
                
                # Start new entity
                current_entity = label[2:]  # Remove 'B-'
                current_tokens = [token]
                
            elif label.startswith('I-') and current_entity == label[2:]:
                # Continue current entity
                current_tokens.append(token)
                
            else:
                # End current entity
                if current_entity:
                    entities.append({
                        'entity': current_entity,
                        'text': self.tokenizer.convert_tokens_to_string(current_tokens),
                        'tokens': current_tokens
                    })
                current_entity = None
                current_tokens = []
        
        # Save final entity
        if current_entity:
            entities.append({
                'entity': current_entity,
                'text': self.tokenizer.convert_tokens_to_string(current_tokens),
                'tokens': current_tokens
            })
        
        return entities

def main():
    """Main training pipeline for cybersecurity NER"""
    
    print("🚀 Cybersecurity NER Enhancement")
    print("=" * 50)
    
    # Load threat intelligence data
    print("📂 Loading threat intelligence data...")
    df = pd.read_csv("data/Cybersecurity_Dataset.csv")
    df = df.rename(columns=lambda x: x.strip())
    
    text_col = "Cleaned Threat Description"
    df = df.dropna(subset=[text_col])
    
    threat_texts = df[text_col].tolist()
    print(f"📊 Loaded {len(threat_texts)} threat descriptions")
    
    # Initialize trainer
    trainer = CyberNERTrainer(model_name="distilbert-base-uncased")
    
    # Train model
    model_trainer = trainer.train(threat_texts)
    
    # Test on sample texts
    print("\n🔍 Testing on sample texts...")
    sample_texts = [
        "APT28 used Emotet malware to exploit CVE-2023-1234 in Microsoft Exchange servers",
        "Lazarus group deployed ransomware targeting financial institutions using phishing emails",
        "TrickBot botnet spreading through malicious email attachments with PowerShell scripts"
    ]
    
    results = trainer.predict(sample_texts)
    
    for i, result in enumerate(results):
        print(f"\n📄 Text {i+1}: {result['text']}")
        print("🏷️  Entities found:")
        for entity in result['entities']:
            print(f"   - {entity['entity']}: {entity['text']}")
    
    print("\n✅ Cybersecurity NER model training complete!")
    print("✅ Model saved to models/cyber_ner/")

if __name__ == "__main__":
    main()
