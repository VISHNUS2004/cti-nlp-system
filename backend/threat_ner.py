import re

def extract_threat_entities(text):
    """
    Simple regex-based entity extraction for demonstration.
    This is a fallback while transformer issues are resolved.
    """
    entities = {}
    
    # IP Address pattern
    ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
    ips = re.findall(ip_pattern, text)
    if ips:
        entities['IP'] = ips
    
    # Email pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        entities['EMAIL'] = emails
    
    # URL pattern
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    if urls:
        entities['URL'] = urls
    
    # Hash pattern (MD5, SHA1, SHA256)
    hash_pattern = r'\b[a-fA-F0-9]{32,64}\b'
    hashes = re.findall(hash_pattern, text)
    if hashes:
        entities['HASH'] = hashes
    
    # CVE pattern
    cve_pattern = r'CVE-\d{4}-\d{4,7}'
    cves = re.findall(cve_pattern, text)
    if cves:
        entities['CVE'] = cves
    
    # Common malware/threat terms
    threat_terms = ['malware', 'phishing', 'ransomware', 'trojan', 'virus', 'botnet', 'apt']
    found_threats = [term for term in threat_terms if term.lower() in text.lower()]
    if found_threats:
        entities['THREAT_TYPE'] = found_threats
    
    return entities
