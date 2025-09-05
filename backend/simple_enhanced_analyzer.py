import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

# You may need to download these resources the first time you run
# nltk.download('punkt')
# nltk.download('stopwords')

def simple_enhanced_analyzer(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return stemmed_tokens

# Example usage:
if __name__ == "__main__":
    sample_text = "This is a simple, enhanced analyzer for processing text! It removes stopwords and stems words."
    print(simple_enhanced_analyzer(sample_text))
