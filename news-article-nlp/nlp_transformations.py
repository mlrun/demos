from newspaper import Article
from transformers import pipeline
from keybert import KeyBERT
import json
import storey

def fetch_article(event):
    """
    Fetch news article from URL via newspaper3k. Retrieves article text,
    author, and publishing date.
    
    Expects "url" field in event.
    """
    article = Article(event["url"])
    article.download()
    article.parse()
    
    event["title"] = article.title
    event["authors"] = article.authors
    event["publish_date"] = str(article.publish_date)
    event["original_text"] = article.text
    return event

class SummarizeArticle:
    """
    Summarizes news article text via Huggingface pipeline using
    DistilBart model.
    
    Expects "original_text" field in event.
    """
    def __init__(self):
        self.summarizer = pipeline("summarization")
        
    def do(self, event):
        event["summarized_text"] = self.summarizer(event["original_text"], truncation=True)[0]['summary_text']
        return event

class ExtractKeywords:
    """
    Extracts single keywords from news article text via KeyBERT using
    BERT-embeddings. Uses Maximal Marginal Relevance to create keywords
    based on cosine similarity (reduces redundancy).
    
    Expects "original_text" field in event.
    """
    def __init__(self):
        self.keybert = KeyBERT()
        
    def do(self, event):
        keywords = self.keybert.extract_keywords(
            event["original_text"],
            keyphrase_ngram_range=(1, 1),
            stop_words='english',
            use_mmr=True,
            diversity=0.7
        )
        event["keywords"] = [k[0] for k in keywords]
        return event

def filter_article(event):
    """
    Optionally filters original article text from response.
    
    Expects "original_text" and optional "filter_article"
    fields in event.
    """
    if event.get("filter_article", True):
        del event["original_text"]
    return event

def kv_format(event):
    """
    Format record to be stored in V3IO KV table. Removes apostrophes in text fields,
    flattens lists into strings, and adds the title as the key for the KV record.
    
    Expects "title", "summarized_text", and optional "original_text" fields
    in event.
    """
    # Remove commas from title and text
    event.body["title"] = event.body["title"].replace("'", "")
    event.body["summarized_text"] = event.body["summarized_text"].replace("'", "")
    if "original_text" in event.body:
        event.body["original_text"] = event.body["original_text"].replace("'", "")
    
    # Add KV key
    event.key = event.body["title"]
    
    # Flatten list into string
    for k, v in event.body.items():
        if type(v) == list:
            event.body[k] = json.dumps(v)
    
    return event