from newspaper import Article
from transformers import pipeline
from keybert import KeyBERT
import json
import storey
import v3io.aio.dataplane

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

def summarize_article(event):
    """
    Summarizes news article text via Huggingface pipeline using
    DistilBart model.
    
    Expects "original_text" field in event.
    """
    summarizer = pipeline("summarization")
    event["summarized_text"] = summarizer(event["original_text"], truncation=True)[0]['summary_text']
    return event

def extract_keywords(event):
    """
    Extracts single keywords from news article text via KeyBERT using
    BERT-embeddings. Uses Maximal Marginal Relevance to create keywords
    based on cosine similarity (reduces redundancy).
    
    Expects "original_text" field in event.
    """
    keywords = KeyBERT().extract_keywords(
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

class WriteToKV(storey.MapClass):
    """
    Write record to V3IO KV table. Flattens lists as JSON strings.
    
    Expects "title" field in event.
    """
    def __init__(self, container, table_path, key, **kwargs):
        super().__init__(**kwargs)
        self.container = container
        self.table_path = table_path
        self.key = key
       
    def flatten_dict(self, event):
        for k, v in event.items():
            if type(v) == list:
                event[k] = json.dumps(v)
        return dict(event)
    
    async def do(self, event):
        v3io_client = v3io.aio.dataplane.Client()
        await v3io_client.kv.put(
            container=self.container,
            table_path=self.table_path,
            key=event[self.key],
            attributes=self.flatten_dict(event.copy())
        )
        return event
        
    def to_dict(self):
        return {
            "class_name": "WriteToKV",
            "name": self.name or "WriteToKV",
            "class_args": {
                "container" : self.container,
                "table_path" : self.table_path,
                "key" : self.key
            }
        }