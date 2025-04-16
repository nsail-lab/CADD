from context.news import NewsRetriever
from context.wikidata import WikiDataRetriever
from context.reddit import RedditRetriever 
from text_embedding import TextEmbedder, TextSummarizer
from tqdm import tqdm
import numpy as np
import json 



class NumpyEncoder(json.JSONEncoder):
    """ Custom JSON encoder for NumPy data types. """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class Subject:
    def __init__(self, name, config):
        self.name = name
        self.config = config

    def get_news(self, news_retriever=None, text_summarizer=None, text_embedder=None, **kwargs):
        """
        Retrieve news articles related to the subject.

        Args:
            **kwargs: Additional keyword arguments to pass to the NewsRetriever.get() method.

        Returns:
            list: A list of dictionaries containing details of the retrieved news articles.
        """
        if news_retriever is None: news_retriever = NewsRetriever()
        if text_summarizer is None: text_summarizer = TextSummarizer()
        if text_embedder is None: text_embedder = TextEmbedder()
        print("[SUBJECT: {}] Retrieving news articles . . . ".format(self.name))
        news_articles = news_retriever.get(subject=self.name, **self.config['news'])
        for news_article in tqdm(news_articles):
            news_article['text_summary_pegasus'] = text_summarizer.summarize(news_article['text'])[0]
            news_article['embedding_title'] = text_embedder.embed(news_article['title'])[0].cpu().numpy().tolist()
            news_article['embedding_text'] = text_embedder.embed(news_article['text'])[0].cpu().numpy().tolist()
            try:
                news_article['embedding_summary'] = text_embedder.embed(news_article['summary'])[0].cpu().numpy().tolist()
            except:
                pass
            news_article['embedding_text_summary_pegasus'] = text_embedder.embed(news_article['text_summary_pegasus'])[0].cpu().numpy().tolist()
        
        self.news_articles = {'params': self.config['news'], 'results': news_articles}
        return self.news_articles
    
    def get_wikidata(self, wikidata_retriever=None, text_summarizer=None, text_embedder=None):
        if wikidata_retriever is None: wikidata_retriever = WikiDataRetriever()
        if text_summarizer is None: text_summarizer = TextSummarizer()
        if text_embedder is None: text_embedder = TextEmbedder()
        print("[SUBJECT: {}] Retrieving details from wikidata . . . ".format(self.name))
        wikidata_features = wikidata_retriever.get(subject=self.name)
        try:
            wikidata_features['embedding_occupations'] = text_embedder.embed(wikidata_features['occupation']).cpu().numpy().mean(axis=0).tolist()
        except:
            wikidata_features['embedding_occupations'] = None 

        if 'followers' not in wikidata_features or isinstance(wikidata_features['followers'], str):   
            followers = [0]
        else:
            followers = [int(t[1:]) for t in wikidata_features['followers']]
        
        wikidata_features['followers_mean'] = np.mean(followers)
        wikidata_features["followers_sum"] = np.sum(followers)
        wikidata_features["followers_max"] = np.max(followers)
        wikidata_features["followers_min"] = np.min(followers)

        wikidata_features["gender_binary"] = (wikidata_features['gender'] == 'male') if 'gender' in wikidata_features else None

        spouse = wikidata_features['spouse'] if 'spouse' in wikidata_features else '0'
        if isinstance(spouse, str):     
            spouse = min(1, len(spouse))
        else:                           
            spouse = len(spouse)
        wikidata_features["spouse_count"] = spouse

        children = wikidata_features['children'] if 'children' in wikidata_features else '0'
        if isinstance(children, str):     
            children = min(1, len(children))
        else:                           
            children = len(children)
        wikidata_features["children_count"] = children
        
        self.wikidata_features = {'params': {}, 'results': wikidata_features}
        return self.wikidata_features
    
    def get_reddit(self, reddit_retriever=None, text_summarizer=None, text_embedder=None, max_comments=10):
        if reddit_retriever is None: reddit_retriever = RedditRetriever()
        if text_summarizer is None: text_summarizer = TextSummarizer()
        if text_embedder is None: text_embedder = TextEmbedder()
        print("[SUBJECT: {}] Retrieving reddit posts . . . ".format(self.name))
        reddit_posts = reddit_retriever.get(subject=self.name, **self.config['reddit'])
        for post in tqdm(reddit_posts):
            post['comments'] = post['comments'][:max_comments]
            post['embedding_title'] = text_embedder.embed(post['title'])[0].cpu().numpy().tolist()
            post['embedding_text'] = text_embedder.embed(post['text'])[0].cpu().numpy().tolist()
            try:
                post['embedding_comments'] = text_embedder.embed(post['comments']).cpu().numpy().mean(axis=0).tolist()
            except: 
                pass
            post['text_summary'] = text_summarizer.summarize(post['text'])[0]
            post['embedding_text_summary'] = text_embedder.embed(post['text_summary'])[0].cpu().numpy().tolist()
        self.reddit_posts = {'params': self.config['reddit'], 'results': reddit_posts}
        return self.reddit_posts

    def to_json(self, file_path=None, process=False):
        if process is True:
            self.get_wikidata()
            self.get_news()
            self.get_reddit()
        dict_json = {'subject': self.name, 'wikidata': self.wikidata_features, 'news': self.news_articles, 'reddit': self.reddit_posts}

        if file_path != None:
            with open(file_path, 'w') as f:
                json.dump(dict_json, f, indent=4, cls=NumpyEncoder)
        
        return dict_json