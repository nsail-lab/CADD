import os
from typing import List, Dict, Optional
import worldnewsapi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class NewsRetriever:
    """Class to retrieve and process news data using the World News API."""

    def __init__(self):
        """Initialize the NewsRetriever with API configuration."""
        api_key = os.getenv('WORLDNEWSAPI_KEY')
        if not api_key:
            raise ValueError("WORLDNEWSAPI_KEY not found in environment variables")

        self.client = worldnewsapi.Configuration(host="https://api.worldnewsapi.com")
        self.client.api_key['apiKey'] = api_key
        self.client.api_key['headerApiKey'] = api_key

    def get(self, 
            subject: str, 
            text: Optional[str] = None,
            source_countries: Optional[str] = None,
            language: str = 'en',
            min_sentiment: float = -1,
            max_sentiment: float = 1,
            earliest_publish_date: Optional[str] = None,
            latest_publish_date: Optional[str] = None,
            news_sources: Optional[str] = None,
            authors: Optional[str] = None,
            location_filter: Optional[str] = None,
            offset: Optional[int] = None,
            number: int = 10,
            sort: str = 'publish-time',
            sort_direction: str = 'desc') -> List[Dict]:
        """
        Retrieve news articles based on various filters and criteria.

        Args:
            subject (str): The subject to filter news by entities.
            text (str, optional): The text to match in the news content.
            source_countries (str, optional): Comma-separated list of ISO 3166 country codes.
            language (str): The ISO 6391 language code of the news. Defaults to 'en'.
            min_sentiment (float): The minimal sentiment of the news in range [-1, 1]. Defaults to -1.
            max_sentiment (float): The maximal sentiment of the news in range [-1, 1]. Defaults to 1.
            earliest_publish_date (str, optional): The news must have been published after this date.
            latest_publish_date (str, optional): The news must have been published before this date.
            news_sources (str, optional): Comma-separated list of news sources.
            authors (str, optional): Comma-separated list of authors.
            location_filter (str, optional): Filter news by radius around a location. Format: "latitude,longitude,radius in kilometers".
            offset (int, optional): The number of news to skip in range [0, 1000].
            number (int): The number of news to return in range [1, 100]. Defaults to 10.
            sort (str): The sorting criteria. Defaults to 'publish-time'.
            sort_direction (str): Whether to sort ascending or descending. Defaults to 'desc'.

        Returns:
            List[Dict]: A list of dictionaries containing details of the retrieved news articles.
        """
        results = []
        with worldnewsapi.ApiClient(self.client) as api_client:
            api_instance = worldnewsapi.NewsApi(api_client)

            try:
                api_response = api_instance.search_news(
                    text=subject if text is None else text,
                    source_countries=source_countries,
                    language=language,
                    news_sources=news_sources,
                    authors=authors,
                    min_sentiment=min_sentiment,
                    max_sentiment=max_sentiment,
                    location_filter=location_filter,
                    earliest_publish_date=earliest_publish_date,
                    latest_publish_date=latest_publish_date,
                    entities=None,  # Entities parameter is not used in this implementation
                    number=number,
                    sort=sort,
                    sort_direction=sort_direction
                )
                results = [news_article.to_dict() for news_article in api_response.news]
            except Exception as e:
                print(f"Exception when calling NewsApi->search_news: {e}")

        return results

# Example usage
if __name__ == "__main__":
    retriever = NewsRetriever()
    articles = retriever.get("deepfake", text="deepfake", source_countries="us", number=5)
    for article in articles:
        print(article)