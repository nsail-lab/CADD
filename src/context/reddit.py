import os
from typing import List, Dict, Optional
import praw
import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RedditRetriever:
    """Class to retrieve and process data from Reddit using the PRAW (Python Reddit API Wrapper) library."""

    def __init__(self):
        """Initialize the RedditRetriever with Reddit API credentials from environment variables."""
        client_id = os.getenv('REDDIT_CLIENT_ID')
        client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        user_agent = os.getenv('REDDIT_USER_AGENT')

        if not all([client_id, client_secret, user_agent]):
            raise ValueError("Missing Reddit API credentials in environment variables")

        self.client = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

    def get(self, 
            subject: str, 
            subreddits: List[str] = ['all'], 
            sort: str = 'relevance', 
            limit: int = 10, 
            created_on_or_before: Optional[str] = None) -> List[Dict]:
        """
        Retrieve Reddit posts based on a subject from specified subreddits.

        Args:
            subject (str): The topic to search for on Reddit.
            subreddits (List[str]): List of subreddits to search within. Defaults to ['all'].
            sort (str): The sorting method for search results. Defaults to 'relevance'.
            limit (int): The maximum number of posts to retrieve. Defaults to 10.
            created_on_or_before (Optional[str]): Filter posts created on or before this date (format: 'yy-mm-dd').

        Returns:
            List[Dict]: A list of dictionaries containing details of the retrieved Reddit posts.
        """
        subreddit = self.client.subreddit('+'.join(subreddits))

        if created_on_or_before:
            created_timestamp = datetime.datetime.strptime(created_on_or_before, '%y-%m-%d').timestamp()
            posts = subreddit.search(subject, sort=sort, limit=None)
            filtered_posts = filter(lambda post: post.created_utx <= created_timestamp, posts)
            sorted_posts = sorted(filtered_posts, key=lambda post: post.created_utc, reverse=True)
            resp = sorted_posts[:limit]
        else:
            resp = subreddit.search(subject, sort=sort, limit=limit)

        results = []
        for submission in resp:
            comments = [comment.body for comment in submission.comments.list() if hasattr(comment, 'body')]

            results.append({
                'id': submission.id,
                'title': self._clean_text(submission.title),
                'score': submission.score,
                'url': self._clean_text(submission.url),
                'text': self._clean_text(submission.selftext),
                'timestamp': datetime.datetime.fromtimestamp(submission.created).strftime("%m/%d/%Y, %H:%M:%S"),
                'comments': comments
            }) 
        return results

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text by removing non-ASCII characters."""
        return text.encode('ascii', 'ignore').decode('ascii')

# Example usage
if __name__ == "__main__":
    retriever = RedditRetriever()
    posts = retriever.get("machine learning", subreddits=['learnmachinelearning', 'MachineLearning'], sort='top', limit=3)
    for post in posts:
        print(post)

# Sort options:
# - 'relevance': Default sort. Factors in word rarity, post age, votes, and comments.
# - 'hot': Prioritizes posts recently getting upvotes and comments. Good for trending topics.
# - 'top': Prioritizes posts with all-time high upvotes and comments. Good for well-known posts.
# - 'new': Prioritizes the newest posts, regardless of upvotes and comments. Best for up-to-date information.
# - 'comments': Prioritizes posts with the most comments. Good for posts with a lot of discussion.