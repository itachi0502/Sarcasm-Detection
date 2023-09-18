from pynytimes import NYTAPI
nyt = NYTAPI("1EAsLuGJb1HZkQyGnAGCGPlwUCZNTQie", parse_dates=True)
articles = nyt.article_search(query="Obama")
print(articles)