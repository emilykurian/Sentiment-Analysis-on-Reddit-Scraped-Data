import praw
import pandas as pd
from praw.models import MoreComments
import pandas as pd
post_comments = []
 
reddit_read_only = praw.Reddit(client_id="",         # your client id
                               client_secret="",      # your client secret
                               user_agent="")        # your user agent
 
# URL of the post
url = "https://old.reddit.com/r/climatechange/comments/16wg9w9/canada_wildfires_continue_to_accelerate_as_much/"
 
# Creating a submission object
submission = reddit_read_only.submission(url=url)

for comment in submission.comments:
    if type(comment) == MoreComments:
        continue
 
    post_comments.append(comment.body)
 
# creating a dataframe
comments_df = pd.DataFrame(post_comments, columns=['comment'])
comments_df.to_csv("ClimateChangeComments.csv", index=True)