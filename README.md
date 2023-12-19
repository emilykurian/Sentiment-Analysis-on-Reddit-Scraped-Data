# Sentiment-Analysis-on-Reddit-Scraped-Data
This project focuses on harnessing the vast repository of user-generated content on Reddit to gain insights into the sentiment of movie reviews. Sentiment analysis, a a subfield of natural language processing, serves as the core technique employed to assess the emotional tone and polarity of these comments.

The project begins by collecting a diverse set of Reddit comments related to movie reviews, spanning various genres, languages, and user demographics. Leveraging the power of machine learning and NLP models, we process and analyze these comments to extract valuable sentiment information. The sentiment analysis process involves categorizing comments as positive, negative, or neutral, thereby gauging the general sentiment of Reddit users towards specific movies.

Sentiment analysis on Reddit user comments about movie reviews offers a unique perspective on the impact of films on audiences. By analyzing the sentiment trends, identifying patterns, and uncovering valuable feedback, this project contributes to the broader discourse on the intersection of film, social media, and audience engagement. Furthermore, it showcases the potential for sentiment analysis to provide meaningful insights in the realm of entertainment and beyond.  

# Methodology
## 1. DATA USED: Reddit Scraping
To start the process of scraping Reddit data, the first step is to sign up for access to the Reddit API. The Reddit API (Application Programming Interface) allows developers to programmatically interact with Reddit's data, including accessing posts, comments, and other content on the platform. During the sign-up process, you will obtain the following credentials:  
* Client ID: This is a unique identifier for your application or script. It's used to authenticate your requests to the Reddit API. The client ID helps Reddit recognize your application and track its usage.  
* Client Secret: The client secret is another crucial piece of authentication. It should be kept secret and not shared publicly. It's used, along with the client ID, to authenticate your application and ensure secure access to Reddit's data.  
* User Agent: The user agent is a string that identifies your application or script when making requests to the Reddit API. It typically includes information about your project and how to contact you as the developer. Providing a meaningful and descriptive user agent helps Reddit track and manage API usage. Once signed up for the Reddit API and obtained these credentials, wecan proceed with using them in your Python script to access Reddit's data.  
## 2. LIBRARIES USED:
* PRAW (Python Reddit API Wrapper): PRAW is a Python library that simplifies the process of interacting with Reddit's API. It provides convenient methods and classes for accessing posts, comments, and other Reddit content. PRAW handles authentication using the client ID, client secret, and user agent you obtained
during the sign-up process.  
* Pandas: Pandas is a powerful data manipulation library in Python. It is often used to tabulate and structure data obtained from various sources, including web scraping. In this context, Pandas helps organize and analyze the data collected from Reddit.  
## 3. LABELING THE DATASET:
Adding labels to a dataset is a crucial step in preparing it for various machine learning and natural language processing tasks, including sentiment analysis. Labeling involves assigning predefined categories or values to the data instances, making it possible for machine learning algorithms to learn patterns and make predictions or classifications. Since the dataset we got from scrapping reddit is unlabelled, we need to label the raw dataset. I am using the Vader Sentiment model for labeling the dataset.    
### 3.1) Vader Sentiment model (a Lexicon method):
VADER (Valence Aware Dictionary for sEntiment Reasoning) is a model used for text sentiment analysis that is sensitive to both polarity (positive/negative) and intensity (strength) of emotion. It combines a dictionary, which maps lexical features to emotion intensity, and five simple heuristics, which encode how contextual elements increment, decrement, or negate the sentiment of text.  

Text Sentiment Analysis is a really big field with a lot of academic literature behind it. However, its tools really just boil down to two approaches: the lexical approach and the
machine learning approach.  

Lexical approaches aim to map words to sentiment by building a lexicon or a ‘dictionary of sentiment.’ We can use this dictionary to assess the sentiment of phrases and sentences, without the need of looking at anything else. Sentiment can be categorical — such as {negative, neutral, positive} — or it can be numerical — like a range of intensities or scores. Lexical approaches look at the sentiment category or score of each word in the sentence and decide what the sentiment category or score of the whole sentence is. The power of lexical approaches lies in the fact that we do not need to train a model using labeled data, since we have everything we need to assess the sentiment of sentences in the dictionary of emotions. VADER is an example of a lexical method.  
#### 3.1.1) Quantifying the Emotion of a Word (or Emoticon) and sentence:
Primarily, VADER sentiment analysis relies on a dictionary which maps lexical features to emotion intensities called sentiment scores. The sentiment score of a text can be obtained by summing up the intensity of each word in the text. Individual words have a sentiment score between -4 to 4, but the returned sentiment score of a sentence is between -1 to 1. The sentiment score of a sentence is the sum of the sentiment score of each sentiment-bearing word. However, normalization is applied to the total to map it to
a value between -1 to 1.  

The normalization used by Hutto is:  
$$\frac{x}{\sqrt{x^2 +α}}$$   

where x is the sum of the sentiment scores of the constituent words of the sentence and alpha is a normalization parameter that we set to 15.The normalization is graphed below.  

![image](https://github.com/emilykurian/Sentiment-Analysis-on-Reddit-Scraped-Data/assets/49084104/7cdbd7b1-5407-467b-8dce-93d4e928e8c7)  

We see here that as x grows larger, it gets more and more close to -1 or 1. To similar effect, if there are a lot of words in the document you’re applying VADER sentiment analysis to, you get a score close to -1 or 1. Thus, VADER sentiment analysis works best on short documents, like tweets and sentences, not on large documents.  

### 3.2) Labeling:
According to the industry standards, if the compound score of sentiment is more than 0.05, then it is categorized as Positive, and if the compound score is less than -0.05, then it is categorized as Negative, otherwise, it’s neutral.  

## 4. NLP
Now that we have the dataset scrapped and labeled, we need to preprocess and clean it so that it can be fed to the machine learning model. The following are performed:
1. remove links and all the special characters from the feature column  
2. tokenize and remove the stopwords from the feature column  
3. stem the words in the feature column
   
### 4.1) Tokenization:  
Tokenization is used in natural language processing to split paragraphs and sentences into smaller units that can be more easily assigned meaning. The first step of the NLP process is gathering the data (a sentence) and breaking it into understandable parts (words).
Here’s an example of a string of data:  
“What restaurants are nearby?“  
In order for this sentence to be understood by a machine, tokenization is performed on the string to break it into individual parts. With tokenization, we’d get something like this: ‘what’ ‘restaurants’ ‘are’ ‘nearby’  

### 4.2) Stemming:
Stemming is the process of reducing the word to its word stem that affixes to suffixes and prefixes or to roots of words known as a lemma. In simple words stemming is reducing a word to its base word or stem in such a way that the words of similar kind lie under a common stem. For example – The words care, cared and caring lie under the
same stem ‘care’. Drawbacks of stemming is it does not consider how the word is being used. For example – the word ‘saw‘ will be stemmed to ‘saw‘ itself but it won’t be considered
whether the word is being used as a noun or a verb in the context. For this reason, Lemmatization is used as it keeps this fact in consideration and will return either ‘see’ or ‘saw’ depending on whether the word ‘saw’ was used as a verb or a noun.  

### 4.3) Vectorization:
Word Embeddings or Word vectorization is a methodology in NLP to map words or phrases from vocabulary to a corresponding vector of real numbers which is used to find word predictions, word similarities/semantics. In simple words the process of converting words into numbers is called Vectorization. Broadly, we can classified word embeddings into the following two categories:  
* Frequency-based or Statistical based Word Embedding  
Eg: Count Vector, TF-IDF Vector  
* Prediction based Word Embedding  
Eg: CBOW, Skip-Gram Model

#### 4.3.1) Count vectorizer  

Count vectorizer will fit and learn the word vocabulary and try to create a document term matrix in which the individual cells denote the frequency of that word in a particular document, which is also known as term frequency, and the columns are dedicated to each word in the corpus.  

Let’s consider the following example:   
Document-1: He is a smart boy. She is also smart.  
Document-2: Chirag is a smart person.  

The dictionary created contains the list of unique tokens(words) present in the corpus  
Unique Words: [‘He’, ’She’, ’smart’, ’boy’, ’Chirag’, ’person’]
Here, D=2, N=6  
So, the count matrix M of size 2 X 6 will be represented as –  

|     | He   | She  | smart | boy  | Chirag | person | 
| --- | ---- | ---- | ----- |  --- | ------ | ------ |
|D1   |1     |  1   | 2     | 1    | 0      | 0      |
|D2   |0     | 0    | 1     | 0    | 1      | 1      |  

Count Vectors can help understand the type of text by the frequency of words in it. But its major disadvantages are:  
* Its inability in identifying more important and less important words for analysis.  
* It will just consider words that are abundant in a corpus as the most statistically significant word.  
* It also doesn’t identify the relationships between words such as linguistic similarity between words.
  
## 5. Model Training
The final step in the process of NLP is to classify or cluster texts. As we are working on the problem of sentiment classification, we will now train a text classification model. I am using a Passive Aggressive Classifier.  

### 5.1) Passive Aggressive Classifier
Just like supervised and unsupervised there are other categories of machine learning such as:  
* Reinforcement Learning
* Batch Learning
* Online Learning
* Instance-Based
* Model-Based
  
Passive Aggressive Classifier is an online learning algorithm where you train a system incrementally by feeding it instances sequentially, individually or in small groups called mini-batches. In online machine learning algorithms, the input data comes in sequential order and the machine learning model is updated step-by-step, as opposed to batch learning, where the entire training dataset is used at once. So we can say that an algorithm like Passive Aggressive Classifier is best for systems that receive data in a continuous stream. This is very useful in situations where there is a huge amount of data and it is computationally infeasible to train the entire dataset because of the sheer size of the data. A very good example of this would be to detect fake news on a social media website like Twitter, where new data is being added every second.  
Passive-Aggressive algorithms are called so because :  
* Passive: If the prediction is correct, keep the model and do not make any changes. i.e., the data in the example is not enough to cause any changes in the model.  
* Aggressive: If the prediction is incorrect, make changes to the model. i.e., some change to the model may correct it.  
The PA algorithm updates the model’s parameters in a way that tries to correct the mistake while also trying to keep the change in the parameters as small as possible. This helps the model to generalize well and avoid overfitting.  

# CONCLUSION
In this project, we embarked on a journey through the realm of sentiment analysis, leveraging Reddit's vast repository of movie reviews as our primary data source. Our objective was to gain deep insights into the sentiments expressed by Reddit users about various movies and to develop a robust sentiment analysis system.   

Our journey began with the collection of Reddit comments through web scraping, ensuring a diverse and comprehensive dataset encompassing different genres, languages, and user demographics. This unstructured raw data was the canvas on which we applied advanced natural language processing (NLP) techniques to extract meaningful insights.  

The data preprocessing phase was instrumental in shaping our dataset for analysis. We systematically cleaned the text data by converting it to lowercase, removing special characters, eliminating URLs and HTML tags, and performing stemming using the Snowball stemmer. The removal of stopwords further refined our text, ensuring that only relevant content contributed to our analysis.  

The transformation of textual data into numerical features was achieved through the process of vectorization, specifically using the Count Vectorizer. This technique converted the text into a matrix of token counts, making it ready for consumption by machine learning algorithms.  

The heart of our sentiment analysis system lay in the application of the Passive Aggressive classification machine learning model. This model, known for its efficiency and effectiveness in text classification tasks, was trained on our labeled dataset. Through extensive training and testing, the model learned to discern the sentiment expressed in Reddit comments, classifying them as positive, negative, or neutral.  

As we delved deeper into the project, we uncovered valuable insights into the sentiments surrounding movies. The sentiment scores provided by our model illuminated trends and patterns in user opinions, enabling us to gauge audience reactions accurately. These insights can be invaluable to movie studios, critics, and enthusiasts seeking to understand the collective public opinion about films.  

In conclusion, our project exemplifies the power of data-driven sentiment analysis and machine learning in deciphering the sentiments encapsulated in Reddit's movie reviews. It serves as a testament to the effectiveness of advanced NLP techniques, data preprocessing, and classification models in gaining meaningful insights from unstructured text data. The journey through this project has not only deepened our understanding of sentiment analysis but also illuminated the profound impact it can have on understanding public perception in the world of cinema and beyond.  

