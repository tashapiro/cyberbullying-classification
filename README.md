# cyberbullying-classification
Classifying tweets using NLP to determine if they are a subclass of cyberbullying (e.g. gender, religion) or not.


Tasked with with developing an algorithm to identify cyberbullying tweets by a fictitious tech company, our group sought to build a classification model that would beat the baseline accuracy score of ~17%. We are taking it a step further and attempting to include multi-class classifications regarding what type of cyberbullying targeting the tweets display, such as gender, race, religion, etc. Despite the low base-line accuracy score to begin with, our goal was to create a model that would achieve at least 70%+ accuracy.


We began with a dataset acquired from Kaggle with a list of tweets gathered using a Dynamic Query Extension that classifies the tweets and a twitter scraper (GetOldTweets) created by Wang et al. that gathers tweets beyond the range of the Twitter API. We began with preprocessing the text, removing URLs, usernames, and any @ symbols, as well as the default english stop words and HTML artifacts. We decided to keep hashtags and emojis to help classify certain texts that might have been targeted toward a specific event or emotion. We then did EDA on the modified texts, looking at character lengths and word counts of the tweets, as well as their most common word usage and some common bigram phrases and created visuals for those counts. With the corpus modified to have the aforementioned symbols and words removed, we implemented a handful of models to test out which models would produce high accuracy scores. We used multinomial logistic regression, multinomial naive bays, a random forest classifier, and a support vector classifier, and we tested out these four models with a count vectorizer version, and a TF-IDF version. Our multionimal logistic regression with TF-IDF was our best performing model, giving us a train/test score of 0.888, 0.824, exceeding our goal of 70%+ accuracy. Using a confusion matrix to investigate which tweets were misclassified, we discovered that the tweets which were more specifically targeted, such as age, religion, race, had an F1 score over 0.95 which was very good. However, the more vague and un-targeted tweets that would fall into either "not cyberbullying" or "other cyberbullying" performed a bit worse, at around 0.56 F1 score for "not cyberbullying", and 0.63 for "other". We decided to take our model and apply it to a seperate corpus of tweets collected from Kaggle that were all tweeted by Donald Trump. It did well in modeling whether his tweets were considered cyberbullying or not, but was not able to classify the tweets specifically as the multi-class classifications very strongly.

Additionally, we tested a SentenceBERT text embedding method and a classifier neural net. The train/test accuracy scores were similar to multinomial logistic regression with TF-IDF. 

In conclusion, we succeeded in creating a model that outperformed the baseline accuracy score and our initial goal of 70% accuracy. We realized that the model struggled with certain emotions like sarcasm or someone using vulgar language when they were actually saying something positive and not cyberbullying anyone. Furthermore, since defining bullying can be subjective, there were likely potential errors in classifcation due to that fact as well.


Tanya Shapiro created a streamlit app that can attempt to classify tweets or text as cyberbullying or not, the streamlit file is located in the repo


Sources
Wang, J., Fu, K., Lu, C. SOSNet: A Graph Convolutional Network Approach to Fine-Grained Cyberbullying Detection. Proceedings - IEEE International Conference on Big Data (Big Data): 1699-1708, 2020.
GetOldTweets3. 2019. https://pypi.org/project/GetOldTweets3/
https://www.kaggle.com/austinreese/trump-tweets
