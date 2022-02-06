import streamlit as st 
import pandas as pd
import pickle 
import gdown
from plotnine import * 
import matplotlib.pyplot as plt
import seaborn as sns
import re

#solution from https://discuss.streamlit.io/t/git-pull-failed-while-deploying/17258
model_url = 'https://drive.google.com/uc?id=1gKWQ8S6K8xfsGd3E4bXUuB6zOYVfM7Qx'
output = 'pickle.pkl'
pipe =  gdown.download(model_url, output, quiet=False) 

def remove_urls (vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)

#dataframe of tweets
df = pd.read_csv("https://raw.githubusercontent.com/tashapiro/cyberbullying-classification/main/data/cyberbullying_tweets.csv")
df['no_links_text'] = [remove_urls(i) for i in df['tweet_text']]
df['char_len'] = [len(i) for i in df.no_links_text]
df['word_count'] = [len(i.split()) for i in df.no_links_text]


with open(pipe,mode='rb') as pickle_in:
    pipe_f = pickle.load(pickle_in)

icon_dict = {"religion": "✝️ ☪️ 🕉 ☸️ ✡️ 🔯 🕎 ☯️ 🛐", "gender": "🚹 🚺🏳️‍🌈 🏳️‍⚧️", "age" : "🚌", "other_cyberbullying" : "⁉️", "ethnicity" : "👩🏾👨🏿👩🏽‍🦱"}

st.title('Policing Cyberbullying Tweets')

st.sidebar.write("**Adjust Settings**")

page = st.sidebar.selectbox(
    "Select Section",
    ("Background", "Analysis","Models", "Predict")
)

if page == "Background":
    st.header("About The Project 📣 ")
    st.subheader("Background")
    st.write('This project uses **NLP classification** to predict whether a tweet, or piece of text, contains language associated to cyberbullying or not. More specifically, the model assesses subclasses of cyberbullying (multi-classification). /n Outcomes include **religion**, **gender**, **ethnicity**, **age**, **other** and **not cyberbullying**.')
    st.write("The dataset was taken from **Kaggle**, and is comprised of 47K tweets. Tweets are evenly balanced among the 6 classes, ~8K each.")
    st.subheader("Problem Statement")
    st.write("Using NLP classification methods, can we build a model that predicts the correct classification and beats the baseline accuracy rate (16%)?")
elif page == "Analysis":
    st.header("EDA 📊")
    st.subheader("Distribution Analysis")
    st.write("How verbose are different classes of cyberbullying tweets? After removing links from the texts, we assessed distribution of character length and word counts for texts in respect to each class.")
    st.write("Tweets in the **religion** and **age** classes appear to be more verbose compared to tweets in other classes. **No cyberebullying** and **other cyebrbullying** appear to be the least verbose.")
    char_len = (ggplot(df, aes(x='char_len', fill="cyberbullying_type"))
                +geom_histogram(bins=30) 
                +facet_wrap("cyberbullying_type")
                +scale_x_continuous(limits=(0,300))
                + ylab("Count")
                + xlab("Character Length")
                +ggtitle("Character Length Distribution")
                +theme(figure_size=(12, 4),
                    text = element_text(family="Roboto"),
                    plot_title=element_text(weight='bold',color='black', size=14),
                    panel_background = element_rect(fill="white"),
                    panel_grid_major = element_line(color="#dee2e6"),
                    subplots_adjust={'wspace': 0.25},
                    legend_position="none",
                    axis_title = element_text(weight='bold'))
          )
    word_dist = (ggplot(df, aes(x='word_count', fill="cyberbullying_type"))
                +geom_histogram(bins=30) 
                +facet_wrap("cyberbullying_type")
                +scale_x_continuous(limits=(0,75))
                + ylab("Count")
                + xlab("Word Count")
                +ggtitle("Word Count")
                +theme(figure_size=(12, 4),
                    text = element_text(family="Roboto"),
                    plot_title=element_text(weight='bold',color='black', size=14),
                    panel_background = element_rect(fill="white"),
                    panel_grid_major = element_line(color="#dee2e6"),
                    subplots_adjust={'wspace': 0.25},
                    legend_position="none",
                    axis_title = element_text(weight='bold'))
          )
    st.pyplot(ggplot.draw(char_len))
    st.pyplot(ggplot.draw(word_dist))
    st.subheader("Top Words & Phrases")
elif page == "Models":
    st.header("Models 📈")
elif page == "Predict":
    model_type = st.sidebar.selectbox(
    "Select The Model",
    ("Model 1", "Model 2","Model 3")
    )
    st.header("Let's Predict 🔮")
    st.write('Are you a cyberbully?')
    input_var = st.text_input(label="Enter your Tweet")
    pred = pipe_f.predict([input_var])[0]
    if input_var == '':
        st.write("")
    elif pred == 'not_cyberbullying':
        st.success("This Tweet is clean, it is **not cyberbullying** ✅ ")
    else:
        icon = icon_dict.get(pred)
        st.error("**WARNING**: This Tweet contains offensive language associated to **cyberbullying** 🤬")
        st.warning(f'More specifically, it targets **{pred}** {icon}')













