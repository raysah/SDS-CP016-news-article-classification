import streamlit as st
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import re
st.set_page_config(layout="wide")

style_heading = 'text-align: center'
# Previous news state
if "prev_news" not in st.session_state:
    st.session_state["prev_news"] = ""


def default_news():
    """
    Example news

    Returns:
    example_news: example news
    """
    return '''
    Shubman Gill must have wanted to feel good about himself before he got down to work. 
    At the start of India's training session in Canberra on Friday, he went around to the back of the nets facility, 
    where only throwdowns were possible, and that too from about 10 yards, and for the first few minutes, he was totally freestyling. 
    Big booming drives. Lovely back foot punches.After a little bit of this, he asked India's assistant coach Ryan ten Doeschate to help him drill down on his defence. 
    Now the balls were coming down on a good length and he focused on meeting them under his eyes. 
    Somehow the sound off the bat was louder than he was trying to whack them. 
    That left thumb that he injured in Perth doesn't seem to be giving him a whole lot of problems anymore.
    Gill began with throwdowns from up close, then graduated to facing them off the sidearm and then capped it off by fronting up against Akash Deep and Yash Dayal at full speed. 
    "He is batting right now and our physio will evaluate him and I will know his status after that," the other assistant coach Abhishek Nayar said on Friday afternoon. 
    "But from what I have seen, he is looking comfortable batting and he looks like he can bat [in a match]. 
    He is batting in the indoor nets and we will know if he can play the practice match or not."
    '''


@st.cache_resource
def load_classical_learning_model():
    """
    Loads a machine learning model.

    """
    model_path = r'/Users/dots/PycharmProjects/SDS-CP016-news-article-classification/web-app/dattu-sahoo/models/news_classification.pkl'
    pickle_in = open(model_path, 'rb')
    classifier = pickle.load(pickle_in)

    return classifier

def remove_tags(text):
    remove = re.compile(r'')
    return re.sub(remove, '', text)

def special_char(text):
  reviews = ''
  for x in text:
    if x.isalnum():
      reviews = reviews + x
    else:
      reviews = reviews + ' '
  return reviews

def convert_lower(text):
   return text.lower()
def classify(text):
    """
    Classifies the article text using a classical machine learning model.
    The article is classified as either business, entertainment, politics, sport, or tech.

    Parameters:
    text (str): the news article text.

    Returns:
    category: the article's classification.
    """
    input_file = r'/Users/dots/PycharmProjects/SDS-CP016-news-article-classification/notebooks/dattu-sahoo/bbc_text_cls.csv'
    df_input_text = pd.read_csv(input_file)
    df_input_text['LabelID'] = df_input_text['labels'].factorize()[0]
    df_input_text['text'] = df_input_text['text'].apply(remove_tags)
    df_input_text['text'] = df_input_text['text'].apply(special_char)
    df_input_text['text'] = df_input_text['text'].apply(convert_lower)

    y = np.array(df_input_text.LabelID.values)
    cv = CountVectorizer(max_features=5000)
    x = cv.fit_transform(df_input_text.text).toarray()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=True)
    classifier = MultinomialNB(alpha=1.0, fit_prior=True).fit(x_train, y_train)
    y_pred = cv.transform([text])
    yy = classifier.predict(y_pred)

    if yy == [0]:
        return "Business News"
    elif yy == [1]:
        return "Tech News"
    elif yy == [2]:
        return "Politics News"
    elif yy == [3]:
        return "Sports News"
    elif yy == [1]:
        return "Entertainment News"

def main():
    st.markdown(f"<h1 style='{style_heading}'>Classify and Summarize News Article</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<h5 style='{style_heading}'>Type of news in scope for classification : Business, Entertainment, Politics, Sport, or Tech.</h5>",
        unsafe_allow_html=True)
    st.text("")
    st.text("")
    st.text("")
    st.markdown(f"<h3 style='{style_heading}'>Types of Models evaluated</h3>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    with st.container():
        c1.write("Logistic Regression")
        c2.write("Random Forest")
        c3.write("Multinomial Naive Bayes")

    with st.container():
        c4.write("Support Vector Classifier")
        c5.write("Decision Tree Classifier")
        c6.write("K Nearest Neighbor")

    # Text input area for the news article
    news_text = st.text_area("Paste news article here:", value=default_news().strip(), height=250)

    classification_header = st.empty()
    classification_results = st.empty()
    classification_dl_results = st.empty()
    summarization_header = st.empty()
    summarization_results = st.empty()

    # Check if the article text has changed
    if news_text != st.session_state["prev_news"]:
        st.session_state["prev_news"] = news_text
        classification_header.empty()
        classification_results.empty()
        classification_dl_results.empty()
        summarization_header.empty()
        summarization_results.empty()

        # Buttons for actions
    _, _, col1, col2, _, _ = st.columns(6)
    with col1:
        classify_button = st.button("Classify Article")
    with col2:
        summarize_button = st.button("Summarize Article")

    if classify_button:
        if news_text.strip():
            with st.spinner("Classifying the article..."):
                classification_header.subheader("Classification", divider=True)
                classification_results.markdown(f'### Using Multinomial Naive Bayes model: :blue[{classify(news_text)}]')
        else:
            st.error("Please paste a news article to classify.")


# Run the app
if __name__ == "__main__":
    main()