from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import re
import pickle


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

def show_news_category(news_input):
    input_file = r'/Users/dots/PycharmProjects/SDS-CP016-news-article-classification/notebooks/dattu-sahoo/bbc_text_cls.csv'
    df_input_text = pd.read_csv(input_file)
    df_input_text['LabelID'] = df_input_text['labels'].factorize()[0]
    #category = df_input_text[['LabelID', 'labels']].drop_duplicates().sort_values('LabelID')
    df_input_text['text'] = df_input_text['text'].apply(remove_tags)
    df_input_text['text'] = df_input_text['text'].apply(special_char)
    df_input_text['text'] = df_input_text['text'].apply(convert_lower)

    #x = np.array(df_input_text.iloc[:, 0].values)
    y = np.array(df_input_text.LabelID.values)
    cv = CountVectorizer(max_features=5000)
    x = cv.fit_transform(df_input_text.text).toarray()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0, shuffle=True)
    classifier = MultinomialNB(alpha=1.0, fit_prior=True).fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    y_pred1 = cv.transform([news_input])
    yy = classifier.predict(y_pred1)

    model_path = r'/Users/dots/PycharmProjects/SDS-CP016-news-article-classification/web-app/dattu-sahoo/models/news_classification.pkl'
    pickle_out = open(model_path, 'wb')
    pickle.dump(classifier, pickle_out)
    pickle_out.close()


    if yy == [0]:
        return "Business News"
    elif yy == [1]:
        return "Tech News"
    elif yy == [2]:
        return "Politics News"
    elif yy == [3]:
        return "Sports News"
    elif yy == [4]:
        return "Entertainment News"


def main():
    news_input = """The kiwifruit industry has experienced remarkable growth, with its global market value surpassing A$10 billion in 2018. 
    Projections indicate that by 2025, global consumption will approach 6 million tonnes, expanding at an annual rate of 3.9%. 
    Zespri, the world's largest kiwifruit marketer, manages over 30% of global supply, collaborating with more than 2,500 growers in New Zealand and 1,500 internationally. 
    To streamline its complex operations and meet rising demand, Zespri adopted SAP S/4HANA Cloud, enhancing its supply chain management and data analysis capabilities. 
    This digital transformation enables Zespri to efficiently deliver ripe kiwifruit year-round to consumers worldwide.."""

    print(show_news_category(news_input))


# Run the app
if __name__ == "__main__":
    main()
