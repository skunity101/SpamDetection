from django.shortcuts import render

# Create your views here.
import pandas as pd # to handle our dataset

from sklearn.feature_extraction.text import CountVectorizer #for converting text data into numbers

from sklearn.model_selection import train_test_split #to split data

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score

from .forms import MessageForm

dataset = pd.read_csv('C:/Users/kiten/OneDrive/Desktop/spam detection/emails.csv')

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(dataset['text'])

x_train, x_test, y_train, y_test = train_test_split(X, dataset['spam'], test_size=0.2)

model = MultinomialNB()
model.fit(x_train, y_train)

#function to predict if message is spam
def predict_spam(message):
    mess_vector = vectorizer.transform([message])
    prediction = model.predict(mess_vector)
    return 'Spam' if prediction[0] == 1 else 'Safe'

def Home(request):
    result = None
    if request.method == 'POST':
        form = MessageForm(request.POST)
        if form.is_valid():
            message = form.cleaned_data['text']
            result = predict_spam(message)
    else:
            form = MessageForm()
    return render(request, 'home.html', {'form': form, 'result': result})
    
    
