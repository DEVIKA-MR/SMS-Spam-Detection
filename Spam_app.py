from flask import Flask, render_template,url_for,request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	# Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
	df = pd.read_table('C:/Users/user/Desktop/SPAM', names=['label', 'sms_message'])
	df['label'] = df.label.map({'spam':1, 'ham':0})

	count_vector = CountVectorizer()
# split into training and testing sets
# USE from sklearn.model_selection import train_test_split to avoid seeing deprecation warning.

	X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 
                                                    df['label'], 
                                                    random_state=1)

# Instantiate the CountVectorizer method
	count_vector = CountVectorizer()

# Fit the training data and then return the matrix
	training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
	testing_data = count_vector.transform(X_test)

	naive_bayes = MultinomialNB()
	naive_bayes.fit(training_data, y_train)
	predictions = naive_bayes.predict(testing_data)


	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = count_vector.transform(data).toarray()
		my_prediction = naive_bayes.predict(vect)
	return render_template('result.html',prediction=my_prediction)



if __name__ == '__main__':
   app.run(debug= True)















