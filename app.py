# Import Flask class from the flask module
from flask import Flask, render_template, request
import pickle


# Create an instance of Flask class
app = Flask(__name__)

cv = pickle.load(open("models\cv.pkl", 'rb'))
clf = pickle.load(open("models\clf.pkl", 'rb'))

# Register a route
@app.route('/')
def home():
    return render_template('index.html')

@app.route("/predict", methods=["post"])
def predict():
    email = request.form.get('email')
    # predict email
    print(email)
    X = cv.transform([email]) 
    prediction = clf.predict(X)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", response=prediction)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)


