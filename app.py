from flask import Flask, request, render_template
import joblib

# Load the trained model & vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)

# Home page with a form
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form["email_text"]
    transformed_text = vectorizer.transform([email_text])  # Convert to TF-IDF
    probability = model.predict_proba(transformed_text)[:, 1][0]  # Get spam probability
    
    threshold = 0.8  # Adjust this if needed
    result = "SPAM" if probability >= threshold else "HAM"
    
    print(f"Email: {email_text}")  # Debugging
    print(f"Spam Probability: {probability}")  # Debugging
    print(f"Prediction: {result}")  # Debugging
    
    return render_template("index.html", prediction=result, email_text=email_text, probability=probability)


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
