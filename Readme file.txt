📘 README.txt

Project Title: Employee Sentiment Analysis
Submitted by: Chandan Shakya
Date: 04 October 2025

🧠 Objective:

The goal of this project is to analyze employee emails and determine their sentiment (Positive, Neutral, or Negative) based on the text content of the “Subject” and “Body” columns.

This task is done using Machine Learning techniques with scikit-learn and PyTorch to demonstrate basic text preprocessing, feature extraction, and classification.

📂 Dataset Information:

File Name: test(in).csv
Total Records: 2191
Columns:

Subject – Email subject line

body – Main content of the email

date – Date and time of the message

from – Sender of the email

There are no missing values in this dataset.

⚙️ Tools & Libraries Used:

Python

Pandas

NumPy

Matplotlib & Seaborn (for visualization)

Scikit-learn (for text preprocessing & ML model)

PyTorch (for building simple neural model)

NLTK (for text cleaning)

🧹 Step-by-Step Workflow:

Importing Libraries
All required Python libraries are imported (pandas, numpy, sklearn, torch, etc.)

Loading the Dataset
The CSV file test(in).csv is loaded using pandas.

Data Exploration

Checked shape, columns, and missing values.

Displayed sample data using head().

Data Cleaning

Converted all text to lowercase.

Removed unwanted characters, punctuation, and extra spaces.

Text Preprocessing

Tokenization and stopword removal using NLTK.

Combined Subject and body into one text column for better sentiment detection.

Feature Extraction

Used TfidfVectorizer from sklearn to convert text into numerical form.

Model Building (Sklearn)

Used Logistic Regression for sentiment classification.

Split data into training and testing sets (80/20).

Trained and tested the model.

Model Building (PyTorch)

Created a simple neural network using torch.nn.

Used ReLU activation and cross-entropy loss.

Evaluated accuracy after training.

Model Evaluation

Checked Accuracy, Precision, Recall, and Confusion Matrix.

Displayed performance using matplotlib graphs.

Result

Model successfully classified sentiments into 3 categories:

Positive

Neutral

Negative

📊 Outcome:

The project demonstrates the complete pipeline for text sentiment analysis, from raw email data to model prediction.
Both traditional ML (Logistic Regression) and Neural Network (PyTorch) methods are implemented in simple and understandable code.