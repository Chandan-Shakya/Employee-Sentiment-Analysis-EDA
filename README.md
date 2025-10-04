# 📘 **Employee Sentiment Analysis**

**👨‍💻 Submitted by:** *Chandan Shakya*  
**📅 Date:** *04 October 2025*  

---

## 🧠 **Objective**

The goal of this project is to **analyze employee emails and determine their sentiment** —  
classified as **Positive**, **Neutral**, or **Negative** — based on the text content of the **Subject** and **Body** columns.

This analysis demonstrates **Natural Language Processing (NLP)** and **Machine Learning** techniques using  
**scikit-learn** and **PyTorch**, covering:
- Text preprocessing  
- Feature extraction  
- Sentiment classification  
- Model evaluation and visualization

---

## 📂 **Dataset Information**

**File Name:** `test(in).csv`  
**Total Records:** `2191`  

**Columns:**
| Column | Description |
|---------|--------------|
| `Subject` | Email subject line |
| `body` | Main content of the email |
| `date` | Date and time of the message |
| `from` | Sender of the email |

✅ **No missing values** were found in the dataset.

---

## ⚙️ **Tools & Libraries Used**

| Category | Libraries |
|-----------|------------|
| Programming | Python |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| NLP & Text Cleaning | NLTK |

---

## 🧹 **Step-by-Step Workflow**

### 1️⃣ **Importing Libraries**
All required Python libraries are imported:
```python
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split


2️⃣ Loading the Dataset

Loaded the CSV file using pandas:

df = pd.read_csv("test(in).csv")

3️⃣ Data Exploration

Checked shape, column names, and missing values.

Displayed sample data using df.head().

Verified data quality for further preprocessing.

4️⃣ Data Cleaning

Converted all text to lowercase.

Removed punctuation, special symbols, and extra spaces.

Ensured all messages were cleaned for consistent processing.

5️⃣ Text Preprocessing

Tokenized each sentence.

Removed stopwords using nltk.corpus.stopwords.

Combined Subject and Body columns into one text column for improved context.

6️⃣ Feature Extraction

Used TF-IDF (Term Frequency - Inverse Document Frequency) from sklearn to convert text into numeric format:

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])

7️⃣ Model Building (Scikit-learn)

Split dataset into training (80%) and testing (20%).

Applied Logistic Regression for sentiment classification.

Trained the model and checked accuracy using:

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

8️⃣ Model Building (PyTorch)

Built a simple Neural Network using torch.nn.

Used ReLU activation and Cross-Entropy Loss.

Trained for multiple epochs and evaluated test accuracy.

9️⃣ Model Evaluation

Checked Accuracy, Precision, and Recall metrics.

Plotted Confusion Matrix using Seaborn for visualization.

Compared the results between Logistic Regression and PyTorch models.

🏆 Result Summary

The project successfully classified employee messages into three sentiment categories:

Sentiment	Description
😊 Positive	Represents appreciation, agreement, or motivation
😐 Neutral	Represents factual or non-emotional messages
😞 Negative	Represents complaints, disagreement, or frustration
📊 Outcome & Insights

End-to-end pipeline from raw text to sentiment classification demonstrated.

Implemented both traditional ML and Neural Network (PyTorch) approaches.

Simple, reproducible, and well-documented code for learning or evaluation.

Can be extended to analyze employee engagement or organizational mood trends.

💡 Future Improvements

Integrate BERT or Transformer-based models for deeper understanding.

Develop a Power BI Dashboard to visualize real-time sentiment trends.

Automate monthly sentiment scoring for HR analytics.

🧾 Project Deliverables
File Name	Description
Employee_Sentiment_Project.ipynb	Main Python notebook
Final_Report.docx	Full project report
visualizations/	All charts and analysis graphs
README.md	Summary and documentation
requirements.txt	Library dependencies
📬 Contact Information

👤 Name: Chandan Shakya
🎓 Role: Data Analyst | NLP & ML Enthusiast
📧 Email: dataanalystchs@gmail.com
🌐 GitHub: github.com/ChandanShakya
