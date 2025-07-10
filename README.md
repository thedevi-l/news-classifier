# 📰 Fake News Classifier

A Streamlit web app that detects whether a news article is **REAL** or **FAKE** using a logistic regression model trained on custom-labeled data.  
This version is optimized for **interactive use and deployment on Replit**.

## 💡 Features

- Input any news text to check if it’s fake or real
- Trained on two datasets: `real.csv` and `fake.csv`
- Shows model accuracy visually with a bar chart
- Uses TF-IDF vectorization with n-grams for better text understanding
- Displays prediction **confidence score**
- Clean UI built with **Streamlit**

### 📦 Files

| File            | Description                                    |
|-----------------|------------------------------------------------|
| `app.py`        | Main Streamlit interface                       |
| `model.py`      | Handles data loading, preprocessing, training  |
| `main.py`       | Optional launcher file for Replit              |
| `requirements.txt` | Python dependencies                        |
| `data/real.csv` | Sample dataset with real news articles         |
| `data/fake.csv` | Sample dataset with fake news                  |

---

## 🚀 How to Run Locally

1. Clone the project
2. Install dependencies
3. Run the app

## 🌐 How to Run on Replit

1. Create a new Repl
2. Upload all project files
3. Click "Run"

