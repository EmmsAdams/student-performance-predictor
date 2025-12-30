# Student Performance Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/)

This is a machine learning project that predicts whether students will pass or fail based on their study habits and engagement.

## Why I Built This

As a Computer Science student with an interest in AI, I wanted to see how machine learning works in practice. This project covers the entire process, from cleaning data to training models and evaluating results. It serves as my hands-on exploration of the basics of data science.

## What It Does

It takes student data, such as study hours, attendance, and previous grades, to predict academic outcomes. I trained two different models and compared their performance.

**Features:**
- Looks at 7 different student factors
- Trains Random Forest and Logistic Regression models
- Achieves around 85% prediction accuracy
- Identifies which factors are most important for success
- Offers visualizations of model performance

## Tech I Used

- Python for all coding
- pandas for handling data
- scikit-learn for machine learning
- matplotlib for creating charts

## How to Run It

```bash
# Get the code
git clone https://github.com/EmmsAdams/student-performance-predictor.git
cd student-performance-predictor

# Install what you need
pip install -r requirements.txt

# Run it
python main.py
```

The script will show you:
- How the data looks
- Training progress
- Model accuracy results
- Comparison charts
- Which features are most important

## What I Learned

This project taught me much more than just following tutorials. I learned how to:
- Clean and prepare real-world data
- Choose between different machine learning algorithms
- Avoid common mistakes like overfitting
- Evaluate models properly, not just by accuracy
- Present results clearly

The biggest insight? Study hours and past grades have the most impact, but attendance and completing assignments are also important predictors.

## What's Next

Some ideas I’m considering include:
- Trying more advanced algorithms like neural networks
- Adding hyperparameter tuning to improve performance
- Building a web interface for easier access
- Testing on real student data, with permission

## About Me

I’m Emmelyn Adams, a Computer Science student at Northumbria University. I built this project independently to explore machine learning beyond what we cover in lectures. It’s part of my journey to understand how AI truly works.

**Get in touch:**
- GitHub: [@EmmsAdams](https://github.com/EmmsAdams)
- Email: emmelynadams40@gmail.com
- Location: Newcastle, UK

## Want to Contribute?

If you find a bug, have suggestions, or want to try a different approach, please open an issue or reach out. I’m always eager to learn from others.

---

*I built this to learn by doing. Machine learning becomes clearer when you actually create something with it.*
