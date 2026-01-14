Soccer Match Predictor
An interactive web application that predicts European soccer match outcomes using machine learning.

Features

5 European Leagues: Spain (La Liga), England (Premier League), France (Ligue 1), Italy (Serie A), Germany (Bundesliga)
Machine Learning Model: Random Forest classifier with 9 engineered features
Real-time Predictions: Instant match outcome predictions with probability breakdowns
Historical Data: Trained on 3 seasons of match data (1,500+ matches)
Interactive UI: User-friendly interface built with Streamlit

Live Demo: https://soccer-match-predictor.streamlit.app

How It Works
The model uses 9 key features to predict match outcomes:

1. Recent Form Metrics

Home team recent goals scored/conceded
Away team recent goals scored/conceded
Home and away form indicators


2. Home Advantage Factor

Calculated from rolling 5-game home vs away performance


3. Head-to-Head History

Weighted average of last 5 H2H matches

Model Performance

Accuracy: 63.5% (Logistic Regression baseline)
Improvement over naive baseline: +20 percentage points
Key Finding: Recent form accounts for 46% of predictions, while H2H history contributes only 7%

Tech Stack

Backend: Python 3.14
ML Framework: Scikit-Learn (Random Forest, GridSearchCV, TimeSeriesSplit)
Data Processing: Pandas, NumPy
Web Framework: Streamlit
Data Source: football-data.co.uk

Project Structure

soccer-predictor/
├── app.py              # Streamlit web application
├── predictor.py        # Original CLI predictor with full analysis
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── data/              # CSV data files (3 seasons per league)

Key Learnings

Feature Engineering: Rolling averages and weighted H2H history significantly improve predictions
Temporal Validation: Using TimeSeriesSplit prevents data leakage in time-series ML
Model Selection: Simpler models (Logistic Regression) can outperform complex ones (Random Forest) for linear relationships
Class Imbalance: Draws are inherently difficult to predict (9% recall) due to their unpredictable nature

Data Source
Historical match data provided by football-data.co.uk

Author

Built as part of a machine learning portfolio project demonstrating:
=======
```bash
>>>>>>> 462634048a93b0bf1da2a19f4bae1836787e41a7

End-to-end ML pipeline development
Feature engineering and model selection
Web application deployment
Data analysis and visualization


<<<<<<< HEAD
=======
Create a virtual environment (optional):

python -m venv venv
\# macOS/Linux
source venv/bin/activate
\# Windows
venv\\Scripts\\activate

Install dependencies:
pip install -r requirements.txt

Dependencies include: pandas, requests, scikit-learn

\##Usage
Run the main script:
python model.py

Enter the following when prompted:
Team name (e.g., Real Madrid)
Opponent team name (e.g., Betis)
Home or away

The program outputs:
Accuracy of the model on historical data
Most recent head-to-head matches
Last 5 matches for both teams
Predicted winner and probability breakdown

\##Data

Data for the 2025/26 season is automatically downloaded from Football-Data.co.uk
Older seasons are included in the data/ folder for better model training

Git Ignore

Make sure the .gitignore includes:
\_\_pycache\_\_/
\*.pyc
data/
venv/

Notes
The model is a Random Forest Classifier trained on team form, goals scored/conceded, and head-to-head stats.
Predictions are only as good as historical data and recent form. Unexpected events (injuries, red cards, transfers) are not considered.
>>>>>>> 462634048a93b0bf1da2a19f4bae1836787e41a7





