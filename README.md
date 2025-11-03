\# Soccer Match Outcome Predictor



This project predicts football (soccer) match outcomes using historical match data from multiple European leagues. The model leverages team form, head-to-head results, and recent goals scored/conceded to forecast the result of upcoming matches.



\## Features



\- \*\*Predict Match Outcome:\*\* Predicts home win, away win, or draw.  

\- \*\*Probability Breakdown:\*\* Shows the model's confidence for each possible outcome.  

\- \*\*Recent Form Analysis:\*\* Considers last 5 matches for both teams.  

\- \*\*Head-to-Head Analysis:\*\* Weights past encounters between the two teams.  

\- \*\*Multiple Leagues:\*\* Supports Spain, England, France, Italy, and Germany.  



\## Installation



Clone this repository:


```bash

git clone https://github.com/ayaannaveed05/soccer-match-predictor.git

cd soccer-match-predictor

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





