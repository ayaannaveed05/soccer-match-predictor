import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import os

st.set_page_config(
    page_title="⚽ Soccer Match Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Soccer Match Predictor")
st.markdown("Predict European soccer match outcomes using machine learning")

# Cache data loading
@st.cache_data
def load_data():
    files = {
        "SP1.csv": "Spain",
        "SP1 (1).csv": "Spain",
        "SP1 (2).csv": "Spain",
        "E0.csv": "England",
        "E0 (1).csv": "England",
        "E0 (2).csv": "England",
        "F1.csv": "France",
        "F1 (1).csv": "France",
        "F1 (2).csv": "France",
        "I1.csv": "Italy",
        "I1 (1).csv": "Italy",
        "I1 (2).csv": "Italy",
        "D1.csv": "Germany",
        "D1 (1).csv": "Germany",
        "D1 (2).csv": "Germany"
    }
    
    dfs = []
    for file, league in files.items():
        path = os.path.join("data", file)
        if os.path.exists(path):
            temp_df = pd.read_csv(path)
            temp_df['league'] = league
            dfs.append(temp_df)
    
    if not dfs:
        st.error("No data files found! Make sure CSV files are in the 'data' folder.")
        st.stop()
    
    df = pd.concat(dfs, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    df.rename(columns={
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'home_goals',
        'FTAG': 'away_goals',
        'FTR': 'result'
    }, inplace=True)
    df['winner'] = df['result'].map({'H': 'home', 'A': 'away', 'D': 'draw'})
    
    return df

@st.cache_resource
def train_models(df):
    """Train models for each league"""
    models = {}
    
    for league in df['league'].unique():
        df_league = df[df['league'] == league].copy()
        df_league = df_league.sort_values('Date')
        
        # Feature engineering
        df_league['home_recent_goals'] = df_league.groupby('home_team')['home_goals'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        df_league['home_recent_conceded'] = df_league.groupby('home_team')['away_goals'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        df_league['away_recent_goals'] = df_league.groupby('away_team')['away_goals'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        df_league['away_recent_conceded'] = df_league.groupby('away_team')['home_goals'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        
        df_league['home_form'] = df_league['home_recent_goals'] - df_league['home_recent_conceded']
        df_league['away_form'] = df_league['away_recent_goals'] - df_league['away_recent_conceded']
        
        df_league['home_team_home_form'] = df_league.groupby('home_team')['home_form'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        df_league['away_team_away_form'] = df_league.groupby('away_team')['away_form'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        df_league['home_advantage'] = df_league['home_team_home_form'] - df_league['away_team_away_form']
        
        # H2H features (simplified for speed)
        weights = [0.4, 0.3, 0.15, 0.1, 0.05]
        h2h_home_goals, h2h_away_goals = [], []
        
        for idx, row in df_league.iterrows():
            team = row['home_team']
            opponent = row['away_team']
            match_date = row['Date']
            
            h2h = df_league[
                (
                    ((df_league['home_team'] == team) & (df_league['away_team'] == opponent)) |
                    ((df_league['home_team'] == opponent) & (df_league['away_team'] == team))
                ) & (df_league['Date'] < match_date)
            ].sort_values('Date', ascending=False).head(5)
            
            hg = ag = 0
            for i, (_, r) in enumerate(h2h.iterrows()):
                w = weights[i] if i < len(weights) else 0
                if r['home_team'] == team:
                    hg += r['home_goals'] * w
                else:
                    ag += r['away_goals'] * w
            
            h2h_home_goals.append(hg)
            h2h_away_goals.append(ag)
        
        df_league['h2h_home_goals'] = h2h_home_goals
        df_league['h2h_away_goals'] = h2h_away_goals
        
        features = [
            'home_recent_goals', 'away_recent_goals',
            'home_recent_conceded', 'away_recent_conceded',
            'home_form', 'away_form',
            'home_advantage',
            'h2h_home_goals', 'h2h_away_goals'
        ]
        
        le = LabelEncoder()
        df_league['winner_encoded'] = le.fit_transform(df_league['winner'])
        
        # Drop NaN
        df_league_clean = df_league.dropna(subset=features + ['winner_encoded'])
        
        X = df_league_clean[features]
        y = df_league_clean['winner_encoded']
        
        # Train model (simplified - no GridSearch for speed)
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X, y)
        
        # Get unique teams
        teams = sorted(list(set(df_league['home_team'].unique()) | set(df_league['away_team'].unique())))
        
        models[league] = {
            'model': model,
            'label_encoder': le,
            'df': df_league,
            'features': features,
            'teams': teams
        }
    
    return models

# Load data
with st.spinner("Loading data..."):
    df = load_data()

# Train models
with st.spinner("Training models (this may take a moment)..."):
    models = train_models(df)

# Sidebar
st.sidebar.header("⚙️ Settings")
league = st.sidebar.selectbox(
    "Select League",
    list(models.keys()),
    help="Choose which league to predict"
)

# Show model info
with st.sidebar.expander("ℹ️ Model Info"):
    st.write(f"**League:** {league}")
    st.write(f"**Teams:** {len(models[league]['teams'])}")
    st.write(f"**Matches:** {len(models[league]['df'])}")
    st.write(f"**Features:** {len(models[league]['features'])}")

# Main content
teams = models[league]['teams']

st.header(f"🏆 {league} Prediction")

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox(
        "🏠 Home Team",
        teams,
        key="home",
        help="Select the team playing at home"
    )

with col2:
    away_teams = [t for t in teams if t != home_team]
    away_team = st.selectbox(
        "✈️ Away Team",
        away_teams,
        key="away",
        help="Select the team playing away"
    )

if st.button("🔮 Predict Match", type="primary", use_container_width=True):
    model_data = models[league]
    model = model_data['model']
    le = model_data['label_encoder']
    df_league = model_data['df']
    features = model_data['features']
    
    # Get latest stats for teams
    home_data = df_league[df_league['home_team'] == home_team]
    away_data = df_league[df_league['away_team'] == away_team]
    
    if len(home_data) > 0 and len(away_data) > 0:
        home_stats = home_data.iloc[-1]
        away_stats = away_data.iloc[-1]
        
        # Create feature dict
        feature_dict = {
            'home_recent_goals': home_stats.get('home_recent_goals', 1.5),
            'away_recent_goals': away_stats.get('away_recent_goals', 1.2),
            'home_recent_conceded': home_stats.get('home_recent_conceded', 1.0),
            'away_recent_conceded': away_stats.get('away_recent_conceded', 1.3),
            'home_form': home_stats.get('home_form', 0.5),
            'away_form': away_stats.get('away_form', -0.1),
            'home_advantage': home_stats.get('home_advantage', 0.3),
            'h2h_home_goals': home_stats.get('h2h_home_goals', 1.5),
            'h2h_away_goals': away_stats.get('h2h_away_goals', 1.0)
        }
        
        # Make prediction
        features_df = pd.DataFrame([feature_dict])
        prediction = model.predict(features_df)[0]
        probabilities = model.predict_proba(features_df)[0]
        
        predicted_winner = le.inverse_transform([prediction])[0]
        
        # Display results
        st.markdown("---")
        st.success("### 🎯 Prediction Results")
        
        st.markdown(f"## {home_team} vs {away_team}")
        
        if predicted_winner == 'home':
            st.markdown(f"### ✅ Predicted Winner: **{home_team}** (Home)")
        elif predicted_winner == 'away':
            st.markdown(f"### ✅ Predicted Winner: **{away_team}** (Away)")
        else:
            st.markdown("### 🤝 Predicted Result: **Draw**")
        
        # Probabilities
        st.markdown("### 📊 Probability Breakdown")
        
        prob_home = probabilities[le.transform(['home'])[0]]
        prob_away = probabilities[le.transform(['away'])[0]]
        prob_draw = probabilities[le.transform(['draw'])[0]]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(f"{home_team} Win", f"{prob_home*100:.1f}%")
        with col2:
            st.metric("Draw", f"{prob_draw*100:.1f}%")
        with col3:
            st.metric(f"{away_team} Win", f"{prob_away*100:.1f}%")
        
        # Visual probability bars
        st.markdown("#### Visual Breakdown")
        st.progress(prob_home, text=f"🏠 {home_team}: {prob_home*100:.1f}%")
        st.progress(prob_draw, text=f"🤝 Draw: {prob_draw*100:.1f}%")
        st.progress(prob_away, text=f"✈️ {away_team}: {prob_away*100:.1f}%")
        
        # Show form
        st.markdown("### 📈 Team Form")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                f"{home_team} Recent Form",
                f"{feature_dict['home_form']:.2f}",
                help="Goal difference in last 5 home games"
            )
        
        with col2:
            st.metric(
                f"{away_team} Recent Form",
                f"{feature_dict['away_form']:.2f}",
                help="Goal difference in last 5 away games"
            )
        
        # Show recent matches for both teams
        st.markdown("### 📋 Recent Matches")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### {home_team} - Last 5 Matches")
            home_recent = df_league[
                (df_league['home_team'] == home_team) | 
                (df_league['away_team'] == home_team)
            ].sort_values('Date', ascending=False).head(5)
            
            if len(home_recent) > 0:
                for _, match in home_recent.iterrows():
                    home_score = match['home_goals']
                    away_score = match['away_goals']
                    
                    # Determine if home_team won, lost, or drew
                    if match['home_team'] == home_team:
                        result = "🟢" if home_score > away_score else ("🔴" if home_score < away_score else "🟡")
                        st.markdown(f"{result} **{match['home_team']}** {int(home_score)}-{int(away_score)} {match['away_team']}")
                    else:
                        result = "🟢" if away_score > home_score else ("🔴" if away_score < home_score else "🟡")
                        st.markdown(f"{result} {match['home_team']} {int(home_score)}-{int(away_score)} **{match['away_team']}**")
            else:
                st.info("No recent matches found")
        
        with col2:
            st.markdown(f"#### {away_team} - Last 5 Matches")
            away_recent = df_league[
                (df_league['home_team'] == away_team) | 
                (df_league['away_team'] == away_team)
            ].sort_values('Date', ascending=False).head(5)
            
            if len(away_recent) > 0:
                for _, match in away_recent.iterrows():
                    home_score = match['home_goals']
                    away_score = match['away_goals']
                    
                    if match['home_team'] == away_team:
                        result = "🟢" if home_score > away_score else ("🔴" if home_score < away_score else "🟡")
                        st.markdown(f"{result} **{match['home_team']}** {int(home_score)}-{int(away_score)} {match['away_team']}")
                    else:
                        result = "🟢" if away_score > home_score else ("🔴" if away_score < home_score else "🟡")
                        st.markdown(f"{result} {match['home_team']} {int(home_score)}-{int(away_score)} **{match['away_team']}**")
            else:
                st.info("No recent matches found")
        
        # Show head-to-head history
        st.markdown("### ⚔️ Head-to-Head History")
        h2h = df_league[
            ((df_league['home_team'] == home_team) & (df_league['away_team'] == away_team)) |
            ((df_league['home_team'] == away_team) & (df_league['away_team'] == home_team))
        ].sort_values('Date', ascending=False).head(5)
        
        if len(h2h) > 0:
            st.markdown(f"Last {len(h2h)} meetings:")
            for _, match in h2h.iterrows():
                home_score = int(match['home_goals'])
                away_score = int(match['away_goals'])
                date = match['Date'].strftime('%Y-%m-%d')
                
                # Highlight winner
                if home_score > away_score:
                    st.markdown(f"🏆 **{match['home_team']}** {home_score}-{away_score} {match['away_team']} *({date})*")
                elif away_score > home_score:
                    st.markdown(f"{match['home_team']} {home_score}-{away_score} **{match['away_team']}** 🏆 *({date})*")
                else:
                    st.markdown(f"{match['home_team']} {home_score}-{away_score} {match['away_team']} 🤝 *({date})*")
        else:
            st.info("No previous meetings found between these teams")
        
    else:
        st.error("❌ Not enough historical data for these teams. Try different teams.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    Built with Streamlit & Scikit-Learn • Data from football-data.co.uk<br>
    Model: Random Forest with 9 engineered features
</div>
""", unsafe_allow_html=True)