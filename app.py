# app.py - Premier League Match Predictor with Goal Predictions (PART 1)

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from datetime import datetime, timedelta
from database import MatchDatabase
from goal_predictor import GoalPredictor
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="EPL Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #38003c;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stat-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    .score-box {
        text-align: center;
        padding: 1rem;
        background: #f0f2f6;
        border-radius: 10px;
        margin: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = MatchDatabase()

if 'model' not in st.session_state:
    try:
        model = xgb.Booster()
        model.load_model('epl_xgboost_model.json')
        st.session_state.model = model

        with open('model_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        st.session_state.artifacts = artifacts
        st.session_state.le = artifacts['label_encoder']
        st.session_state.feature_columns = artifacts['feature_columns']
    except Exception as e:
        st.error(f"Error loading outcome prediction model: {e}")
        st.stop()

if 'goal_predictor' not in st.session_state:
    try:
        goal_predictor = GoalPredictor.load('goal_predictor_model.pkl')
        st.session_state.goal_predictor = goal_predictor
        print("✓ Goal predictor loaded")
    except Exception as e:
        print(f"  Goal predictor not available: {e}")
        st.session_state.goal_predictor = None

db = st.session_state.db
model = st.session_state.model
le = st.session_state.le
feature_columns = st.session_state.feature_columns
goal_predictor = st.session_state.goal_predictor


# Helper Functions
def calculate_team_features(team, is_home, ref_date):
    """Calculate features for a team before a specific date"""

    # Get team's previous matches
    team_matches = db.get_team_matches(team, before_date=ref_date.strftime('%Y-%m-%d'))

    if len(team_matches) < 3:
        return None

    # Last 5 matches
    last_5 = team_matches.head(5)

    goals_scored_5 = 0
    goals_conceded_5 = 0
    points_5 = 0
    wins_5 = 0

    for _, match in last_5.iterrows():
        if match['home_team'] == team:
            goals_scored_5 += match['home_goals']
            goals_conceded_5 += match['away_goals']
            if match['result'] == 'H':
                points_5 += 3
                wins_5 += 1
            elif match['result'] == 'D':
                points_5 += 1
        else:
            goals_scored_5 += match['away_goals']
            goals_conceded_5 += match['home_goals']
            if match['result'] == 'A':
                points_5 += 3
                wins_5 += 1
            elif match['result'] == 'D':
                points_5 += 1

    # Overall stats
    total_goals_scored = 0
    total_goals_conceded = 0
    total_points = 0
    total_wins = 0

    for _, match in team_matches.iterrows():
        if match['home_team'] == team:
            total_goals_scored += match['home_goals']
            total_goals_conceded += match['away_goals']
            if match['result'] == 'H':
                total_points += 3
                total_wins += 1
            elif match['result'] == 'D':
                total_points += 1
        else:
            total_goals_scored += match['away_goals']
            total_goals_conceded += match['home_goals']
            if match['result'] == 'A':
                total_points += 3
                total_wins += 1
            elif match['result'] == 'D':
                total_points += 1

    matches_played = len(team_matches)

    return {
        'matches_played': matches_played,
        'points_last_5': points_5,
        'goals_scored_last_5': goals_scored_5,
        'goals_conceded_last_5': goals_conceded_5,
        'wins_last_5': wins_5,
        'avg_goals_scored': total_goals_scored / matches_played,
        'avg_goals_conceded': total_goals_conceded / matches_played,
        'win_rate': total_wins / matches_played,
        'points_per_game': total_points / matches_played,
    }


def calculate_h2h_features(home_team, away_team, ref_date):
    """Calculate head-to-head features"""

    h2h_matches = db.get_h2h_matches(home_team, away_team,
                                     before_date=ref_date.strftime('%Y-%m-%d'),
                                     limit=5)

    if len(h2h_matches) == 0:
        return {
            'H2H_matches': 0,
            'H2H_home_wins': 0,
            'H2H_draws': 0,
            'H2H_away_wins': 0,
            'H2H_home_goals': 0,
            'H2H_away_goals': 0,
        }

    home_wins = 0
    draws = 0
    away_wins = 0
    home_goals = 0
    away_goals = 0

    for _, match in h2h_matches.iterrows():
        if match['home_team'] == home_team:
            home_goals += match['home_goals']
            away_goals += match['away_goals']
            if match['result'] == 'H':
                home_wins += 1
            elif match['result'] == 'D':
                draws += 1
            else:
                away_wins += 1
        else:
            home_goals += match['away_goals']
            away_goals += match['home_goals']
            if match['result'] == 'A':
                home_wins += 1
            elif match['result'] == 'D':
                draws += 1
            else:
                away_wins += 1

    return {
        'H2H_matches': len(h2h_matches),
        'H2H_home_wins': home_wins,
        'H2H_draws': draws,
        'H2H_away_wins': away_wins,
        'H2H_home_goals': home_goals / len(h2h_matches),
        'H2H_away_goals': away_goals / len(h2h_matches),
    }


def prepare_match_features(home_team, away_team, match_date=None):
    """Prepare features for prediction"""

    if match_date is None:
        match_date = datetime.now()

    # Calculate features for both teams
    home_features = calculate_team_features(home_team, True, match_date)
    away_features = calculate_team_features(away_team, False, match_date)

    if home_features is None or away_features is None:
        return None

    # Calculate H2H
    h2h_features = calculate_h2h_features(home_team, away_team, match_date)

    # Build feature dictionary
    features = {}

    # Home team features
    for key, value in home_features.items():
        features[f'Home_{key}'] = value

    # Away team features
    for key, value in away_features.items():
        features[f'Away_{key}'] = value

    # H2H features
    features.update(h2h_features)

    # Comparative features
    features['Form_Difference'] = home_features['points_last_5'] - away_features['points_last_5']
    features['GoalScoring_Difference'] = home_features['goals_scored_last_5'] - away_features['goals_scored_last_5']
    features['Defense_Difference'] = away_features['goals_conceded_last_5'] - home_features['goals_conceded_last_5']
    features['WinRate_Difference'] = home_features['win_rate'] - away_features['win_rate']
    features['PPG_Difference'] = home_features['points_per_game'] - away_features['points_per_game']

    # Time features
    features['Month'] = match_date.month
    features['DayOfWeek'] = match_date.dayofweek
    features['MatchNumber'] = 20  # Mid-season estimate

    # Add missing features with 0
    for col in feature_columns:
        if col not in features:
            features[col] = 0

    return features, home_features, away_features, h2h_features


def make_prediction(features_dict):
    """Make prediction using the model"""

    # Create DataFrame with correct column order
    features_df = pd.DataFrame([features_dict])[feature_columns]

    # Create DMatrix and predict
    dmatrix = xgb.DMatrix(features_df)
    predictions = model.predict(dmatrix)

    predicted_class = le.inverse_transform([np.argmax(predictions[0])])[0]

    return {
        'predicted_result': predicted_class,
        'prob_away': float(predictions[0][0]),
        'prob_draw': float(predictions[0][1]),
        'prob_home': float(predictions[0][2]),
        'confidence': float(predictions[0].max())
    }


# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/f/f2/Premier_League_Logo.svg", width=200)

    page = st.radio("Navigation",
                    ["🎯 Match Predictor",
                     "➕ Add Match Results",
                     "📊 Prediction History",
                     "📈 Team Analysis",
                     "ℹ️ Model Info"])

    st.markdown("---")

    # Quick stats
    all_matches = db.get_all_matches()
    st.metric("Total Matches", len(all_matches))

    if len(all_matches) > 0:
        st.metric("Latest Match", all_matches['date'].max().strftime('%Y-%m-%d'))

    predictions = db.get_predictions()
    st.metric("Total Predictions", len(predictions))

    if len(predictions) > 0:
        accuracy_stats = db.get_prediction_accuracy()
        if accuracy_stats['total'] > 0:
            st.metric("Prediction Accuracy", f"{accuracy_stats['accuracy'] * 100:.1f}%")

    # Show goal predictor status
    if goal_predictor is not None:
        st.success("⚽ Goal Prediction: Active")
    else:
        st.warning("⚽ Goal Prediction: Not Available")

# app.py - PART 2: Match Predictor Page

# Main Content
if page == "🎯 Match Predictor":
    st.markdown('<div class="main-header">⚽ Premier League Match Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict match outcomes and exact scores using advanced machine learning</div>',
                unsafe_allow_html=True)

    # Get all teams
    teams = sorted(db.get_all_teams())

    if len(teams) == 0:
        st.error("No teams found in database. Please add match data first.")
        st.stop()

    # Match selection
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        home_team = st.selectbox("🏠 Home Team", teams, key='home_select')

    with col2:
        st.markdown("<div style='text-align: center; padding-top: 1.8rem; font-size: 1.5rem;'>VS</div>",
                    unsafe_allow_html=True)

    with col3:
        away_team = st.selectbox("✈️ Away Team", teams, key='away_select')

    # Match date
    match_date = st.date_input("📅 Match Date", datetime.now() + timedelta(days=7))

    if st.button("🔮 Predict Match Outcome", type="primary", use_container_width=True):
        if home_team == away_team:
            st.error("Please select different teams!")
        else:
            with st.spinner("Analyzing team form and calculating prediction..."):
                # Prepare features
                result = prepare_match_features(home_team, away_team, pd.Timestamp(match_date))

                if result is None:
                    st.error("Insufficient historical data for one or both teams. Need at least 3 matches.")
                else:
                    features, home_form, away_form, h2h = result

                    # Make outcome prediction
                    prediction = make_prediction(features)

                    # Make goal prediction if available
                    goal_prediction = None
                    if goal_predictor is not None:
                        try:
                            features_df = pd.DataFrame([features])[feature_columns]
                            home_goals_pred, away_goals_pred = goal_predictor.predict(features_df)
                            score_details = goal_predictor.predict_score(features_df)[0]

                            goal_prediction = {
                                'home_goals': int(round(home_goals_pred[0])),
                                'away_goals': int(round(away_goals_pred[0])),
                                'home_expected': float(home_goals_pred[0]),
                                'away_expected': float(away_goals_pred[0]),
                                'top_scores': score_details['most_likely_scores']
                            }
                        except Exception as e:
                            print(f"Goal prediction error: {e}")
                            goal_prediction = None

                    # Display prediction
                    st.markdown("---")

                    # ===== MAIN PREDICTION BOX WITH SCORE =====
                    result_emoji = {"H": "🏠", "D": "🤝", "A": "✈️"}
                    result_text = {"H": f"{home_team} Win", "D": "Draw", "A": f"{away_team} Win"}

                    if goal_prediction:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h1>{result_emoji[prediction['predicted_result']]} {result_text[prediction['predicted_result']]}</h1>
                            <h2 style="margin-top: 1rem; font-size: 2.5rem;">
                                {home_team} {goal_prediction['home_goals']} - {goal_prediction['away_goals']} {away_team}
                            </h2>
                            <h3>Confidence: {prediction['confidence'] * 100:.1f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h1>{result_emoji[prediction['predicted_result']]} {result_text[prediction['predicted_result']]}</h1>
                            <h3>Confidence: {prediction['confidence'] * 100:.1f}%</h3>
                        </div>
                        """, unsafe_allow_html=True)

                    # ===== GOAL PREDICTION DETAILS =====
                    if goal_prediction:
                        st.subheader("⚽ Score Prediction Details")

                        col1, col2, col3 = st.columns([1, 2, 1])

                        with col1:
                            st.metric(
                                label=f"🏠 {home_team}",
                                value=f"{goal_prediction['home_goals']} goals",
                                delta=f"xG: {goal_prediction['home_expected']:.2f}"
                            )

                        with col2:
                            st.markdown(f"""
                            <div class="score-box">
                                <h1 style='color: #38003c; font-size: 4rem; margin: 0;'>
                                    {goal_prediction['home_goals']} - {goal_prediction['away_goals']}
                                </h1>
                                <p style='color: #666; margin-top: 0.5rem;'>Most Likely Score</p>
                            </div>
                            """, unsafe_allow_html=True)

                        with col3:
                            st.metric(
                                label=f"✈️ {away_team}",
                                value=f"{goal_prediction['away_goals']} goals",
                                delta=f"xG: {goal_prediction['away_expected']:.2f}"
                            )

                        # Top 5 most likely scores
                        st.markdown("### 📊 Other Likely Scorelines")

                        cols = st.columns(5)
                        for i, (h, a, prob) in enumerate(goal_prediction['top_scores'][:5]):
                            with cols[i]:
                                result_indicator = "🏠" if h > a else ("✈️" if a > h else "🤝")
                                st.markdown(f"""
                                <div style='text-align: center; padding: 0.8rem; background: #f8f9fa; border-radius: 8px; border: 2px solid #e0e0e0;'>
                                    <div style='font-size: 1.8rem; font-weight: bold; color: #38003c;'>{h}-{a}</div>
                                    <div style='font-size: 1rem; color: #666; margin: 0.3rem 0;'>{prob * 100:.1f}%</div>
                                    <div style='font-size: 1.2rem;'>{result_indicator}</div>
                                </div>
                                """, unsafe_allow_html=True)

                        # Expected Goals Chart
                        st.markdown("### 📈 Expected Goals (xG)")

                        fig_xg = go.Figure()

                        fig_xg.add_trace(go.Bar(
                            x=[home_team, away_team],
                            y=[goal_prediction['home_expected'], goal_prediction['away_expected']],
                            marker_color=['#38003c', '#e90052'],
                            text=[f"{goal_prediction['home_expected']:.2f}",
                                  f"{goal_prediction['away_expected']:.2f}"],
                            textposition='auto',
                            textfont=dict(size=16, color='white'),
                        ))

                        fig_xg.update_layout(
                            showlegend=False,
                            height=300,
                            yaxis_title="Expected Goals",
                            template="plotly_white",
                            yaxis=dict(range=[0, max(goal_prediction['home_expected'],
                                                     goal_prediction['away_expected']) + 0.5])
                        )

                        st.plotly_chart(fig_xg, use_container_width=True)

                    # ===== OUTCOME PROBABILITIES =====
                    st.subheader("📊 Match Outcome Probabilities")

                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        name='Probabilities',
                        x=[f'{home_team} Win', 'Draw', f'{away_team} Win'],
                        y=[prediction['prob_home'] * 100, prediction['prob_draw'] * 100, prediction['prob_away'] * 100],
                        marker_color=['#38003c', '#00ff87', '#e90052'],
                        text=[f"{prediction['prob_home'] * 100:.1f}%",
                              f"{prediction['prob_draw'] * 100:.1f}%",
                              f"{prediction['prob_away'] * 100:.1f}%"],
                        textposition='auto',
                        textfont=dict(size=14, color='white'),
                    ))

                    fig.update_layout(
                        showlegend=False,
                        height=400,
                        yaxis_title="Probability (%)",
                        yaxis_range=[0, 100],
                        template="plotly_white"
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # ===== TEAM FORM COMPARISON =====
                    st.subheader("📈 Team Form Comparison (Last 5 Matches)")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"### 🏠 {home_team}")
                        st.markdown(f"""
                        <div class="stat-box">
                            <b>Points:</b> {home_form['points_last_5']}/15<br>
                            <b>Goals Scored:</b> {home_form['goals_scored_last_5']}<br>
                            <b>Goals Conceded:</b> {home_form['goals_conceded_last_5']}<br>
                            <b>Wins:</b> {home_form['wins_last_5']}/5<br>
                            <b>Avg Goals/Game:</b> {home_form['avg_goals_scored']:.2f}<br>
                            <b>Points Per Game:</b> {home_form['points_per_game']:.2f}
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"### ✈️ {away_team}")
                        st.markdown(f"""
                        <div class="stat-box">
                            <b>Points:</b> {away_form['points_last_5']}/15<br>
                            <b>Goals Scored:</b> {away_form['goals_scored_last_5']}<br>
                            <b>Goals Conceded:</b> {away_form['goals_conceded_last_5']}<br>
                            <b>Wins:</b> {away_form['wins_last_5']}/5<br>
                            <b>Avg Goals/Game:</b> {away_form['avg_goals_scored']:.2f}<br>
                            <b>Points Per Game:</b> {away_form['points_per_game']:.2f}
                        </div>
                        """, unsafe_allow_html=True)

                    # ===== H2H HISTORY =====
                    if h2h['H2H_matches'] > 0:
                        st.subheader(f"🤼 Head-to-Head (Last {h2h['H2H_matches']} matches)")

                        col1, col2, col3 = st.columns(3)
                        col1.metric(f"{home_team} Wins", h2h['H2H_home_wins'])
                        col2.metric("Draws", h2h['H2H_draws'])
                        col3.metric(f"{away_team} Wins", h2h['H2H_away_wins'])

                        st.markdown(f"""
                        <div style='text-align: center; margin-top: 1rem; padding: 1rem; background: #f0f2f6; border-radius: 5px;'>
                            <p style='color: #666; margin: 0;'>Average Goals in Head-to-Head</p>
                            <p style='font-size: 1.5rem; font-weight: bold; margin: 0.5rem 0;'>
                                {home_team} {h2h['H2H_home_goals']:.1f} - {h2h['H2H_away_goals']:.1f} {away_team}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # ===== SAVE PREDICTION BUTTON =====
                    st.markdown("---")
                    if st.button("💾 Save This Prediction", use_container_width=True):
                        prediction_data = {
                            'match_date': match_date.strftime('%Y-%m-%d'),
                            'home_team': home_team,
                            'away_team': away_team,
                            'predicted_result': prediction['predicted_result'],
                            'prob_home': prediction['prob_home'],
                            'prob_draw': prediction['prob_draw'],
                            'prob_away': prediction['prob_away'],
                            'confidence': prediction['confidence']
                        }

                        # Add goal prediction if available
                        if goal_prediction:
                            prediction_data['predicted_home_goals'] = goal_prediction['home_goals']
                            prediction_data['predicted_away_goals'] = goal_prediction['away_goals']

                        db.save_prediction(prediction_data)
                        st.success("✅ Prediction saved!")
                        st.rerun()

elif page == "➕ Add Match Results":
    st.title("➕ Add Match Results")
    st.markdown("Update the database with the latest match results")

    teams = sorted(db.get_all_teams())

    with st.form("add_match_form"):
        st.subheader("Match Details")

        col1, col2 = st.columns(2)

        with col1:
            match_date = st.date_input("Match Date", datetime.now())
            home_team = st.selectbox("Home Team", teams)
            home_goals = st.number_input("Home Goals", min_value=0, max_value=20, value=0)

        with col2:
            current_season = f"{datetime.now().year}/{datetime.now().year + 1}"
            season = st.text_input("Season", value=current_season)
            away_team = st.selectbox("Away Team", teams)
            away_goals = st.number_input("Away Goals", min_value=0, max_value=20, value=0)

        # Determine result
        if home_goals > away_goals:
            result = 'H'
        elif home_goals < away_goals:
            result = 'A'
        else:
            result = 'D'

        st.info(f"Result: {result} ({home_team} {home_goals}-{away_goals} {away_team})")

        # Optional stats
        with st.expander("📊 Additional Match Statistics (Optional)"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Home Team**")
                home_shots = st.number_input("Shots", min_value=0, value=0, key='hs')
                home_sot = st.number_input("Shots on Target", min_value=0, value=0, key='hst')
                home_corners = st.number_input("Corners", min_value=0, value=0, key='hc')
                home_fouls = st.number_input("Fouls", min_value=0, value=0, key='hf')
                home_yellows = st.number_input("Yellow Cards", min_value=0, value=0, key='hy')
                home_reds = st.number_input("Red Cards", min_value=0, value=0, key='hr')

            with col2:
                st.markdown("**Away Team**")
                away_shots = st.number_input("Shots", min_value=0, value=0, key='as')
                away_sot = st.number_input("Shots on Target", min_value=0, value=0, key='ast')
                away_corners = st.number_input("Corners", min_value=0, value=0, key='ac')
                away_fouls = st.number_input("Fouls", min_value=0, value=0, key='af')
                away_yellows = st.number_input("Yellow Cards", min_value=0, value=0, key='ay')
                away_reds = st.number_input("Red Cards", min_value=0, value=0, key='ar')

        submitted = st.form_submit_button("Add Match", type="primary", use_container_width=True)

        if submitted:
            if home_team == away_team:
                st.error("Home and Away teams must be different!")
            else:
                match_data = {
                    'date': match_date.strftime('%Y-%m-%d'),
                    'season': season,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_goals': int(home_goals),
                    'away_goals': int(away_goals),
                    'result': result,
                    'home_shots': int(home_shots),
                    'away_shots': int(away_shots),
                    'home_shots_target': int(home_sot),
                    'away_shots_target': int(away_sot),
                    'home_corners': int(home_corners),
                    'away_corners': int(away_corners),
                    'home_fouls': int(home_fouls),
                    'away_fouls': int(away_fouls),
                    'home_yellows': int(home_yellows),
                    'away_yellows': int(away_yellows),
                    'home_reds': int(home_reds),
                    'away_reds': int(away_reds),
                }

                db.add_match(match_data)
                st.success(f"✅ Match added: {home_team} {home_goals}-{away_goals} {away_team}")
                st.rerun()

    # Show recent matches
    st.markdown("---")
    st.subheader("📋 Recent Matches (Last 10)")

    recent_matches = db.get_all_matches().tail(10).sort_values('date', ascending=False)

    if len(recent_matches) > 0:
        for _, match in recent_matches.iterrows():
            col1, col2, col3 = st.columns([1, 3, 1])

            with col1:
                st.text(match['date'].strftime('%Y-%m-%d'))

            with col2:
                result_color = {'H': '🟢', 'D': '🟡', 'A': '🔴'}
                st.text(
                    f"{match['home_team']} {match['home_goals']}-{match['away_goals']} {match['away_team']} {result_color[match['result']]}")

            with col3:
                st.text(match['season'])

# app.py - PART 3: Prediction History, Team Analysis, Model Info

elif page == "📊 Prediction History":
    st.title("📊 Prediction History")

    predictions = db.get_predictions()

    if len(predictions) == 0:
        st.info("No predictions yet. Make your first prediction!")
    else:
        # Overall stats
        accuracy_stats = db.get_prediction_accuracy()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", accuracy_stats['total'])
        col2.metric("Correct Predictions", accuracy_stats['correct'])

        if accuracy_stats['total'] > 0:
            col3.metric("Outcome Accuracy", f"{accuracy_stats['accuracy'] * 100:.1f}%")

        # Score prediction accuracy (if available)
        if 'predicted_home_goals' in predictions.columns:
            score_predictions = predictions[predictions['actual_home_goals'].notna()]
            if len(score_predictions) > 0:
                exact_scores = score_predictions[
                    (score_predictions['predicted_home_goals'] == score_predictions['actual_home_goals']) &
                    (score_predictions['predicted_away_goals'] == score_predictions['actual_away_goals'])
                    ]
                score_acc = len(exact_scores) / len(score_predictions) * 100
                col4.metric("Exact Score Accuracy", f"{score_acc:.1f}%")

        st.markdown("---")

        # Display predictions
        st.subheader("Recent Predictions")

        for _, pred in predictions.head(20).iterrows():
            with st.expander(f"{pred['match_date']} | {pred['home_team']} vs {pred['away_team']}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Predicted Result:** {pred['predicted_result']}")

                    # Show predicted score if available
                    if pd.notna(pred.get('predicted_home_goals')):
                        pred_score = f"{int(pred['predicted_home_goals'])}-{int(pred['predicted_away_goals'])}"
                        st.markdown(f"**Predicted Score:** {pred_score}")

                    st.markdown(f"**Confidence:** {pred['confidence'] * 100:.1f}%")
                    st.markdown(f"**Made on:** {pred['prediction_date']}")

                with col2:
                    if pred['actual_result']:
                        correct_emoji = "✅" if pred['correct'] == 1 else "❌"
                        st.markdown(f"**Actual Result:** {pred['actual_result']} {correct_emoji}")

                        # Show actual score if available
                        if pd.notna(pred.get('actual_home_goals')):
                            actual_score = f"{int(pred['actual_home_goals'])}-{int(pred['actual_away_goals'])}"
                            st.markdown(f"**Actual Score:** {actual_score}")

                            # Check if score was correct
                            if (pred.get('predicted_home_goals') == pred.get('actual_home_goals') and
                                    pred.get('predicted_away_goals') == pred.get('actual_away_goals')):
                                st.markdown("### ⚽ **Exact score predicted!** ⚽")
                            else:
                                # Calculate goal difference
                                home_diff = abs(pred.get('predicted_home_goals', 0) - pred.get('actual_home_goals', 0))
                                away_diff = abs(pred.get('predicted_away_goals', 0) - pred.get('actual_away_goals', 0))
                                total_diff = home_diff + away_diff
                                st.markdown(f"Goal difference: {total_diff} goals off")
                    else:
                        st.markdown("**Actual Result:** _Not yet played_")

                    st.markdown(f"**Probabilities:**")
                    st.markdown(f"- Home: {pred['prob_home'] * 100:.1f}%")
                    st.markdown(f"- Draw: {pred['prob_draw'] * 100:.1f}%")
                    st.markdown(f"- Away: {pred['prob_away'] * 100:.1f}%")

elif page == "📈 Team Analysis":
    st.title("📈 Team Analysis")

    teams = sorted(db.get_all_teams())
    selected_team = st.selectbox("Select Team", teams)

    if selected_team:
        team_matches = db.get_team_matches(selected_team)

        if len(team_matches) > 0:
            # Overall stats
            st.subheader(f"📊 {selected_team} - Overall Statistics")

            total_matches = len(team_matches)
            home_matches = team_matches[team_matches['home_team'] == selected_team]
            away_matches = team_matches[team_matches['away_team'] == selected_team]

            # Calculate stats
            wins = 0
            draws = 0
            losses = 0
            goals_for = 0
            goals_against = 0

            for _, match in team_matches.iterrows():
                if match['home_team'] == selected_team:
                    goals_for += match['home_goals']
                    goals_against += match['away_goals']
                    if match['result'] == 'H':
                        wins += 1
                    elif match['result'] == 'D':
                        draws += 1
                    else:
                        losses += 1
                else:
                    goals_for += match['away_goals']
                    goals_against += match['home_goals']
                    if match['result'] == 'A':
                        wins += 1
                    elif match['result'] == 'D':
                        draws += 1
                    else:
                        losses += 1

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Matches Played", total_matches)
            col2.metric("Wins", wins)
            col3.metric("Draws", draws)
            col4.metric("Losses", losses)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Goals For", goals_for)
            col2.metric("Goals Against", goals_against)
            col3.metric("Goal Difference", goals_for - goals_against)
            col4.metric("Win Rate", f"{(wins / total_matches) * 100:.1f}%")

            # Goals per game
            col1, col2 = st.columns(2)
            col1.metric("Goals Per Game", f"{goals_for / total_matches:.2f}")
            col2.metric("Goals Conceded Per Game", f"{goals_against / total_matches:.2f}")

            # Recent form
            st.subheader("📅 Recent Matches (Last 10)")

            for _, match in team_matches.head(10).iterrows():
                is_home = match['home_team'] == selected_team

                if is_home:
                    opponent = match['away_team']
                    score = f"{match['home_goals']}-{match['away_goals']}"
                    result = match['result']
                else:
                    opponent = match['home_team']
                    score = f"{match['away_goals']}-{match['home_goals']}"
                    result = 'H' if match['result'] == 'A' else ('A' if match['result'] == 'H' else 'D')

                result_color = {'H': '🟢 W', 'D': '🟡 D', 'A': '🔴 L'}
                venue = "🏠" if is_home else "✈️"

                st.text(
                    f"{match['date'].strftime('%Y-%m-%d')} | {venue} vs {opponent} | {score} | {result_color[result]}")

            # Form chart
            st.subheader("📈 Form Chart (Last 15 Matches)")

            last_15 = team_matches.head(15)
            form_data = []

            for _, match in last_15.iterrows():
                if match['home_team'] == selected_team:
                    if match['result'] == 'H':
                        points = 3
                    elif match['result'] == 'D':
                        points = 1
                    else:
                        points = 0
                else:
                    if match['result'] == 'A':
                        points = 3
                    elif match['result'] == 'D':
                        points = 1
                    else:
                        points = 0

                form_data.append({
                    'Date': match['date'],
                    'Points': points
                })

            form_df = pd.DataFrame(form_data).sort_values('Date')

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=form_df['Date'],
                y=form_df['Points'],
                mode='lines+markers',
                name='Points',
                line=dict(color='#38003c', width=3),
                marker=dict(size=10)
            ))

            fig.update_layout(
                title=f"{selected_team} - Points Per Match (Last 15)",
                xaxis_title="Date",
                yaxis_title="Points",
                template="plotly_white",
                height=400,
                yaxis=dict(range=[-0.5, 3.5], tickmode='linear', tick0=0, dtick=1)
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No matches found for this team")

elif page == "ℹ️ Model Info":
    st.title("ℹ️ Model Information")

    # Model performance
    st.subheader("📊 Model Performance")

    artifacts = st.session_state.artifacts

    col1, col2, col3 = st.columns(3)
    col1.metric("Outcome Accuracy", f"{artifacts['test_accuracy'] * 100:.2f}%")
    col2.metric("Log Loss", f"{artifacts['test_logloss']:.4f}")
    col3.metric("Model Version", artifacts['model_version'])

    # Goal prediction stats if available
    if 'goal_mae_home' in artifacts:
        st.markdown("---")
        st.subheader("⚽ Goal Prediction Performance")

        col1, col2, col3 = st.columns(3)
        col1.metric("Home Goals MAE", f"{artifacts['goal_mae_home']:.3f}")
        col2.metric("Away Goals MAE", f"{artifacts['goal_mae_away']:.3f}")
        col3.metric("Exact Score Accuracy", f"{artifacts['exact_score_accuracy'] * 100:.1f}%")

        st.info("MAE (Mean Absolute Error) measures average goal prediction error. Lower is better!")

    st.markdown("---")

    # Training info
    st.subheader("🎓 Training Information")
    st.markdown(f"**Training Date:** {artifacts['training_date']}")
    st.markdown(f"**Training Period:** {artifacts['train_date_range']}")
    st.markdown(f"**Test Period:** {artifacts['test_date_range']}")
    st.markdown(f"**Best Iteration:** {artifacts['best_iteration']}")

    st.markdown("---")

    # Features
    st.subheader("🔧 Model Features")
    st.markdown(f"**Total Features:** {len(artifacts['feature_columns'])}")

    with st.expander("View All Features"):
        for i, feat in enumerate(artifacts['feature_columns'], 1):
            st.text(f"{i}. {feat}")

    st.markdown("---")

    # Top features
    st.subheader("⭐ Top 10 Most Important Features")

    if 'top_10_features' in artifacts:
        for i, feat in enumerate(artifacts['top_10_features'], 1):
            st.text(f"{i}. {feat}")

    st.markdown("---")

    # Best parameters
    st.subheader("⚙️ Model Hyperparameters")

    params = artifacts['best_params']

    col1, col2 = st.columns(2)

    with col1:
        for key in list(params.keys())[:len(params) // 2]:
            st.text(f"{key}: {params[key]}")

    with col2:
        for key in list(params.keys())[len(params) // 2:]:
            st.text(f"{key}: {params[key]}")

    st.markdown("---")

    # Usage notes
    st.subheader("📝 Usage Notes")
    st.markdown("""
    ### Outcome Prediction Model
    - **Model Type:** XGBoost Multi-class Classification
    - **Predictions:** Home Win (H), Draw (D), Away Win (A)
    - **Training Data:** 2013/14 - 2024/25 seasons
    - **Best Use:** Understanding match dynamics and probabilities

    ### Goal Prediction Model
    - **Model Type:** XGBoost Poisson Regression
    - **Predictions:** Exact number of goals for each team
    - **Output:** Most likely score + top 5 scorelines with probabilities
    - **Expected Goals (xG):** Average predicted goals per team

    ### Limitations
    - Cannot account for injuries, suspensions, or team motivation
    - Based on historical patterns - unusual situations may not be captured
    - Weather, referee decisions, and red cards not included
    - Manager changes and tactical shifts take time to reflect

    ### Update Frequency
    - Recommended: Update match results weekly (every Monday)
    - Model retraining: Every 3-6 months for optimal performance

    ### Accuracy Expectations
    - **Outcome Prediction:** 55-60% is excellent for football (inherently unpredictable)
    - **Exact Score:** 15-20% is very good (most bookmakers achieve 10-15%)
    - **Goal MAE:** 1.2-1.4 goals is strong performance

    ### How to Interpret Predictions
    - **High Confidence (>60%):** Model is very sure about the outcome
    - **Medium Confidence (45-60%):** Competitive match, could go either way
    - **Low Confidence (<45%):** Very uncertain, consider draw

    ### Best Practices
    1. Use predictions as **one input** among many
    2. Consider current news (injuries, form, motivation)
    3. Home advantage matters - models factor this in
    4. Recent form (last 5 matches) is heavily weighted
    5. Head-to-head history provides valuable context
    """)

    st.markdown("---")

    st.subheader("🔮 Future Enhancements")
    st.markdown("""
    Potential improvements for future versions:
    - **xG Data Integration:** Use expected goals from matches
    - **Player-Level Data:** Account for key player availability
    - **Weather Data:** Include weather conditions
    - **Betting Odds:** Use market odds as additional features
    - **Transfer Impact:** Model impact of new signings
    - **Tactical Analysis:** Formation and playing style features
    - **Deep Learning:** LSTM models for time-series patterns
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>⚽ Premier League Match Predictor | Powered by XGBoost & Streamlit</p>
    <p style='font-size: 0.9rem;'>
        Outcome Accuracy: 57% | Goal Prediction MAE: ~1.3 | Exact Score: ~18%
    </p>
    <p style='font-size: 0.8rem; margin-top: 0.5rem;'>
        For educational purposes only. Past performance does not guarantee future results.
    </p>
    <p style='font-size: 0.7rem; margin-top: 0.5rem;'>
        Model trained on historical data from 2013-2025 | Update database weekly for best results
    </p>
</div>
""", unsafe_allow_html=True)