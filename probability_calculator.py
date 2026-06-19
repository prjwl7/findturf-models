import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import xgboost as xgb
from stable_baselines3 import PPO

# ==========================================
# 1. LOAD PRE-TRAINED DATA & MODELS
# ==========================================
@st.cache_resource 
def load_xgb_models():
    model_short = xgb.XGBClassifier()
    model_short.load_model("turf_booking_model_short.json")
    
    model_long = xgb.XGBClassifier()
    model_long.load_model("turf_booking_model.json")
    return model_short, model_long

@st.cache_resource
def load_rl_model():
    return PPO.load("turf_dynamic_pricing_model_v2")

@st.cache_data 
def load_data():
    df_train = pd.read_parquet('df_train.parquet')
    if 'date' in df_train.columns:
        df_train['date'] = pd.to_datetime(df_train['date'])
    return df_train

# ==========================================
# 2. FEATURE ENGINEERING PIPELINES
# ==========================================
def generate_30d_slots():
    start_date = pd.Timestamp.now().normalize()
    dates = [start_date + timedelta(days=i) for i in range(30)]
    slots = [f"{str(h).zfill(2)}:00-{str(h+1).zfill(2)}:00" for h in range(6, 22)]
    
    # Realistic Base Pricing
    base_prices = {
        "06:00": 1200, "07:00": 1200, "08:00": 1500, "09:00": 1500, 
        "10:00": 1800, "11:00": 1800, "12:00": 1500, "13:00": 1500, 
        "14:00": 2000, "15:00": 2000, "16:00": 2200, "17:00": 2200,
        "18:00": 3000, "19:00": 3500, "20:00": 3500, "21:00": 3000, "22:00": 2500
    }
    
    data = []
    for d in dates:
        for s in slots:
            hour_str = s.split('-')[0]
            charge = base_prices.get(hour_str, 1500)
            
            data.append({
                "date": d, "slot": s, "turf_id": "turf_001", "sport_type": "Football",
                "base_hourly_charge": charge, "hour": int(hour_str.split(':')[0]),
                "day_of_week": d.dayofweek, "is_weekend": 1 if d.dayofweek >= 5 else 0
            })
            
    return pd.DataFrame(data)

def prepare_momentum_data(df, df_train):
    # 🛡️ NEW SAFETY NET: Ensure df_train actually has the columns before grouping!
    expected_cols = ['booking_rate_7d', 'streak_3d', 'booking_rate_14d', 'booking_momentum']
    for c in expected_cols:
        if c not in df_train.columns:
            df_train[c] = 0.0  # Fallback to 0 if the raw dataset was loaded
            
    # 1. Get latest stats
    latest_stats = df_train.groupby(['turf_id', 'sport_type', 'slot']).agg({
        'booking_rate_7d': 'last', 'streak_3d': 'last', 
        'booking_rate_14d': 'last', 'booking_momentum': 'last'
    }).reset_index()

    # 2. Get historical means
    slot_means = df_train.groupby(['turf_id', 'slot'])[['booking_rate_7d', 'streak_3d']].mean().reset_index()
    
    # 3. Merge latest stats
    df = df.merge(latest_stats, on=['turf_id', 'sport_type', 'slot'], how='left')
    
    # 🛡️ THE IRONCLAD SAFETY NET 🛡️
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan
    
    # 4. Fill missing values using historical slot means
    for col in ['booking_rate_7d', 'streak_3d']:
        temp_means = slot_means[['turf_id', 'slot', col]].rename(columns={col: f'{col}_mean'})
        df = df.merge(temp_means, on=['turf_id', 'slot'], how='left')
        
        if f'{col}_mean' not in df.columns:
            df[f'{col}_mean'] = 0.0 
            
        df[col] = df[col].fillna(df[f'{col}_mean'])
        df = df.drop(columns=[f'{col}_mean'])
        
    return df.fillna(0)

def add_climate_features(df):
    climate_map = {
        1: {'is_sunny': 0.7, 'temp': 15}, 2: {'is_sunny': 0.8, 'temp': 20},
        3: {'is_sunny': 0.9, 'temp': 28}, 4: {'is_sunny': 1.0, 'temp': 35},
        5: {'is_sunny': 1.0, 'temp': 40}, 6: {'is_sunny': 0.6, 'temp': 38},
        7: {'is_sunny': 0.2, 'temp': 30}, 8: {'is_sunny': 0.2, 'temp': 29},
        9: {'is_sunny': 0.5, 'temp': 30}, 10: {'is_sunny': 0.8, 'temp': 28},
        11: {'is_sunny': 0.9, 'temp': 22}, 12: {'is_sunny': 0.8, 'temp': 16},
    }
    
    df['month'] = df['date'].dt.month
    climate_df = pd.DataFrame.from_dict(climate_map, orient='index')
    df = df.merge(climate_df, left_on='month', right_index=True, how='left')
    
    df['is_rainy'] = 0
    df['is_foggy'] = 0
    df['is_cool'] = (df['temp'] < 25).astype(int)
    df['is_extreme_heat'] = (df['temp'] > 38).astype(int)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def prepare_long_term_data(df, df_train):
    df_long = df.copy()
    df_long['month'] = df_long['date'].dt.month
    df_long['day_of_year'] = df_long['date'].dt.dayofyear
    df_long['week_of_year'] = df_long['date'].dt.isocalendar().week.astype(int)
    df_long['is_month_end'] = df_long['date'].dt.is_month_end.astype(int)
    df_long['month_cos'] = np.cos(2 * np.pi * df_long['month'] / 12)
    
    df_long['summer_season'] = df_long['month'].isin([3, 4, 5, 6]).astype(int)
    df_long['summer_peak'] = df_long['month'].isin([5, 6]).astype(int)
    df_long['is_holiday'] = 0 
    df_long['holiday_week'] = 0
    df_long['weather_code'] = 1 
    df_long['is_indoor'] = 1    
    df_long['sport_Pickleball'] = (df_long['sport_type'] == 'Pickleball').astype(int)
    df_long['weekend_peak'] = (df_long['day_of_week'] >= 5).astype(int)
    df_long['has_parking'] = 1
    
    if 'dow_hour_mean' in df_train.columns:
        dow_map = df_train.groupby(['day_of_week', 'hour'])['dow_hour_mean'].first().reset_index()
    else:
        dow_map = df_train.groupby(['day_of_week', 'hour'])['is_booked'].mean().reset_index()
        dow_map.rename(columns={'is_booked': 'dow_hour_mean'}, inplace=True)
        
    df_long = df_long.merge(dow_map, on=['day_of_week', 'hour'], how='left')
    df_long['dow_hour_mean'] = df_long['dow_hour_mean'].fillna(df_long['dow_hour_mean'].mean())
    return df_long

# ==========================================
# 3. SINGLE-PASS SYSTEM PIPELINE
# ==========================================
def calculate_xgboost_probs(df_future, df_train, model_short_term, model_long_term):
    """Runs the Dual-Horizon XGBoost models to get the baseline probability."""
    cutoff_date = pd.Timestamp.now().normalize() + timedelta(days=7)
    mask_short = df_future['date'] < cutoff_date
    df_short = df_future[mask_short].copy()
    df_long = df_future[~mask_short].copy()
    
    if not df_short.empty:
        df_short = prepare_momentum_data(df_short, df_train)
        df_short = add_climate_features(df_short) 
        short_features = ['day_of_week', 'hour', 'is_weekend', 'base_hourly_charge', 'streak_3d', 'booking_momentum', 'is_sunny', 'is_rainy', 'is_foggy', 'is_cool', 'temp', 'is_extreme_heat', 'month_sin', 'month_cos']
        df_short['booking_prob'] = model_short_term.predict_proba(df_short[short_features])[:, 1]
    
    if not df_long.empty:
        df_long = prepare_long_term_data(df_long, df_train)
        long_features = [
            'weather_code', 'is_indoor', 'dow_hour_mean', 'base_hourly_charge', 
            'month', 'holiday_week', 'is_holiday', 'weekend_peak', 'month_cos', 
            'day_of_week', 'day_of_year', 'summer_peak', 'summer_season', 
            'has_parking', 'week_of_year'
        ]
        df_long['booking_prob'] = model_long_term.predict_proba(df_long[long_features])[:, 1]

    return pd.concat([df_short, df_long]).sort_values(by=['date', 'slot'])

@st.cache_data
def get_predictions(use_ai_pricing=False):
    df_future = generate_30d_slots()
    df_train = load_data()
    model_short_term, model_long_term = load_xgb_models()
    
    # STEP 1: XGBoost Pass (Baseline Probability)
    df_preds = calculate_xgboost_probs(df_future, df_train, model_short_term, model_long_term)
    
    # STEP 2: Reinforcement Learning Price Adjustment (No re-run of XGBoost)
    if use_ai_pricing:
        rl_model = load_rl_model()
        dynamic_prices = []
        multipliers = []
        
        RL_TURF_IDS = ["turf_001", "turf_002", "turf_003", "turf_004", "turf_005"]
        RL_TIME_SLOTS = ["06:00", "07:00", "08:00", "09:00", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00", "21:00", "22:00"]
        price_multipliers = [0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]

        for _, row in df_preds.iterrows():
            base_price = row['base_hourly_charge']
            baseline_prob = row['booking_prob']
            hour_str = row['slot'].split('-')[0]
            
            turf_idx = RL_TURF_IDS.index(row['turf_id']) if row['turf_id'] in RL_TURF_IDS else 0
            slot_idx = RL_TIME_SLOTS.index(hour_str) if hour_str in RL_TIME_SLOTS else 12
            
            obs = {
                "turf_id_encoded": turf_idx,
                "time_slot_encoded": slot_idx,
                "day_of_week": row['day_of_week'],
                "base_price": np.array([base_price / 10000.0], dtype=np.float32),
                "baseline_probability": np.array([baseline_prob], dtype=np.float32)
            }
            
            action, _ = rl_model.predict(obs, deterministic=True)
            chosen_multiplier = price_multipliers[int(action)]
            
            dynamic_prices.append(int(base_price * chosen_multiplier))
            multipliers.append(chosen_multiplier)
            
        df_preds['original_price'] = df_preds['base_hourly_charge']
        df_preds['price_multiplier'] = multipliers
        df_preds['base_hourly_charge'] = dynamic_prices 
        
    return df_preds

# ==========================================
# 4. STREAMLIT UI BUILDER 
# ==========================================
def main():
    st.set_page_config(page_title="Turf Forecast ML", layout="centered", page_icon="🏟️")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["📅 Forecast Dashboard", "🧠 System Architecture"])
    st.sidebar.divider()
    st.sidebar.caption("Machine Learning Pipeline")
    
    if page == "📅 Forecast Dashboard":
        st.title("🏟️ Context-Aware Yield Manager")
        st.markdown("XGBoost Forecasting + PPO Deep Reinforcement Learning")
        
        use_ai_pricing = st.toggle("🤖 Enable V2 PPO Dynamic Pricing", value=False)
        df_preds = get_predictions(use_ai_pricing)
        
        unique_dates = df_preds['date'].dt.date.unique()
        formatted_dates = [d.strftime("%a, %b %d") for d in unique_dates]
        selected_date_str = st.selectbox("Select Date:", formatted_dates)
        
        selected_date = pd.to_datetime(selected_date_str + f", {unique_dates[0].year}", format="%a, %b %d, %Y")
        day_data = df_preds[df_preds['date'] == selected_date]
        
        st.divider()
        
        for _, row in day_data.iterrows():
            prob = row['booking_prob'] # This is now pure baseline probability
            slot = row['slot']
            current_price = row['base_hourly_charge']
            
            if use_ai_pricing:
                original_price = row['original_price']
                if current_price > original_price:
                    display_price = f"<s>₹{original_price}</s> <span style='color: #ff4c4c; font-weight: bold;'>₹{current_price} 📈</span>"
                elif current_price < original_price:
                    display_price = f"<s>₹{original_price}</s> <span style='color: #4caf50; font-weight: bold;'>₹{current_price} 📉</span>"
                else:
                    display_price = f"₹{current_price} ➖"
            else:
                display_price = f"₹{current_price}"
            
            if prob >= 0.75: # Adjusted to match your new RL guardrails!
                color, text, border = "#ffebeb", "🔴 Selling Fast!", "#ff4c4c"
            elif prob >= 0.50:
                color, text, border = "#fff4e5", "🟠 Filling Up", "#ffa500"
            else:
                color, text, border = "#e8f5e9", "🟢 Available", "#4caf50"

            st.markdown(f"""
                <div style="border: 2px solid {border}; background-color: {color}; padding: 15px; margin-bottom: 12px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 18px; font-weight: bold; color: #333;">{slot}</span><br>
                        <span style="font-size: 14px; color: #666;">{display_price}</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 16px; font-weight: bold; color: {border};">{text}</span><br>
                        <span style="font-size: 12px; color: #666; font-style: italic;">(Base Prob: {prob:.2f})</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    elif page == "🧠 System Architecture":
        st.title("🧠 System Architecture")
        st.markdown("### 🏟️ Streamlined Yield Management Pipeline")
        st.markdown("""
        **1. XGBoost Demand Forecast:** The system ingests temporal, climate, and momentum data to calculate a highly accurate baseline probability of a slot booking at standard rates.
        **2. Deep RL Pricing Engine:** The trained Reinforcement Learning agent (PPO) ingests the baseline probability and acts as an autonomous Yield Manager, dynamically surging or discounting the price to maximize expected revenue. 
        """)

if __name__ == "__main__":
    main()