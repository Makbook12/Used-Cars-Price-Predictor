import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
import plotly.express as px
import base64
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 2px solid #E2E8F0;
    }
    .sub-header {
        font-size: 24px;
        color: #334155;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .prediction-box {
        background-color: #10B981;
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar .stButton>button {
        width: 100%;
        background-color: #1E40AF;
        color: white;
        font-weight: bold;
        height: 50px;
        margin-top: 20px;
    }
    .info-box {
        background-color: #DBEAFE;
        padding: 15px;
        border-left: 5px solid #3B82F6;
        margin: 10px 0;
        border-radius: 5px;
    }
    .stSlider > div > div > div {
        background-color: #3B82F6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to create a car animation for the loading state
def get_car_animation_base64():
    # Simple car SVG animation
    car_svg = '''
    <svg width="200" height="100" xmlns="http://www.w3.org/2000/svg">
        <style>
            .car { animation: drive 2s infinite linear; }
            @keyframes drive { 
                0% { transform: translateX(-50px); }
                100% { transform: translateX(250px); }
            }
            .wheel { animation: spin 1s infinite linear; }
            @keyframes spin { 
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
        <g class="car">
            <rect x="50" y="20" width="100" height="40" rx="10" fill="#1E40AF"/>
            <rect x="30" y="40" width="140" height="30" rx="5" fill="#3B82F6"/>
            <circle class="wheel" cx="50" cy="70" r="15" fill="#1F2937"/>
            <circle class="wheel" cx="150" cy="70" r="15" fill="#1F2937"/>
            <rect x="70" y="5" width="60" height="30" rx="5" fill="#60A5FA"/>
        </g>
    </svg>
    '''
    return base64.b64encode(car_svg.encode('utf-8')).decode('utf-8')

# Load cleaned data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data_ne_fifteen.csv")
    df['brand'] = df['brand'].str.split().str[0]
    return df

with st.spinner('Loading data...'):
    df = load_data()

# Feature engineering
@st.cache_data
def prepare_data(df):
    df['car_age'] = 2025 - df['model_year']
    df['mileage_per_year'] = df['mileage'] / df['car_age'].replace(0, 1)
    df['mileage_to_age_ratio'] = df['mileage'] / (df['car_age'] + 1)
    df['mileage_squared'] = df['mileage'] ** 2
    df['mileage_per_year_squared'] = df['mileage_per_year'] ** 2
    return df

df = prepare_data(df)

# Define features
X = df.drop(['price', 'log_price'], axis=1, errors='ignore')
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns

# Preprocessing pipeline
numerical_transformer = Pipeline([
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Model pipeline
@st.cache_resource
def create_model():
    model = XGBRegressor()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline

pipeline = create_model()

# Train the model
@st.cache_resource
def train_model(_pipeline, _X, _y):
    _pipeline.fit(_X, _y)
    return _pipeline

with st.spinner('Training model... Please wait'):
    pipeline = train_model(pipeline, X, df['price'])

# Main application
st.markdown('<div class="main-header">🚗 Used Car Price Predictor</div>', unsafe_allow_html=True)

# Create columns for layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="sub-header">Enter Car Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Create tabs for organized input
    tabs = st.tabs(["Basic Info", "Detailed Specs", "Colors & History"])
    
    with tabs[0]:
        brand = st.selectbox("Brand", sorted(df["brand"].unique()), index=0)
        
        # Filter models based on selected brand
        brand_models = sorted(df[df["brand"] == brand]["model"].unique())
        model = st.selectbox("Model", brand_models)
        
        model_year = st.slider(
            "Model Year", 
            int(df["model_year"].min()), 
            2025, 
            2020,
            help="Slide to select the manufacturing year of the car"
        )
        
        mileage = st.number_input(
            "Mileage (miles)", 
            0, 
            int(df["mileage"].max()), 
            50000,
            step=1000,
            format="%d",
            help="Enter the total miles driven"
        )
    
    with tabs[1]:
        fuel_type = st.selectbox("Fuel Type", sorted(df["fuel_type"].unique()))
        engine = st.selectbox("Engine", sorted(df["engine"].dropna().unique()))
        transmission = st.selectbox("Transmission", sorted(df["transmission"].dropna().unique()))
    
    with tabs[2]:
        ext_col = st.selectbox("Exterior Color", sorted(df["ext_col"].dropna().unique()))
        int_col = st.selectbox("Interior Color", sorted(df["int_col"].dropna().unique()))
        accident = st.selectbox("Accident History", ["No", "Yes"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create the input DataFrame
    input_data = {
        "brand": brand,
        "model": model,
        "model_year": model_year,
        "mileage": mileage,
        "fuel_type": fuel_type,
        "engine": engine,
        "transmission": transmission,
        "ext_col": ext_col,
        "int_col": int_col,
        "accident": accident
    }
    
    input_df = pd.DataFrame([input_data])
    input_df['car_age'] = 2025 - input_df['model_year']
    input_df['mileage_per_year'] = input_df['mileage'] / input_df['car_age'].replace(0, 1)
    input_df['mileage_to_age_ratio'] = input_df['mileage'] / (input_df['car_age'] + 1)
    input_df['mileage_squared'] = input_df['mileage'] ** 2
    input_df['mileage_per_year_squared'] = input_df['mileage_per_year'] ** 2
    
    # Align columns
    missing_cols = set(X.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[X.columns]
    
    # Predict button
    predict_btn = st.button("Predict Price", use_container_width=True)

with col2:
    st.markdown('<div class="sub-header">Market Insights</div>', unsafe_allow_html=True)
    
    # Display car age vs price chart
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig = px.scatter(
        df.sample(min(1000, len(df))), 
        x="car_age", 
        y="price", 
        color="brand",
        size="mileage",
        opacity=0.7,
        title="Car Age vs Price",
        labels={"car_age": "Car Age (Years)", "price": "Price ($)"}
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display brand price comparison
    st.markdown('<div class="card">', unsafe_allow_html=True)
    brand_avg = df.groupby("brand")["price"].mean().sort_values(ascending=False).head(10).reset_index()
    fig2 = px.bar(
        brand_avg,
        x="brand",
        y="price",
        title="Average Price by Brand (Top 10)",
        labels={"brand": "Brand", "price": "Average Price ($)"},
        color="price",
        color_continuous_scale=px.colors.sequential.Blues
    )
    fig2.update_layout(height=300)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Car value factors
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Key Factors Affecting Car Value")
    st.markdown("""
    <div class="info-box">
        <b>Age & Mileage:</b> Newer cars with lower mileage typically command higher prices.
    </div>
    <div class="info-box">
        <b>Brand & Model:</b> Luxury brands and popular models retain value better.
    </div>
    <div class="info-box">
        <b>Accident History:</b> Clean vehicle history significantly improves resale value.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Prediction section
if predict_btn:
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # Display loading animation
    car_animation = get_car_animation_base64()
    with st.spinner("Calculating price..."):
        st.markdown(f"""
        <div style="display: flex; justify-content: center; margin: 20px 0;">
            <img src="data:image/svg+xml;base64,{car_animation}" alt="Loading...">
        </div>
        """, unsafe_allow_html=True)
        
        # Make prediction
        prediction = pipeline.predict(input_df)[0]
        
        # Create price range (±5%)
        price_low = prediction * 0.95
        price_high = prediction * 1.05
        
        # Hide the spinner
        st.empty()
    
    # Display prediction
    st.markdown(f"""
    <div class="prediction-box">
        Estimated Price: ${prediction:,.2f}
        <div style="font-size: 16px; margin-top: 10px; opacity: 0.8;">
            Price Range: ${price_low:,.2f} - ${price_high:,.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display car details summary
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Vehicle Summary")
    
    # Create two columns for the summary
    sum_col1, sum_col2 = st.columns(2)
    
    with sum_col1:
        st.markdown(f"""
        - **Brand**: {brand}
        - **Model**: {model}
        - **Year**: {model_year}
        - **Age**: {2025 - model_year} years
        - **Mileage**: {mileage:,} miles
        """)
    
    with sum_col2:
        st.markdown(f"""
        - **Engine**: {engine}
        - **Fuel Type**: {fuel_type}
        - **Transmission**: {transmission}
        - **Exterior Color**: {ext_col}
        - **Accident History**: {accident}
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Comparison with similar cars
    similar_cars = df[
        (df["brand"] == brand) & 
        (df["model"] == model) & 
        (df["model_year"] >= model_year - 2) & 
        (df["model_year"] <= model_year + 2)
    ]
    
    if len(similar_cars) > 1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### How Your Car Compares")
        
        avg_price = similar_cars["price"].mean()
        median_price = similar_cars["price"].median()
        
        # Create a gauge chart to show where the prediction falls
        similar_cars_range = similar_cars["price"].quantile([0.1, 0.9])
        min_price, max_price = similar_cars_range[0.1], similar_cars_range[0.9]
        
        # Normalize the prediction for the gauge
        if max_price > min_price:
            norm_position = (prediction - min_price) / (max_price - min_price)
            norm_position = max(0, min(norm_position, 1))  # Clamp between 0 and 1
        else:
            norm_position = 0.5
        
        # Display comparison metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metrics_col1.metric("Your Estimated Price", f"${prediction:,.2f}")
        metrics_col2.metric("Average Market Price", f"${avg_price:,.2f}", f"{(prediction - avg_price) / avg_price:.1%}")
        metrics_col3.metric("Median Market Price", f"${median_price:,.2f}", f"{(prediction - median_price) / median_price:.1%}")
        
        # Create price distribution chart
        fig3 = px.histogram(
            similar_cars, 
            x="price",
            title=f"Price Distribution for Similar {brand} {model} ({model_year-2}-{model_year+2})",
            labels={"price": "Price ($)"},
            opacity=0.7
        )
        
        # Add vertical line for prediction
        fig3.add_vline(x=prediction, line_dash="dash", line_color="red", annotation_text="Your Car")
        
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; color: #64748B; font-size: 12px;">
    © 2025 Used Car Price Predictor | Data Updated: April 2025 | Powered by XGBoost
</div>
""", unsafe_allow_html=True)