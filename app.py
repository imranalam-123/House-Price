# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from house_price_predictor import HousePricePredictor

# Set page configuration
st.set_page_config(
    page_title="🏡 House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better tab visibility
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    .upload-section {
        background-color: #e8f4fd;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #1f77b4;
        margin: 1rem 0;
    }
    .tab-instruction {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏡 House Price Prediction</h1>', unsafe_allow_html=True)
    st.markdown("### Predict house prices using Machine Learning")
    
    # Important instruction about tabs
    st.markdown("""
    <div class="tab-instruction">
    <h4>🚀 How to Use This App:</h4>
    <ul>
    <li><b>🎯 Single Prediction Tab:</b> Predict one house at a time using sliders</li>
    <li><b>📁 Bulk Prediction Tab:</b> <span style="color: #ff0000; font-weight: bold;">UPLOAD CSV FILES for multiple predictions</span></li>
    <li><b>📊 Data Overview Tab:</b> View training data statistics</li>
    <li><b>🤖 Model Info Tab:</b> See model performance metrics</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize predictor with caching
    @st.cache_resource
    def load_predictor():
        predictor = HousePricePredictor()
        predictor.load_and_explore_data('house_data.csv')
        predictor.preprocess_data()
        predictor.train_models()
        return predictor
    
    # Load predictor
    with st.spinner("🚀 Loading and training machine learning models..."):
        predictor = load_predictor()
        st.success("✅ Models loaded and trained successfully!")
    
    # Create tabs with clear labels
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 SINGLE PREDICTION", 
        "📁 BULK PREDICTION (FILE UPLOAD)", 
        "📊 DATA OVERVIEW", 
        "🤖 MODEL INFO"
    ])
    
    with tab1:
        st.header("🎯 Predict Single House Price")
        st.info("Use the sliders and inputs below to predict the price of a single house")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # House feature inputs
            col1a, col1b, col1c = st.columns(3)
            
            with col1a:
                area = st.number_input("**Area (sq ft)**", 500, 10000, 1500, 100)
                bedrooms = st.slider("**Bedrooms**", 1, 6, 3)
            
            with col1b:
                bathrooms = st.slider("**Bathrooms**", 1, 4, 2)
                stories = st.slider("**Stories**", 1, 3, 2)
            
            with col1c:
                parking = st.slider("**Parking Spaces**", 0, 3, 1)
                location = st.selectbox("**Location**", ["Urban", "Suburban", "Rural"])
            
            year_built = st.slider("**Year Built**", 1950, 2023, 2010)
            
            if st.button("🎯 PREDICT PRICE", type="primary", use_container_width=True):
                house_features = {
                    'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
                    'stories': stories, 'parking': parking, 'location': location,
                    'year_built': year_built
                }
                
                try:
                    predicted_price, model_used = predictor.predict_new_house(house_features)
                    
                    st.success(f"## Predicted Price: ${predicted_price:,.2f}")
                    st.info(f"**Model used:** {model_used}")
                    
                    # Show comparison
                    avg_price = predictor.df['price'].mean()
                    diff = predicted_price - avg_price
                    if diff > 0:
                        st.warning(f"📈 ${diff:,.0f} ABOVE market average")
                    else:
                        st.success(f"📉 ${abs(diff):,.0f} BELOW market average")
                        
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        
        with col2:
            st.subheader("💡 Quick Tips")
            st.info("""
            **For bulk predictions:**
            Go to **📁 BULK PREDICTION** tab to upload CSV files with multiple houses!
            
            **Location Impact:**
            - Urban: +$100K premium
            - Suburban: +$50K premium  
            - Rural: Base price
            """)
    
    with tab2:
        st.header("📁 BULK PREDICTION - FILE UPLOAD SYSTEM")
        st.success("✅ Upload a CSV file to predict prices for multiple houses at once!")
        
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "**DRAG & DROP YOUR CSV FILE HERE**",
            type=['csv'],
            help="Upload CSV with columns: area, bedrooms, bathrooms, location"
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                df_upload = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded! {df_upload.shape[0]} houses found.")
                
                # Show preview
                st.subheader("📄 File Preview")
                st.dataframe(df_upload.head(), width='stretch')
                
                # Check required columns
                required_columns = ['area', 'bedrooms', 'bathrooms', 'location']
                missing_columns = [col for col in required_columns if col not in df_upload.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing columns: {missing_columns}")
                    st.info("""
                    **Required CSV columns:**
                    - area (square feet)
                    - bedrooms (number)
                    - bathrooms (number)
                    - location (Urban/Suburban/Rural)
                    
                    **Optional columns:**
                    - stories (default: 2)
                    - parking (default: 1) 
                    - year_built (default: 2010)
                    """)
                else:
                    # Add missing optional columns
                    if 'stories' not in df_upload.columns:
                        df_upload['stories'] = 2
                    if 'parking' not in df_upload.columns:
                        df_upload['parking'] = 1
                    if 'year_built' not in df_upload.columns:
                        df_upload['year_built'] = 2010
                    
                    if st.button("🚀 PREDICT ALL HOUSES", type="primary", use_container_width=True):
                        predictions = []
                        models_used = []
                        
                        with st.spinner("🔄 Predicting prices for all houses..."):
                            for idx, row in df_upload.iterrows():
                                try:
                                    house_features = {
                                        'area': row['area'],
                                        'bedrooms': row['bedrooms'], 
                                        'bathrooms': row['bathrooms'],
                                        'stories': row['stories'],
                                        'parking': row['parking'],
                                        'location': row['location'],
                                        'year_built': row['year_built']
                                    }
                                    
                                    price, model = predictor.predict_new_house(house_features)
                                    predictions.append(price)
                                    models_used.append(model)
                                    
                                except Exception as e:
                                    predictions.append(None)
                                    models_used.append("Error")
                                    st.warning(f"Row {idx+1} error: {e}")
                        
                        # Add predictions to dataframe
                        df_upload['predicted_price'] = predictions
                        df_upload['model_used'] = models_used
                        
                        # Display results
                        st.subheader("📊 PREDICTION RESULTS")
                        st.dataframe(df_upload, width='stretch')
                        
                        # Statistics
                        valid_predictions = [p for p in predictions if p is not None]
                        if valid_predictions:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Houses", len(df_upload))
                            with col2:
                                st.metric("Successful", len(valid_predictions))
                            with col3:
                                st.metric("Avg Price", f"${np.mean(valid_predictions):,.0f}")
                            with col4:
                                st.metric("Total Value", f"${np.sum(valid_predictions):,.0f}")
                            
                            # Download button
                            csv = df_upload.to_csv(index=False)
                            st.download_button(
                                label="📥 DOWNLOAD PREDICTIONS AS CSV",
                                data=csv,
                                file_name="house_price_predictions.csv",
                                mime="text/csv",
                                type="primary",
                                use_container_width=True
                            )
            
            except Exception as e:
                st.error(f"❌ File error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # CSV Template Section
        st.subheader("📋 NEED A TEMPLATE?")
        st.info("Download this template CSV file and fill in your house data:")
        
        sample_data = {
            'area': [1200, 1800, 2200],
            'bedrooms': [2, 3, 4],
            'bathrooms': [1, 2, 2],
            'stories': [1, 2, 2],
            'parking': [1, 1, 2],
            'location': ['Urban', 'Suburban', 'Suburban'],
            'year_built': [2005, 2015, 2020]
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, width='stretch')
        
        csv_template = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 DOWNLOAD CSV TEMPLATE",
            data=csv_template,
            file_name="house_data_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with tab3:
        st.header("📊 Training Data Overview")
        st.info("This is the data used to train the machine learning models")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Houses", f"{predictor.df.shape[0]:,}")
            st.metric("Average Price", f"${predictor.df['price'].mean():,.0f}")
        with col2:
            st.metric("Features", predictor.df.shape[1])
            st.metric("Price Range", f"${predictor.df['price'].min():,.0f} - ${predictor.df['price'].max():,.0f}")
        
        st.dataframe(predictor.df.head(10), width='stretch')
    
    with tab4:
        st.header("🤖 Model Performance")
        st.info("Comparison of different machine learning models")
        
        if hasattr(predictor, 'results'):
            performance_data = []
            for name, result in predictor.results.items():
                performance_data.append({
                    'Model': name,
                    'R² Score': f"{result['r2']:.4f}",
                    'RMSE': f"${result['rmse']:,.0f}",
                    'MAE': f"${result['mae']:,.0f}"
                })
            
            st.dataframe(pd.DataFrame(performance_data), width='stretch', hide_index=True)
            
            # Find best model
            best_model = max(predictor.results.items(), key=lambda x: x[1]['r2'])
            st.success(f"🎯 Best Performing Model: **{best_model[0]}** (R²: {best_model[1]['r2']:.4f})")

if __name__ == "__main__":
    main()