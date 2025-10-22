# house_price_predictor.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.df = None
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_explore_data(self, file_path):
        """Load dataset and perform exploratory data analysis"""
        try:
            # Load dataset
            self.df = pd.read_csv(file_path)
            print("📊 Dataset loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            
            # Display basic info
            print("\n📈 Dataset Info:")
            print(self.df.info())
            
            print("\n🔍 First 5 rows:")
            print(self.df.head())
            
            print("\n📊 Statistical Summary:")
            print(self.df.describe())
            
            print("\n❓ Missing Values:")
            print(self.df.isnull().sum())
            
            return self.df
            
        except FileNotFoundError:
            print("❌ File not found. Creating sample data...")
            self._create_sample_data()
            return self.df
    
    def _create_sample_data(self):
        """Create sample housing data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'area': np.random.normal(1500, 500, n_samples).astype(int),
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'stories': np.random.randint(1, 3, n_samples),
            'parking': np.random.randint(0, 3, n_samples),
            'location': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples),
            'year_built': np.random.randint(1980, 2023, n_samples),
        }
        
        self.df = pd.DataFrame(data)
        # Make price dependent on features (more realistic)
        self.df['price'] = (
            self.df['area'] * 200 + 
            self.df['bedrooms'] * 50000 + 
            self.df['bathrooms'] * 30000 +
            self.df['stories'] * 25000 +
            self.df['parking'] * 15000 +
            (self.df['location'].map({'Urban': 100000, 'Suburban': 50000, 'Rural': 0})) +
            ((self.df['year_built'] - 2000) * 1000) +
            np.random.normal(0, 30000, n_samples)
        ).astype(int)
        
        print("✅ Sample data created successfully!")
        print(f"Sample data shape: {self.df.shape}")
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        if self.df is None:
            raise ValueError("No data loaded. Please call load_and_explore_data() first.")
            
        print("\n🔄 Preprocessing data...")
        
        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
        # Handle categorical variables
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            self.df[col] = self.label_encoders[col].fit_transform(self.df[col])
        
        print("✅ Data preprocessing completed!")
        return self.df
    
    def visualize_data(self, save_plots=False):
        """Create visualizations to understand the data"""
        if self.df is None:
            raise ValueError("No data available for visualization.")
            
        print("\n📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Price distribution
        axes[0, 0].hist(self.df['price'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Price Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Area vs Price
        axes[0, 1].scatter(self.df['area'], self.df['price'], alpha=0.6, color='coral')
        axes[0, 1].set_title('Area vs Price', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Area (sq ft)')
        axes[0, 1].set_ylabel('Price ($)')
        
        # Bedrooms vs Price
        bedrooms_price = self.df.groupby('bedrooms')['price'].mean()
        axes[0, 2].bar(bedrooms_price.index, bedrooms_price.values, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('Average Price by Bedrooms', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Bedrooms')
        axes[0, 2].set_ylabel('Average Price ($)')
        
        # Correlation heatmap
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1, 0], 
                   fmt='.2f', linewidths=0.5)
        axes[1, 0].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        # Location impact
        if 'location' in self.df.columns and 'location' in self.label_encoders:
            # Create a temporary dataframe with decoded location names for visualization
            temp_df = self.df.copy()
            temp_df['location_name'] = self.label_encoders['location'].inverse_transform(temp_df['location'])
            location_price = temp_df.groupby('location_name')['price'].mean()
            
            axes[1, 1].bar(range(len(location_price)), location_price.values, 
                          color='orange', edgecolor='black')
            axes[1, 1].set_title('Average Price by Location', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Location')
            axes[1, 1].set_ylabel('Average Price ($)')
            axes[1, 1].set_xticks(range(len(location_price)))
            axes[1, 1].set_xticklabels(location_price.index, rotation=45)
        
        # Year built vs Price
        if 'year_built' in self.df.columns:
            year_price = self.df.groupby('year_built')['price'].mean()
            axes[1, 2].plot(year_price.index, year_price.values, marker='o', linewidth=2, markersize=4)
            axes[1, 2].set_title('Price Trend by Year Built', fontsize=14, fontweight='bold')
            axes[1, 2].set_xlabel('Year Built')
            axes[1, 2].set_ylabel('Average Price ($)')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('house_price_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Visualization saved as 'house_price_analysis.png'")
        
        plt.show()
        
        # Additional pairplot for relationships
        print("\n🔄 Creating pairplot...")
        numeric_columns = ['price', 'area', 'bedrooms', 'bathrooms']
        available_columns = [col for col in numeric_columns if col in self.df.columns]
        
        if len(available_columns) >= 2:
            sns.pairplot(self.df[available_columns], diag_kind='hist', corner=True)
            plt.suptitle('Feature Relationships', y=1.02, fontsize=16, fontweight='bold')
            plt.show()
    
    def train_models(self, test_size=0.2):
        """Train multiple models and compare performance"""
        if self.df is None:
            raise ValueError("No data available for training.")
            
        print("\n🤖 Training machine learning models...")
        
        # Prepare features and target
        X = self.df.drop('price', axis=1)
        y = self.df['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"🔧 Training {name}...")
            
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                predictions = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            results[name] = {
                'model': model,
                'predictions': predictions,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'y_test': y_test
            }
            
            print(f"✅ {name} - RMSE: ${rmse:,.2f}, R²: {r2:.4f}")
        
        self.results = results
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        
        return results
    
    def compare_models(self):
        """Compare performance of all trained models"""
        if not self.results:
            raise ValueError("No models trained. Please call train_models() first.")
            
        print("\n📊 Model Comparison:")
        
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'RMSE': f"${result['rmse']:,.2f}",
                'MAE': f"${result['mae']:,.2f}",
                'R² Score': f"{result['r2']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Visual comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # RMSE comparison
        models = list(self.results.keys())
        rmses = [self.results[model]['rmse'] for model in models]
        
        bars1 = axes[0].bar(models, rmses, color=['skyblue', 'lightgreen', 'orange'], 
                           edgecolor='black', alpha=0.7)
        axes[0].set_title('Model Comparison - RMSE\n(Lower is better)', fontweight='bold')
        axes[0].set_ylabel('RMSE ($)')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, rmses):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmses)*0.01,
                        f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')
        
        # R² comparison
        r2_scores = [self.results[model]['r2'] for model in models]
        
        bars2 = axes[1].bar(models, r2_scores, color=['skyblue', 'lightgreen', 'orange'], 
                           edgecolor='black', alpha=0.7)
        axes[1].set_title('Model Comparison - R² Score\n(Higher is better)', fontweight='bold')
        axes[1].set_ylabel('R² Score')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars2, r2_scores):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def plot_feature_importance(self, top_n=10):
        """Plot feature importance for tree-based models"""
        if not self.results:
            raise ValueError("No models trained. Please call train_models() first.")
            
        print("\n🔍 Analyzing feature importance...")
        
        tree_models = ['Random Forest', 'XGBoost']
        available_models = [model for model in tree_models if model in self.results]
        
        if not available_models:
            print("❌ No tree-based models available for feature importance analysis.")
            return
        
        n_models = len(available_models)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for i, model_name in enumerate(available_models):
            model = self.results[model_name]['model']
            
            if model_name == 'Random Forest':
                importances = model.feature_importances_
            else:  # XGBoost
                importances = model.feature_importances_
            
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': importances
            }).sort_values('importance', ascending=True).tail(top_n)
            
            axes[i].barh(feature_importance['feature'], feature_importance['importance'],
                        color='lightcoral', edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{model_name} - Top {top_n} Features', fontweight='bold')
            axes[i].set_xlabel('Importance Score')
            
            # Add value labels
            for j, (feature, importance) in enumerate(zip(feature_importance['feature'], 
                                                         feature_importance['importance'])):
                axes[i].text(importance, j, f' {importance:.3f}', 
                           va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def plot_predictions_vs_actual(self):
        """Plot predicted vs actual prices"""
        if not self.results:
            raise ValueError("No models trained. Please call train_models() first.")
            
        print("\n📈 Plotting predictions vs actual values...")
        
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (name, result) in enumerate(self.results.items()):
            predictions = result['predictions']
            y_test = result['y_test']
            r2 = result['r2']
            
            axes[i].scatter(y_test, predictions, alpha=0.6, color='purple')
            
            # Perfect prediction line
            max_val = max(max(y_test), max(predictions))
            min_val = min(min(y_test), min(predictions))
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            axes[i].set_xlabel('Actual Price ($)')
            axes[i].set_ylabel('Predicted Price ($)')
            axes[i].set_title(f'{name}\nR² = {r2:.4f}', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict_new_house(self, house_features):
        """Predict price for a new house"""
        if not self.results:
            raise ValueError("No models trained. Please call train_models() first.")
            
        # Find the best model based on R² score
        best_model_name = max(self.results.items(), 
                            key=lambda x: x[1]['r2'])[0]
        best_model = self.results[best_model_name]['model']
        
        # Create feature DataFrame
        feature_df = pd.DataFrame([house_features])
        
        # Preprocess the input features
        for col in feature_df.columns:
            if col in self.label_encoders:
                if house_features[col] in self.label_encoders[col].classes_:
                    feature_df[col] = self.label_encoders[col].transform([house_features[col]])[0]
                else:
                    # Default to first category if unknown
                    feature_df[col] = 0
                    print(f"⚠️ Warning: Unknown category '{house_features[col]}' for '{col}'. Using default value.")
        
        # Ensure all columns are present and in correct order
        for col in self.X_train.columns:
            if col not in feature_df.columns:
                feature_df[col] = 0
                print(f"⚠️ Warning: Missing feature '{col}'. Using default value 0.")
        
        feature_df = feature_df[self.X_train.columns]
        
        # Make prediction
        if best_model_name == 'Linear Regression':
            feature_scaled = self.scaler.transform(feature_df)
            prediction = best_model.predict(feature_scaled)[0]
        else:
            prediction = best_model.predict(feature_df)[0]
        
        print(f"\n🏡 Price Prediction using {best_model_name}:")
        print(f"💰 Estimated Price: ${prediction:,.2f}")
        
        return prediction, best_model_name
    
    def get_model_performance(self):
        """Get detailed model performance metrics"""
        if not self.results:
            raise ValueError("No models trained. Please call train_models() first.")
            
        performance_data = []
        for name, result in self.results.items():
            performance_data.append({
                'Model': name,
                'R² Score': result['r2'],
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'MSE': result['mse']
            })
        
        return pd.DataFrame(performance_data)

# Example usage and testing
def main():
    """Example usage of the HousePricePredictor class"""
    print("🏡 House Price Predictor Demo")
    print("=" * 50)
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Load data (will create sample data if file not found)
    df = predictor.load_and_explore_data('house_data.csv')
    
    # Preprocess data
    predictor.preprocess_data()
    
    # Visualize data
    predictor.visualize_data()
    
    # Train models
    results = predictor.train_models()
    
    # Compare models
    comparison_df = predictor.compare_models()
    
    # Show feature importance
    predictor.plot_feature_importance()
    
    # Show predictions vs actual
    predictor.plot_predictions_vs_actual()
    
    # Example prediction
    sample_house = {
        'area': 1800,
        'bedrooms': 3,
        'bathrooms': 2,
        'stories': 2,
        'parking': 1,
        'location': 'Urban',
        'year_built': 2015
    }
    
    predicted_price, model_used = predictor.predict_new_house(sample_house)
    
    print(f"\n🎯 Prediction Summary:")
    print(f"Best Model: {model_used}")
    print(f"Predicted Price: ${predicted_price:,.2f}")
    
    return predictor

if __name__ == "__main__":
    predictor = main()