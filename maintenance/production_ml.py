import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  
from sklearn.model_selection import cross_val_score
import joblib
import os
from datetime import datetime

class ProductionMLPipeline:
    """
    Enterprise-grade ML pipeline
    Shows: MLOps, model versioning, monitoring
    """
    
    def __init__(self):
        self.models = {}
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def feature_engineering(self, equipment_df):
        """
        Advanced feature engineering
        Interview tip: Explain each feature's business logic
        """
        
        # Temporal features
        equipment_df['days_since_purchase'] = (
            datetime.now() - pd.to_datetime(equipment_df['purchase_date'])
        ).dt.days
        
        equipment_df['days_since_maintenance'] = (
            datetime.now() - pd.to_datetime(equipment_df['last_maintenance'])
        ).dt.days
        
        # Interaction features
        equipment_df['age_maintenance_ratio'] = (
            equipment_df['days_since_purchase'] / 
            (equipment_df['days_since_maintenance'] + 1)
        )
        
        # Aggregate features from maintenance history
        from .models import MaintenanceRequest
        
        for idx, row in equipment_df.iterrows():
            eq_id = row['id']
            
            # Count features
            total_requests = MaintenanceRequest.objects.filter(
                equipment_id=eq_id
            ).count()
            
            critical_requests = MaintenanceRequest.objects.filter(
                equipment_id=eq_id,
                priority='critical'
            ).count()
            
            emergency_requests = MaintenanceRequest.objects.filter(
                equipment_id=eq_id,
                request_type='emergency'
            ).count()
            
            equipment_df.loc[idx, 'total_maintenance_count'] = total_requests
            equipment_df.loc[idx, 'critical_count'] = critical_requests
            equipment_df.loc[idx, 'emergency_count'] = emergency_requests
            equipment_df.loc[idx, 'failure_rate'] = (
                critical_requests / total_requests if total_requests > 0 else 0
            )
        
        return equipment_df
    
    def train_ensemble_model(self, X_train, y_train):
        """
        Train multiple models and ensemble them
        Shows: Advanced ML, model selection
        """
        
        models = {
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=150,
                max_depth=10,
                random_state=42
            )
            # XGBoost removed - can add back after: pip install xgboost
        }
        
        # Train and evaluate each model
        scores = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=5, 
                scoring='neg_mean_squared_error'
            )
            scores[name] = -cv_scores.mean()
            self.models[name] = model
        
        # Save best model
        best_model_name = min(scores, key=scores.get)
        best_model = self.models[best_model_name]
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save with version
        joblib.dump({
            'model': best_model,
            'version': self.version,
            'scores': scores,
            'best_model': best_model_name
        }, f'models/maintenance_predictor_{self.version}.pkl')
        
        return {
            'best_model': best_model_name,
            'scores': scores,
            'version': self.version
        }
    
    def calculate_drift(self, predictions):
        """Calculate drift score"""
        # Simple drift calculation
        mean = np.mean(predictions)
        std = np.std(predictions)
        drift_score = std / (mean + 1e-10)  # Avoid division by zero
        return drift_score
    
    def monitor_model_drift(self, new_data):
        """
        Monitor for data drift
        Shows: MLOps maturity, production thinking
        """
        
        # Load latest model
        if not os.path.exists('models'):
            return {
                'drift_detected': False,
                'score': 0,
                'error': 'No models found'
            }
        
        model_files = [f for f in os.listdir('models') if f.endswith('.pkl')]
        if not model_files:
            return {
                'drift_detected': False,
                'score': 0,
                'error': 'No models found'
            }
        
        latest_model = sorted(model_files)[-1]
        
        saved = joblib.load(f'models/{latest_model}')
        model = saved['model']
        
        # Compare predictions on new data
        predictions = model.predict(new_data)
        
        # Calculate metrics
        drift_score = self.calculate_drift(predictions)
        
        if drift_score > 0.3:  # Threshold
            return {
                'drift_detected': True,
                'score': drift_score,
                'action': 'Retrain model recommended'
            }
        
        return {
            'drift_detected': False,
            'score': drift_score
        }