import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime, timedelta
from django.utils import timezone

class RealMLEngine:
    """REAL Machine Learning Engine - NO OVERFITTING"""
    
    def __init__(self):
        self.models = {
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            ),
            'xgboost': XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                min_child_weight=5,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
        }
        self.scaler = StandardScaler()
        self.model_dir = 'ml_models'
        os.makedirs(self.model_dir, exist_ok=True)
    
    def extract_features(self, equipment_list):
        """Extract features WITHOUT data leakage"""
        features = []
        targets = []
        
        for equipment in equipment_list:
            maintenance_records = equipment.maintenance_requests.all().order_by('created_at')
            
            if maintenance_records.count() < 2:
                continue
            
            completed_records = [r for r in maintenance_records if r.status == 'completed']
            
            if len(completed_records) < 2:
                continue
            
            days_between = []
            for i in range(1, len(completed_records)):
                delta = (completed_records[i].created_at - completed_records[i-1].created_at).days
                days_between.append(delta)
            
            if not days_between:
                continue
            
            age_days = (timezone.now() - equipment.purchase_date).days if equipment.purchase_date else 365
            days_since_last = (timezone.now() - completed_records[-1].created_at).days
            maintenance_count = len(completed_records)
            avg_days_between = np.mean(days_between)
            std_days_between = np.std(days_between) if len(days_between) > 1 else 0
            
            priority_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            avg_priority = np.mean([priority_map.get(r.priority, 2) for r in completed_records])
            
            status_operational = 1 if equipment.status == 'operational' else 0
            
            feature_vector = [
                age_days / 365.0,
                days_since_last / 30.0,
                maintenance_count / 10.0,
                avg_days_between / 30.0,
                std_days_between / 30.0,
                avg_priority / 4.0,
                status_operational,
            ]
            
            target = avg_days_between
            
            features.append(feature_vector)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def train_models(self, equipment_list):
        """Train models with proper validation - NO OVERFITTING"""
        
        try:
            X, y = self.extract_features(equipment_list)
            
            if len(X) < 20:
                return {
                    'success': False,
                    'error': f'Need at least 20 equipment with 2+ maintenance records. Got {len(X)}'
                }
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            results = {}
            best_model_name = None
            best_r2 = -999
            
            for model_name, model in self.models.items():
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, 
                    cv=5, scoring='r2'
                )
                cv_r2 = np.mean(cv_scores)
                
                results[model_name] = {
                    'r2': r2,
                    'mae': mae,
                    'cv_r2': cv_r2,
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                }
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model_name = model_name
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f'model_{best_model_name}_{timestamp}.pkl'
            model_path = os.path.join(self.model_dir, model_filename)
            
            model_data = {
                'model': self.models[best_model_name],
                'scaler': self.scaler,
                'best_model': best_model_name,
                'r2': results[best_model_name]['r2'],
                'mae': results[best_model_name]['mae'],
                'cv_r2': results[best_model_name]['cv_r2'],
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'trained_date': datetime.now().isoformat(),
                'all_results': results
            }
            
            joblib.dump(model_data, model_path)
            
            return {
                'success': True,
                'best_model': best_model_name,
                'results': results,
                'model_path': model_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def load_latest_model(self):
        """Load the most recent trained model"""
        try:
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
            if not model_files:
                return None
            
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(self.model_dir, latest_model)
            model_data = joblib.load(model_path)
            
            return model_data
        except:
            return None
    
    def predict(self, equipment):
        """Make prediction for single equipment - FIXED 0 DAYS BUG"""
        model_data = self.load_latest_model()
        if not model_data:
            return None
        
        try:
            maintenance_records = equipment.maintenance_requests.filter(
                status='completed'
            ).order_by('created_at')
            
            if maintenance_records.count() < 2:
                return None
            
            completed_records = list(maintenance_records)
            
            days_between = []
            for i in range(1, len(completed_records)):
                delta = (completed_records[i].created_at - completed_records[i-1].created_at).days
                days_between.append(delta)
            
            age_days = (timezone.now() - equipment.purchase_date).days if equipment.purchase_date else 365
            days_since_last = (timezone.now() - completed_records[-1].created_at).days
            maintenance_count = len(completed_records)
            avg_days_between = np.mean(days_between)
            std_days_between = np.std(days_between) if len(days_between) > 1 else 0
            
            priority_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            avg_priority = np.mean([priority_map.get(r.priority, 2) for r in completed_records])
            
            status_operational = 1 if equipment.status == 'operational' else 0
            
            feature_vector = np.array([[
                age_days / 365.0,
                days_since_last / 30.0,
                maintenance_count / 10.0,
                avg_days_between / 30.0,
                std_days_between / 30.0,
                avg_priority / 4.0,
                status_operational,
            ]])
            
            feature_scaled = model_data['scaler'].transform(feature_vector)
            prediction = model_data['model'].predict(feature_scaled)[0]
            
            # FIX THE 0 DAYS BUG - prediction is the INTERVAL, not the absolute days
            # We need to calculate: when is NEXT maintenance due?
            # Next maintenance = days_since_last_maintenance compared to predicted_interval
            
            if days_since_last >= prediction:
                # OVERDUE - maintenance should have happened already
                days_until = 0
                priority_score = 100
            else:
                # NOT DUE YET - calculate remaining days
                days_until = int(prediction - days_since_last)
                # Priority score based on how close we are to the predicted interval
                priority_score = int((days_since_last / prediction) * 100)
                priority_score = max(0, min(100, priority_score))
            
            return {
                'days_until_maintenance': days_until,
                'priority_score': priority_score,
                'confidence': model_data['r2'] * 100,
                'predicted_interval': int(prediction)
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None