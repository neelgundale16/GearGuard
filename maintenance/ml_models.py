import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
from django.conf import settings
from django.utils import timezone
from sklearn.ensemble import RandomForestRegressor



class PredictiveMaintenanceModel:
    """Predict when equipment will need maintenance"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = os.path.join(settings.BASE_DIR, 'ml_models', 'maintenance_predictor.pkl')
        
    def prepare_features(self, equipment_data):
        """Extract features from equipment maintenance history"""
        features = []
        
        for equipment in equipment_data:
            # Calculate features
            days_since_last_maintenance = (
                timezone.now().date() - equipment.last_maintenance
            ).days if equipment.last_maintenance else 365
            
            days_since_purchase = (
                timezone.now().date() - equipment.purchase_date
            ).days if equipment.purchase_date else 0
            
            maintenance_count = equipment.maintenance_requests.count()
            avg_time_between_maintenance = (
                days_since_purchase / maintenance_count 
                if maintenance_count > 0 else 0
            )
            
            # Status encoding
            status_map = {
                'operational': 0, 
                'maintenance': 1, 
                'broken': 2, 
                'retired': 3
            }
            
            features.append([
                days_since_last_maintenance,
                days_since_purchase,
                maintenance_count,
                avg_time_between_maintenance,
                status_map.get(equipment.status, 0),
            ])
        
        return np.array(features)
    
    def train(self, equipment_queryset):
        """Train the model on historical maintenance data"""
        
        # Prepare features
        X = self.prepare_features(equipment_queryset)
        
        # Create labels (days until next maintenance)
        y = []
        for equipment in equipment_queryset:
            if equipment.next_maintenance and equipment.last_maintenance:
                days_until_next = (
                    equipment.next_maintenance - equipment.last_maintenance
                ).days
                y.append(days_until_next)
            else:
                y.append(30)  # Default
        
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, self.model_path)
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'samples': len(X)
        }
    
    def predict(self, equipment):
        """Predict days until next maintenance needed"""
        
        # Load model if not loaded
        if self.model is None:
            if os.path.exists(self.model_path):
                saved = joblib.load(self.model_path)
                self.model = saved['model']
                self.scaler = saved['scaler']
            else:
                return None
        
        # Prepare features
        features = self.prepare_features([equipment])
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        return max(1, int(prediction))  # At least 1 day
    
    def get_maintenance_priority(self, equipment):
        """Calculate maintenance priority score (0-100)"""
        days_until = self.predict(equipment)
        
        if days_until is None:
            return 50
        
        # Lower days = higher priority
        if days_until <= 7:
            return 100
        elif days_until <= 30:
            return 80
        elif days_until <= 90:
            return 50
        else:
            return 20

class AnomalyDetectionModel:
    """Detect unusual patterns in equipment behavior"""
    
    def __init__(self):
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(contamination=0.1, random_state=42)
    
    def detect_anomalies(self, maintenance_requests):
        """Find unusual maintenance patterns"""
        
        features = []
        for request in maintenance_requests:
            # Extract temporal features
            hour = request.created_at.hour
            day_of_week = request.created_at.weekday()
            
            # Priority encoding
            priority_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
            priority_score = priority_map.get(request.priority, 0)
            
            # Type encoding
            type_map = {'preventive': 0, 'corrective': 1, 'emergency': 2}
            type_score = type_map.get(request.request_type, 0)
            
            features.append([
                hour,
                day_of_week,
                priority_score,
                type_score,
                request.estimated_hours or 0,
            ])
        
        features = np.array(features)
        
        # Detect anomalies
        predictions = self.model.fit_predict(features)
        
        # Return anomalous requests (prediction = -1)
        anomalies = []
        for i, pred in enumerate(predictions):
            if pred == -1:
                anomalies.append({
                    'request': maintenance_requests[i],
                    'anomaly_score': 1.0
                })
        
        return anomalies


class MaintenanceCostPredictor:
    """Predict maintenance costs based on historical data"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def train(self, maintenance_logs):
        """Train on historical cost data"""
        
        features = []
        costs = []
        
        for log in maintenance_logs:
            if log.cost:
                # Extract features
                request = log.maintenance_request
                
                priority_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
                type_map = {'preventive': 0, 'corrective': 1, 'emergency': 2}
                
                features.append([
                    priority_map.get(request.priority, 0),
                    type_map.get(request.request_type, 0),
                    log.maintenance_request.estimated_hours or 0,
                ])
                costs.append(float(log.cost))
        
        if len(features) > 10:  # Need minimum data
            X = np.array(features)
            y = np.array(costs)
            
            self.model.fit(X, y)
            return True
        
        return False
    
    def predict_cost(self, maintenance_request):
        """Predict cost for a maintenance request"""
        
        priority_map = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
        type_map = {'preventive': 0, 'corrective': 1, 'emergency': 2}
        
        features = np.array([[
            priority_map.get(maintenance_request.priority, 0),
            type_map.get(maintenance_request.request_type, 0),
            maintenance_request.estimated_hours or 0,
        ]])
        
        try:
            prediction = self.model.predict(features)[0]
            return max(0, prediction)
        except:
            return None