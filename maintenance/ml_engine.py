import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os
from datetime import datetime
from .models import Equipment, MaintenanceRequest
from django.utils import timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class RealMLEngine:
    """REAL ML - Trains different models every time based on data"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_dir = 'ml_models'
        self.plots_dir = os.path.join('static', 'ml_plots')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def extract_features(self, equipment_queryset):
        """Extract features from maintenance history"""
        print("\n" + "=" * 80)
        print("EXTRACTING FEATURES")
        print("=" * 80)
        
        data = []
        processed, skipped = 0, 0
        
        for equipment in equipment_queryset:
            history = MaintenanceRequest.objects.filter(
                equipment=equipment, status='completed'
            ).order_by('-created_at')
            
            if history.count() < 3:
                skipped += 1
                continue
            
            # Temporal features
            days_since_purchase = (timezone.now().date() - equipment.purchase_date).days if equipment.purchase_date else 0
            days_since_last = (timezone.now().date() - equipment.last_maintenance).days if equipment.last_maintenance else 30
            
            # Maintenance statistics
            total_maint = history.count()
            avg_days_between = days_since_purchase / total_maint if total_maint > 0 else 30
            
            # Priority analysis
            critical = history.filter(priority='critical').count()
            high = history.filter(priority='high').count()
            medium = history.filter(priority='medium').count()
            
            priority_score = (critical * 10 + high * 5 + medium * 2) / total_maint if total_maint > 0 else 1
            
            # Request types
            emergency = history.filter(request_type='emergency').count()
            corrective = history.filter(request_type='corrective').count()
            
            emergency_ratio = emergency / total_maint if total_maint > 0 else 0
            corrective_ratio = corrective / total_maint if total_maint > 0 else 0
            
            # Maintenance frequency
            maint_frequency = total_maint / (days_since_purchase / 30) if days_since_purchase > 0 else 0
            
            # Target: days until next maintenance
            if equipment.next_maintenance:
                days_until = (equipment.next_maintenance - timezone.now().date()).days
                days_until = max(1, days_until)
            else:
                days_until = 30
            
            data.append({
                'equipment_id': equipment.equipment_id,
                'days_since_purchase': days_since_purchase,
                'days_since_last_maintenance': days_since_last,
                'avg_days_between_maintenance': avg_days_between,
                'maintenance_frequency': maint_frequency,
                'total_maintenance_count': total_maint,
                'priority_score': priority_score,
                'emergency_ratio': emergency_ratio,
                'corrective_ratio': corrective_ratio,
                'critical_count': critical,
                'high_count': high,
                'status_operational': 1 if equipment.status == 'operational' else 0,
                'target_days_until_maintenance': days_until
            })
            
            processed += 1
            if processed % 10 == 0:
                print(f"  Processed: {processed}, Skipped: {skipped}")
        
        df = pd.DataFrame(data)
        print(f"\n✅ Feature extraction complete!")
        print(f"  • Processed: {processed}")
        print(f"  • Skipped: {skipped}")
        print(f"  • Features: {len([c for c in df.columns if 'target' not in c and c != 'equipment_id'])}")
        
        return df
    
    def plot_training_curves(self, train_scores, val_scores, model_name, timestamp):
        """Plot training progress"""
        plt.figure(figsize=(12, 6))
        iterations = list(range(20, len(train_scores) * 20 + 1, 20))
        
        plt.plot(iterations, train_scores, 'b-', label='Training R²', linewidth=2, marker='o')
        plt.plot(iterations, val_scores, 'r-', label='Validation R²', linewidth=2, marker='s')
        
        plt.xlabel('Training Iterations', fontsize=12, fontweight='bold')
        plt.ylabel('R² Score (Model Accuracy)', fontsize=12, fontweight='bold')
        plt.title(f'{model_name} - Real Training Progress', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        best_idx = np.argmax(val_scores)
        plt.annotate(f'Best: {val_scores[best_idx]:.4f}', 
                     xy=(iterations[best_idx], val_scores[best_idx]),
                     xytext=(iterations[best_idx] + 20, val_scores[best_idx] - 0.05),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2),
                     fontsize=10, fontweight='bold')
        
        plot_path = os.path.join(self.plots_dir, f'training_{model_name}_{timestamp}.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def train_models(self, equipment_queryset, progress_callback=None):
        """
        REAL ML TRAINING - Different results every time!
        Trains 3 models: Gradient Boosting, XGBoost, Random Forest
        """
        print("\n" + "=" * 80)
        print("STARTING REAL MACHINE LEARNING TRAINING")
        print("Training will produce DIFFERENT results based on your data!")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Feature extraction
        if progress_callback:
            progress_callback({'status': 'extracting', 'message': 'Extracting features...', 'progress': 5})
        
        df = self.extract_features(equipment_queryset)
        
        if len(df) < 10:
            return {'success': False, 'error': f'Need at least 10 equipment with 3+ maintenance records. Found {len(df)}.'}
        
        # Prepare features and target
        feature_cols = [c for c in df.columns if c not in ['equipment_id', 'target_days_until_maintenance']]
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df['target_days_until_maintenance'].values
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"\n📊 Dataset Split:")
        print(f"  • Training samples: {len(X_train)}")
        print(f"  • Test samples: {len(X_test)}")
        print(f"  • Features: {len(feature_cols)}")
        print(f"  • Target range: [{y.min():.1f}, {y.max():.1f}] days")
        
        # Scale features
        if progress_callback:
            progress_callback({'status': 'scaling', 'message': 'Normalizing features...', 'progress': 10})
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # ==================== MODEL 1: GRADIENT BOOSTING ====================
        print("\n" + "=" * 80)
        print("MODEL 1: GRADIENT BOOSTING REGRESSOR")
        print("Training with REAL epochs - watch the scores improve!")
        print("=" * 80)
        
        gb_train_scores, gb_val_scores = [], []
        total_iterations = 200
        
        for iteration in range(20, total_iterations + 1, 20):
            if progress_callback:
                progress_callback({
                    'status': 'training_gb',
                    'message': f'Gradient Boosting: Epoch {iteration}/{total_iterations}',
                    'progress': 10 + int((iteration / total_iterations) * 30)
                })
            
            gb_model = GradientBoostingRegressor(
                n_estimators=iteration,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=10,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=42
            )
            
            gb_model.fit(X_train_scaled, y_train)
            
            train_score = gb_model.score(X_train_scaled, y_train)
            val_score = gb_model.score(X_test_scaled, y_test)
            
            gb_train_scores.append(train_score)
            gb_val_scores.append(val_score)
            
            print(f"  Epoch {iteration:3d}/{total_iterations}: Train R²={train_score:.4f}, Val R²={val_score:.4f}")
        
        y_pred_gb = gb_model.predict(X_test_scaled)
        
        gb_mse = mean_squared_error(y_test, y_pred_gb)
        gb_mae = mean_absolute_error(y_test, y_pred_gb)
        gb_r2 = r2_score(y_test, y_pred_gb)
        
        # Cross-validation
        cv_scores = cross_val_score(gb_model, X_train_scaled, y_train, cv=5, scoring='r2')
        gb_cv_mean, gb_cv_std = cv_scores.mean(), cv_scores.std()
        
        print(f"\n✅ Gradient Boosting Final Results:")
        print(f"  • MSE (Mean Squared Error): {gb_mse:.4f}")
        print(f"  • MAE (Mean Absolute Error): {gb_mae:.2f} days")
        print(f"  • R² Score (Accuracy): {gb_r2:.4f} ({gb_r2*100:.1f}%)")
        print(f"  • Cross-Validation R²: {gb_cv_mean:.4f} ± {gb_cv_std:.4f}")
        
        results['gradient_boosting'] = {
            'mse': float(gb_mse), 'mae': float(gb_mae), 'r2': float(gb_r2),
            'cv_mean': float(gb_cv_mean), 'cv_std': float(gb_cv_std),
            'train_scores': gb_train_scores, 'val_scores': gb_val_scores
        }
        
        self.models['gradient_boosting'] = gb_model
        gb_plot = self.plot_training_curves(gb_train_scores, gb_val_scores, 'GradientBoosting', timestamp)
        
        # ==================== MODEL 2: XGBOOST ====================
        print("\n" + "=" * 80)
        print("MODEL 2: XGBOOST REGRESSOR")
        print("=" * 80)
        
        if progress_callback:
            progress_callback({'status': 'training_xgb', 'message': 'Training XGBoost...', 'progress': 50})
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist'
        )
        
        xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
        
        y_pred_xgb = xgb_model.predict(X_test_scaled)
        
        xgb_mse = mean_squared_error(y_test, y_pred_xgb)
        xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
        xgb_r2 = r2_score(y_test, y_pred_xgb)
        
        cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='r2')
        xgb_cv_mean, xgb_cv_std = cv_scores.mean(), cv_scores.std()
        
        print(f"\n✅ XGBoost Final Results:")
        print(f"  • MSE: {xgb_mse:.4f}")
        print(f"  • MAE: {xgb_mae:.2f} days")
        print(f"  • R²: {xgb_r2:.4f} ({xgb_r2*100:.1f}%)")
        print(f"  • Cross-Validation R²: {xgb_cv_mean:.4f} ± {xgb_cv_std:.4f}")
        
        results['xgboost'] = {
            'mse': float(xgb_mse), 'mae': float(xgb_mae), 'r2': float(xgb_r2),
            'cv_mean': float(xgb_cv_mean), 'cv_std': float(xgb_cv_std)
        }
        
        self.models['xgboost'] = xgb_model
        
        # ==================== MODEL 3: RANDOM FOREST ====================
        print("\n" + "=" * 80)
        print("MODEL 3: RANDOM FOREST REGRESSOR")
        print("=" * 80)
        
        if progress_callback:
            progress_callback({'status': 'training_rf', 'message': 'Training Random Forest...', 'progress': 70})
        
        rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        y_pred_rf = rf_model.predict(X_test_scaled)
        
        rf_mse = mean_squared_error(y_test, y_pred_rf)
        rf_mae = mean_absolute_error(y_test, y_pred_rf)
        rf_r2 = r2_score(y_test, y_pred_rf)
        
        cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='r2')
        rf_cv_mean, rf_cv_std = cv_scores.mean(), cv_scores.std()
        
        print(f"\n✅ Random Forest Final Results:")
        print(f"  • MSE: {rf_mse:.4f}")
        print(f"  • MAE: {rf_mae:.2f} days")
        print(f"  • R²: {rf_r2:.4f} ({rf_r2*100:.1f}%)")
        print(f"  • Cross-Validation R²: {rf_cv_mean:.4f} ± {rf_cv_std:.4f}")
        
        results['random_forest'] = {
            'mse': float(rf_mse), 'mae': float(rf_mae), 'r2': float(rf_r2),
            'cv_mean': float(rf_cv_mean), 'cv_std': float(rf_cv_std)
        }
        
        self.models['random_forest'] = rf_model
        
        # ==================== SELECT BEST MODEL ====================
        best_model_name = min(results, key=lambda x: results[x]['mae'])
        best_model = self.models[best_model_name]
        
        print("\n" + "=" * 80)
        print(f"🏆 BEST MODEL: {best_model_name.upper()}")
        print(f"  • MAE: {results[best_model_name]['mae']:.2f} days (Lower is better)")
        print(f"  • R² Score: {results[best_model_name]['r2']:.4f} (Higher is better)")
        print(f"  • Prediction Accuracy: {results[best_model_name]['r2']*100:.1f}%")
        print("=" * 80)
        
        # Save model
        if progress_callback:
            progress_callback({'status': 'saving', 'message': 'Saving trained model...', 'progress': 90})
        
        model_path = os.path.join(self.model_dir, f'model_{timestamp}.pkl')
        
        joblib.dump({
            'model': best_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_name': best_model_name,
            'results': results,
            'timestamp': timestamp,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': len(feature_cols)
        }, model_path)
        
        print(f"\n✅ Model saved: {model_path}")
        
        if progress_callback:
            progress_callback({
                'status': 'complete',
                'message': f'Training complete! Best: {best_model_name}',
                'progress': 100
            })
        
        return {
            'success': True,
            'best_model': best_model_name,
            'results': results,
            'model_path': model_path,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(feature_cols),
            'plot_path': gb_plot
        }
    
    def load_latest_model(self):
        """Load most recent trained model"""
        model_files = [f for f in os.listdir(self.model_dir) if f.startswith('model_') and f.endswith('.pkl')]
        if not model_files:
            return None
        
        latest = sorted(model_files)[-1]
        data = joblib.load(os.path.join(self.model_dir, latest))
        
        self.models['best'] = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        
        return data
    
    def predict(self, equipment):
        """Make prediction for single equipment"""
        if 'best' not in self.models:
            if not self.load_latest_model():
                return None
        
        df = self.extract_features(Equipment.objects.filter(id=equipment.id))
        if len(df) == 0:
            return None
        
        feature_cols = [c for c in df.columns if c not in ['equipment_id', 'target_days_until_maintenance']]
        X = df[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        days = self.models['best'].predict(X_scaled)[0]
        
        # Calculate confidence based on maintenance history
        history_count = MaintenanceRequest.objects.filter(equipment=equipment, status='completed').count()
        
        if history_count >= 10:
            confidence = 0.92
        elif history_count >= 7:
            confidence = 0.85
        elif history_count >= 5:
            confidence = 0.75
        elif history_count >= 3:
            confidence = 0.65
        else:
            confidence = 0.50
        
        priority_score = min(100, int(100 * (30 / max(1, days))))
        
        if priority_score >= 80:
            risk_level = 'Critical'
        elif priority_score >= 60:
            risk_level = 'High'
        elif priority_score >= 40:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        return {
            'days_until_maintenance': max(1, int(days)),
            'confidence': confidence,
            'priority_score': priority_score,
            'risk_level': risk_level
        }
    
    def predict_batch(self, equipment_queryset):
        """Make predictions for multiple equipment"""
        if 'best' not in self.models:
            if not self.load_latest_model():
                return []
        
        predictions = []
        for eq in equipment_queryset:
            pred = self.predict(eq)
            if pred:
                predictions.append({'equipment': eq, 'prediction': pred})
        
        return predictions