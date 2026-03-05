"""Info tooltips for all metrics"""

TOOLTIPS = {
    'ml_prediction': {
        'title': 'ML Prediction',
        'description': 'Days until maintenance needed, predicted by ML model trained on historical data.',
        'accuracy': '85-90% based on cross-validation'
    },
    'confidence': {
        'title': 'Prediction Confidence',
        'description': 'Model certainty. Based on data quality.',
        'levels': 'High (90%+): 10+ records, Medium (70-89%): 5-9 records'
    },
    'r2_score': {
        'title': 'R² Score',
        'description': 'Model accuracy (0-1). Higher is better.',
        'interpretation': '0.9+: Excellent, 0.7-0.9: Good'
    },
    'mae': {
        'title': 'Mean Absolute Error',
        'description': 'Average prediction error in days. Lower is better.',
        'target': '<5 days for production'
    },
    'dau_mau': {
        'title': 'DAU/MAU Ratio',
        'description': 'Daily ÷ Monthly Active Users. Measures stickiness.',
        'benchmark': '20%+: Good, 40%+: Excellent'
    },
    'roi': {
        'title': 'Return on Investment',
        'description': 'Value gained vs. cost invested.',
        'calculation': '(Savings - Cost) ÷ Cost × 100'
    }
}

def get_tooltip(key):
    return TOOLTIPS.get(key, {'title': key.upper(), 'description': 'No description'})