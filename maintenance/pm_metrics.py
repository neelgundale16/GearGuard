from django.db.models import Count
from datetime import timedelta
from django.utils import timezone

class PMMetrics:
    """Product Management Metrics"""
    
    @staticmethod
    def get_user_engagement():
        """DAU/MAU - Daily/Monthly Active Users"""
        from django.contrib.sessions.models import Session
        from django.contrib.auth.models import User
        
        dau = Session.objects.filter(expire_date__gte=timezone.now()).count()
        mau = User.objects.filter(is_active=True, last_login__gte=timezone.now() - timedelta(days=30)).count()
        stickiness = (dau / mau * 100) if mau > 0 else 0
        
        return {
            'dau': dau,
            'mau': mau,
            'stickiness': round(stickiness, 1),
            'status': 'excellent' if stickiness >= 40 else 'good' if stickiness >= 20 else 'needs_improvement'
        }
    
    @staticmethod
    def get_feature_adoption():
        """ML feature adoption rate"""
        from .models import Equipment, MaintenanceRequest
        
        total = Equipment.objects.count()
        ml_ready = sum(1 for eq in Equipment.objects.all() if MaintenanceRequest.objects.filter(equipment=eq, status='completed').count() >= 3)
        adoption_rate = (ml_ready / total * 100) if total > 0 else 0
        
        return {
            'total': total,
            'ml_ready': ml_ready,
            'adoption_rate': round(adoption_rate, 1),
            'status': 'on_track' if adoption_rate >= 60 else 'behind'
        }
    
    @staticmethod
    def get_cost_impact():
        """ROI calculation"""
        from .models import MaintenanceRequest
        
        emergency = MaintenanceRequest.objects.filter(request_type='emergency').count()
        preventive = MaintenanceRequest.objects.filter(request_type='preventive').count()
        
        emergency_cost = emergency * 800
        preventive_cost = preventive * 200
        total_cost = emergency_cost + preventive_cost
        
        prevented = int(preventive * 0.6)
        savings = prevented * 600
        
        roi = (savings / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'total_cost': total_cost,
            'savings': savings,
            'roi': round(roi, 1),
            'prevented_emergencies': prevented
        }