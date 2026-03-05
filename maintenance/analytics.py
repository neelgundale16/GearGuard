from django.db.models import Count, Avg, Sum, F, Q
from django.utils import timezone
from datetime import timedelta
import pandas as pd
from maintenance.models import MaintenanceRequest, Equipment, WorkCenter
from maintenance.ml_models import PredictiveMaintenanceModel

from maintenance.models import Equipment

class MaintenanceAnalytics:
    """Generate analytics and insights"""
    
    def __init__(self, organization=None):
        self.organization = organization
    
    def get_equipment_health_score(self):
        """Calculate overall equipment health (0-100)"""
        
        equipment = Equipment.objects.all()
        if self.organization:
            equipment = equipment.filter(organization=self.organization)
        
        total = equipment.count()
        if total == 0:
            return 0
        
        operational = equipment.filter(status='operational').count()
        health_score = (operational / total) * 100
        
        return round(health_score, 2)
    
    def get_mtbf(self, equipment=None):
        """Mean Time Between Failures"""
        
        if equipment:
            requests = equipment.maintenance_requests.filter(
                request_type='corrective'
            ).order_by('created_at')
        else:
            requests = MaintenanceRequest.objects.filter(
                request_type='corrective'
            )
            if self.organization:
                requests = requests.filter(
                    equipment__organization=self.organization
                )
        
        if requests.count() < 2:
            return None
        
        time_diffs = []
        for i in range(1, len(requests)):
            diff = (requests[i].created_at - requests[i-1].created_at).days
            time_diffs.append(diff)
        
        return sum(time_diffs) / len(time_diffs) if time_diffs else None
    
    def get_mttr(self):
        """Mean Time To Repair"""
        
        completed = MaintenanceRequest.objects.filter(
            status='completed',
            completed_date__isnull=False
        )
        if self.organization:
            completed = completed.filter(
                equipment__organization=self.organization
            )
        
        repair_times = []
        for request in completed:
            if request.scheduled_date:
                repair_time = (
                    request.completed_date - request.scheduled_date
                ).total_seconds() / 3600  # Hours
                repair_times.append(repair_time)
        
        return sum(repair_times) / len(repair_times) if repair_times else None
    
    def get_trends(self, days=30):
        """Get maintenance trends over time"""
        
        start_date = timezone.now() - timedelta(days=days)
        
        requests = MaintenanceRequest.objects.filter(
            created_at__gte=start_date
        )
        if self.organization:
            requests = requests.filter(
                equipment__organization=self.organization
            )
        
        # Group by date
        daily_counts = requests.extra(
            select={'day': 'date(created_at)'}
        ).values('day').annotate(count=Count('id')).order_by('day')
        
        return list(daily_counts)
    
    def get_equipment_utilization(self):
        """Calculate equipment utilization rates"""
        
        equipment = Equipment.objects.all()
        if self.organization:
            equipment = equipment.filter(organization=self.organization)
        
        utilization = []
        for eq in equipment:
            maintenance_time = eq.maintenance_requests.filter(
                status='in_progress'
            ).aggregate(
                total_hours=Sum('actual_hours')
            )['total_hours'] or 0
            
            # Assume 24/7 operation
            total_hours = 24 * 30  # Last 30 days
            uptime = total_hours - maintenance_time
            
            utilization.append({
                'equipment': eq,
                'utilization_rate': (uptime / total_hours) * 100,
                'downtime_hours': maintenance_time
            })
        
        return utilization
    
    def predict_next_month_maintenance(self):
        """Predict maintenance load for next month"""
        
        # Get historical average
        last_30_days = MaintenanceRequest.objects.filter(
            created_at__gte=timezone.now() - timedelta(days=30)
        )
        if self.organization:
            last_30_days = last_30_days.filter(
                equipment__organization=self.organization
            )
        
        avg_per_day = last_30_days.count() / 30
        predicted_next_month = avg_per_day * 30
        
        # Get equipment needing maintenance soon
        equipment_due = Equipment.objects.filter(
            next_maintenance__lte=timezone.now() + timedelta(days=30),
            next_maintenance__gte=timezone.now()
        )
        if self.organization:
            equipment_due = equipment_due.filter(organization=self.organization)
        
        return {
            'predicted_requests': round(predicted_next_month),
            'equipment_due_soon': equipment_due.count(),
            'avg_daily_requests': round(avg_per_day, 2)
        }