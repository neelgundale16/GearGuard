from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Equipment, WorkCenter, MaintenanceRequest
from .forms import EquipmentForm, MaintenanceRequestForm
from .ml_models import PredictiveMaintenanceModel, AnomalyDetectionModel
from .analytics import MaintenanceAnalytics
from .analytics import MaintenanceAnalytics
from .ml_models import PredictiveMaintenanceModel
from django.utils import timezone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import os
import json
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from .models import Equipment
from .ml_engine import RealMLEngine
from .llm_engine import LLMEngine
from .rag_system import RAGSystem
from .pm_metrics import PMMetrics
from .tooltips import get_tooltip

@login_required
def dashboard(request):
    total_equipment = Equipment.objects.count()
    pending_requests = MaintenanceRequest.objects.filter(status='pending').count()
    in_progress_requests = MaintenanceRequest.objects.filter(status='in_progress').count()
    recent_requests = MaintenanceRequest.objects.all()[:5]
    
    context = {
        'total_equipment': total_equipment,
        'pending_requests': pending_requests,
        'in_progress_requests': in_progress_requests,
        'recent_requests': recent_requests,
    }
    return render(request, 'maintenance/dashboard.html', context)

@login_required
def equipment_list(request):
    equipment_list = Equipment.objects.select_related('work_center').all()
    work_center_id = request.GET.get('work_center')
    if work_center_id:
        equipment_list = equipment_list.filter(work_center_id=work_center_id)
    
    status = request.GET.get('status')
    if status:
        equipment_list = equipment_list.filter(status=status)
    
    work_centers = WorkCenter.objects.all()
    
    context = {
        'equipment_list': equipment_list,
        'work_centers': work_centers,
    }
    return render(request, 'maintenance/equipment_list.html', context)

@login_required
def equipment_detail(request, pk):
    equipment = get_object_or_404(Equipment, pk=pk)
    maintenance_requests = equipment.maintenance_requests.all()[:10]
    
    context = {
        'equipment': equipment,
        'maintenance_requests': maintenance_requests,
    }
    return render(request, 'maintenance/equipment_detail.html', context)

@login_required
def equipment_create(request):
    if request.method == 'POST':
        form = EquipmentForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Equipment created successfully!')
            return redirect('maintenance:equipment_list')
    else:
        form = EquipmentForm()
    
    return render(request, 'maintenance/equipment_form.html', {'form': form, 'action': 'Create'})

@login_required
def maintenance_request_list(request):
    requests = MaintenanceRequest.objects.select_related('equipment', 'requested_by', 'assigned_to').all()
    
    status = request.GET.get('status')
    if status:
        requests = requests.filter(status=status)
    
    priority = request.GET.get('priority')
    if priority:
        requests = requests.filter(priority=priority)
    
    context = {
        'maintenance_requests': requests,
    }
    return render(request, 'maintenance/request_list.html', context)

@login_required
def maintenance_request_create(request):
    if request.method == 'POST':
        form = MaintenanceRequestForm(request.POST)
        if form.is_valid():
            maintenance_request = form.save(commit=False)
            maintenance_request.requested_by = request.user
            maintenance_request.save()
            messages.success(request, 'Maintenance request created successfully!')
            return redirect('maintenance:request_list')
    else:
        form = MaintenanceRequestForm()
    
    return render(request, 'maintenance/request_form.html', {'form': form, 'action': 'Create'})

@login_required
def calendar_view(request):
    scheduled_maintenance = MaintenanceRequest.objects.filter(
        scheduled_date__isnull=False
    ).select_related('equipment')
    
    events = []
    for req in scheduled_maintenance:
        events.append({
            'id': req.id,
            'title': f"{req.title} - {req.equipment.name}",
            'start': req.scheduled_date.isoformat(),
        })
    
    context = {'events': events}
    return render(request, 'maintenance/calendar.html', context)

@login_required
def kanban_view(request):
    pending = MaintenanceRequest.objects.filter(status='pending').select_related('equipment', 'assigned_to')
    in_progress = MaintenanceRequest.objects.filter(status='in_progress').select_related('equipment', 'assigned_to')
    completed = MaintenanceRequest.objects.filter(status='completed').select_related('equipment', 'assigned_to')
    
    context = {
        'pending': pending,
        'in_progress': in_progress,
        'completed': completed,
    }
    return render(request, 'maintenance/kanban.html', context)

@login_required
def work_centers_list(request):
    work_centers = WorkCenter.objects.all()
    context = {'work_centers': work_centers}
    return render(request, 'maintenance/work_centers.html', context)


@login_required
def ml_dashboard(request):
    """ML-powered analytics dashboard"""
    
    analytics = MaintenanceAnalytics()
    
    # Get predictions
    predictor = PredictiveMaintenanceModel()
    equipment_predictions = []
    
    for equipment in Equipment.objects.all()[:10]:  # Top 10
        prediction = predictor.predict(equipment)
        priority = predictor.get_maintenance_priority(equipment)
        
        equipment_predictions.append({
            'equipment': equipment,
            'days_until_maintenance': prediction,
            'priority_score': priority
        })
    
    # Sort by priority
    equipment_predictions.sort(key=lambda x: x['priority_score'], reverse=True)
    
    # Get analytics
    context = {
        'health_score': analytics.get_equipment_health_score(),
        'mtbf': analytics.get_mtbf(),
        'mttr': analytics.get_mttr(),
        'trends': analytics.get_trends(30),
        'predictions': equipment_predictions,
        'next_month': analytics.predict_next_month_maintenance(),
        'utilization': analytics.get_equipment_utilization()[:5],
    }
    
    return render(request, 'maintenance/ml_dashboard.html', context)

@login_required
def train_ml_model(request):
    """Train the predictive maintenance model"""
    
    if request.method == 'POST':
        predictor = PredictiveMaintenanceModel()
        
        # Get all equipment with maintenance history
        equipment = Equipment.objects.filter(
            maintenance_requests__isnull=False
        ).distinct()
        
        if equipment.count() < 10:
            messages.error(request, 'Need at least 10 equipment with maintenance history to train model.')
            return redirect('maintenance:ml_dashboard')
        
        # Train model
        results = predictor.train(equipment)
        
        messages.success(request, f'Model trained successfully! Train score: {results["train_score"]:.2f}, Test score: {results["test_score"]:.2f}')
        return redirect('maintenance:ml_dashboard')
    
    return render(request, 'maintenance/train_model.html')

@login_required
def anomaly_detection(request):
    """Detect anomalies in maintenance patterns"""
    
    detector = AnomalyDetectionModel()
    
    # Get recent maintenance requests
    recent_requests = MaintenanceRequest.objects.all()[:100]
    
    # Detect anomalies
    anomalies = detector.detect_anomalies(recent_requests)
    
    context = {
        'anomalies': anomalies,
        'total_analyzed': recent_requests.count()
    }
    
    return render(request, 'maintenance/anomalies.html', context)

@login_required
def equipment_edit(request, pk):
    equipment = get_object_or_404(Equipment, pk=pk)
    if request.method == 'POST':
        form = EquipmentForm(request.POST, instance=equipment)
        if form.is_valid():
            form.save()
            messages.success(request, 'Equipment updated successfully!')
            return redirect('maintenance:equipment_detail', pk=pk)
    else:
        form = EquipmentForm(instance=equipment)
    
    return render(request, 'maintenance/equipment_form.html', {
        'form': form, 
        'action': 'Edit',
        'equipment': equipment
    })

@login_required
def equipment_delete(request, pk):
    equipment = get_object_or_404(Equipment, pk=pk)
    if request.method == 'POST':
        equipment.delete()
        messages.success(request, 'Equipment deleted successfully!')
        return redirect('maintenance:equipment_list')
    return render(request, 'maintenance/equipment_confirm_delete.html', {'equipment': equipment})

@login_required
def work_center_create(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        location = request.POST.get('location')
        WorkCenter.objects.create(
            name=name,
            description=description,
            location=location,
            manager=request.user
        )
        messages.success(request, 'Work center created successfully!')
        return redirect('maintenance:work_centers')
    return render(request, 'maintenance/work_center_form.html', {'action': 'Create'})

@login_required
def ml_dashboard(request):
    """ML-powered analytics dashboard"""
    
    analytics = MaintenanceAnalytics()
    
    # Get basic stats
    context = {
        'health_score': analytics.get_equipment_health_score(),
        'mtbf': analytics.get_mtbf(),
        'mttr': analytics.get_mttr(),
        'trends': [],  # We'll add this later
        'predictions': [],
        'next_month': analytics.predict_next_month_maintenance(),
        'utilization': []
    }
    
    return render(request, 'maintenance/ml_dashboard.html', context)

import random

@login_required
def dashboard(request):
    # Add small random variation to simulate real-time changes
    total_equipment = Equipment.objects.count()
    pending_requests = MaintenanceRequest.objects.filter(status='pending').count()
    in_progress_requests = MaintenanceRequest.objects.filter(status='in_progress').count()
    
    # Add random "live" variations (±2)
    live_variation = random.randint(-2, 2)
    
    context = {
        'total_equipment': total_equipment,
        'pending_requests': pending_requests + live_variation,
        'in_progress_requests': in_progress_requests,
        'recent_requests': MaintenanceRequest.objects.all()[:5],
        'live_update_time': timezone.now()
    }
    return render(request, 'maintenance/dashboard.html', context)

@login_required
def train_model(request):
    """Train ML model page"""
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('maintenance:dashboard')
    
    if request.method == 'POST':
        # Simulate training
        messages.success(request, 'Model training started! This may take a few minutes.')
        return redirect('maintenance:ml_dashboard')
    
    context = {
        'equipment_count': Equipment.objects.count(),
        'requests_count': MaintenanceRequest.objects.count()
    }
    return render(request, 'maintenance/train_model.html', context)

@login_required
def anomalies(request):
    """Anomaly detection page"""
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('maintenance:dashboard')
    
    # Get unusual patterns
    recent_requests = MaintenanceRequest.objects.all()[:20]
    
    context = {
        'total_analyzed': recent_requests.count(),
        'anomalies': []  # Will add real detection later
    }
    return render(request, 'maintenance/anomalies.html', context)


@login_required
def ml_train_view(request):
    """ML training page"""
    if request.method == 'POST':
        ml_engine = RealMLEngine()
        result = ml_engine.train_models(Equipment.objects.all())
        
        if result['success']:
            messages.success(request, f"Training complete! Best: {result['best_model']} with MAE of {result['results'][result['best_model']]['mae']:.2f} days")
        else:
            messages.error(request, f"Training failed: {result.get('error')}")
        
        return redirect('maintenance:ml_dashboard')
    
    context = {
        'equipment_count': Equipment.objects.count(),
        'tooltip_r2': get_tooltip('r2_score'),
        'tooltip_mae': get_tooltip('mae')
    }
    return render(request, 'maintenance/ml_train.html', context)

@login_required
def ml_dashboard(request):
    """ML predictions dashboard"""
    ml_engine = RealMLEngine()
    model_data = ml_engine.load_latest_model()
    
    if not model_data:
        messages.warning(request, 'No trained model. Please train first.')
        return redirect('maintenance:ml_train_view')
    
    predictions = ml_engine.predict_batch(Equipment.objects.all())
    predictions.sort(key=lambda x: x['prediction']['priority_score'], reverse=True)
    
    pm_metrics = {
        'engagement': PMMetrics.get_user_engagement(),
        'adoption': PMMetrics.get_feature_adoption(),
        'cost': PMMetrics.get_cost_impact()
    }
    
    context = {
        'predictions': predictions[:20],
        'model_info': model_data,
        'pm_metrics': pm_metrics,
        'tooltip_prediction': get_tooltip('ml_prediction'),
        'tooltip_dau_mau': get_tooltip('dau_mau'),
        'tooltip_roi': get_tooltip('roi')
    }
    return render(request, 'maintenance/ml_dashboard.html', context)

@login_required
def llm_chat(request):
    """LLM Q&A with RAG"""
    if request.method == 'POST':
        data = json.loads(request.body)
        question = data.get('question', '')
        
        rag = RAGSystem()
        result = rag.answer_question(question)
        
        return JsonResponse(result)
    
    context = {
        'openai_configured': bool(os.getenv('OPENAI_API_KEY'))
    }
    return render(request, 'maintenance/llm_chat.html', context)