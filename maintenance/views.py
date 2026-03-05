from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.db.models import Q, Count
from django.utils import timezone
from datetime import timedelta

from .models import Equipment, WorkCenter, MaintenanceRequest
from .forms import EquipmentForm, MaintenanceRequestForm
from .ml_engine import RealMLEngine
from .pm_metrics import PMMetrics

@login_required
def dashboard(request):
    """Main dashboard with REAL DYNAMIC data"""
    
    total_equipment = Equipment.objects.count()
    pending_requests = MaintenanceRequest.objects.filter(
        Q(status='open') | Q(status='pending')
    ).count()
    in_progress_requests = MaintenanceRequest.objects.filter(status='in_progress').count()
    
    recent_requests = MaintenanceRequest.objects.select_related(
        'equipment', 'requested_by'
    ).order_by('-created_at')[:5]
    
    context = {
        'total_equipment': total_equipment,
        'pending_requests': pending_requests,
        'in_progress_requests': in_progress_requests,
        'recent_requests': recent_requests,
    }
    
    return render(request, 'maintenance/dashboard.html', context)

@login_required
def ml_dashboard(request):
    """Analytics Dashboard with REAL ML predictions"""
    
    ml_engine = RealMLEngine()
    model_data = ml_engine.load_latest_model()
    
    if not model_data:
        context = {
            'model_exists': False,
            'equipment_count': Equipment.objects.count(),
        }
        return render(request, 'maintenance/ml_dashboard.html', context)
    
    equipment_list = Equipment.objects.all()
    predictions = []
    
    for eq in equipment_list:
        pred = ml_engine.predict(eq)
        if pred:
            predictions.append({
                'equipment': eq,
                'prediction': pred
            })
    
    predictions.sort(key=lambda x: x['prediction']['days_until_maintenance'])
    
    context = {
        'model_exists': True,
        'model_info': {
            'model_name': f"Ensemble ({model_data.get('best_model', 'Unknown')})",
            'r2': model_data.get('r2', 0) * 100,
            'cv_r2': model_data.get('cv_r2', 0) * 100,
            'mae': model_data.get('mae', 0),
            'training_samples': model_data.get('training_samples', 0),
            'trained_date': model_data.get('trained_date', 'Unknown')
        },
        'predictions': predictions[:20],
        'total_equipment': equipment_list.count(),
    }
    
    return render(request, 'maintenance/ml_dashboard.html', context)

@login_required
def pm_dashboard(request):
    """PM Metrics Dashboard"""
    metrics = PMMetrics.get_complete_metrics()
    return render(request, 'maintenance/pm_dashboard.html', {'metrics': metrics})

@login_required
def kanban_view(request):
    """Kanban board"""
    
    pending = MaintenanceRequest.objects.filter(
        Q(status='open') | Q(status='pending')
    ).select_related('equipment').order_by('-priority', '-created_at')
    
    in_progress = MaintenanceRequest.objects.filter(
        status='in_progress'
    ).select_related('equipment').order_by('-priority', '-created_at')
    
    completed = MaintenanceRequest.objects.filter(
        status='completed'
    ).select_related('equipment').order_by('-created_at')[:20]
    
    context = {
        'pending': pending,
        'in_progress': in_progress,
        'completed': completed,
    }
    
    return render(request, 'maintenance/kanban.html', context)

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
    maintenance_requests = equipment.maintenance_requests.all().order_by('-created_at')[:10]
    
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
def maintenance_request_list(request):
    requests = MaintenanceRequest.objects.select_related(
        'equipment', 'requested_by', 'assigned_to'
    ).all().order_by('-created_at')
    
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
def work_centers_list(request):
    work_centers = WorkCenter.objects.all()
    context = {'work_centers': work_centers}
    return render(request, 'maintenance/work_centers.html', context)

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
def train_model(request):
    """Train ML model - REAL TRAINING"""
    if not request.user.is_staff:
        messages.error(request, 'Admin access required.')
        return redirect('maintenance:dashboard')
    
    if request.method == 'POST':
        ml_engine = RealMLEngine()
        equipment = Equipment.objects.all()
        
        equipment_with_history = []
        for eq in equipment:
            if eq.maintenance_requests.filter(status='completed').count() >= 2:
                equipment_with_history.append(eq)
        
        if len(equipment_with_history) < 20:
            messages.error(
                request, 
                f'❌ Need at least 20 equipment with 2+ completed maintenance records. '
                f'Currently have {len(equipment_with_history)}.'
            )
            return redirect('maintenance:ml_dashboard')
        
        result = ml_engine.train_models(equipment)
        
        if result['success']:
            best_model = result['best_model']
            r2 = result['results'][best_model]['r2']
            mae = result['results'][best_model]['mae']
            cv_r2 = result['results'][best_model]['cv_r2']
            
            messages.success(
                request,
                f"✅ Training Complete! "
                f"Best Model: {best_model} | "
                f"Test R²: {r2*100:.1f}% | "
                f"CV R²: {cv_r2*100:.1f}% | "
                f"MAE: {mae:.1f} days"
            )
        else:
            messages.error(request, f"❌ Training failed: {result.get('error')}")
        
        return redirect('maintenance:ml_dashboard')
    
    equipment_count = Equipment.objects.count()
    
    equipment_with_history = 0
    for eq in Equipment.objects.all():
        if eq.maintenance_requests.filter(status='completed').count() >= 2:
            equipment_with_history += 1
    
    requests_count = MaintenanceRequest.objects.filter(status='completed').count()
    
    context = {
        'equipment_count': equipment_count,
        'equipment_with_history': equipment_with_history,
        'requests_count': requests_count,
        'ready_to_train': equipment_with_history >= 20
    }
    return render(request, 'maintenance/train_model.html', context)