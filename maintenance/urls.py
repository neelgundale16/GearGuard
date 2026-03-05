from django.urls import path
from . import views

app_name = 'maintenance'

urlpatterns = [
    # Main Dashboard
    path('', views.dashboard, name='dashboard'),
    
    # Equipment Management
    path('equipment/', views.equipment_list, name='equipment_list'),
    path('equipment/create/', views.equipment_create, name='equipment_create'),
    path('equipment/<int:pk>/', views.equipment_detail, name='equipment_detail'),
    path('equipment/<int:pk>/edit/', views.equipment_edit, name='equipment_edit'),
    path('equipment/<int:pk>/delete/', views.equipment_delete, name='equipment_delete'),
    
    # Maintenance Requests
    path('requests/', views.maintenance_request_list, name='request_list'),
    path('requests/create/', views.maintenance_request_create, name='request_create'),
    
    # Work Centers
    path('work-centers/', views.work_centers_list, name='work_centers'),
    path('work-centers/create/', views.work_center_create, name='work_center_create'),
    
    # Views
    path('calendar/', views.calendar_view, name='calendar'),
    path('kanban/', views.kanban_view, name='kanban'),
    
    # ML & Analytics
    path('ml-dashboard/', views.ml_dashboard, name='ml_dashboard'),
    path('train-model/', views.train_model, name='train_model'),
    
    # PM Metrics
    path('pm-dashboard/', views.pm_dashboard, name='pm_dashboard'),
]