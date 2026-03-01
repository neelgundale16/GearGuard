from django.urls import path
from . import views

app_name = 'maintenance'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('equipment/', views.equipment_list, name='equipment_list'),
    path('equipment/create/', views.equipment_create, name='equipment_create'),
    path('equipment/<int:pk>/', views.equipment_detail, name='equipment_detail'),
    path('requests/', views.maintenance_request_list, name='request_list'),
    path('requests/create/', views.maintenance_request_create, name='request_create'),
    path('calendar/', views.calendar_view, name='calendar'),
    path('kanban/', views.kanban_view, name='kanban'),
    path('work-centers/', views.work_centers_list, name='work_centers'),
    path('equipment/<int:pk>/edit/', views.equipment_edit, name='equipment_edit'),
    path('equipment/<int:pk>/delete/', views.equipment_delete, name='equipment_delete'),
    path('work-centers/create/', views.work_center_create, name='work_center_create'),
    path('ml-dashboard/', views.ml_dashboard, name='ml_dashboard'),
    path('train-model/', views.train_ml_model, name='train_model'),
    path('anomalies/', views.anomaly_detection, name='anomalies'),
    path('ml-dashboard/', views.ml_dashboard, name='ml_dashboard'),
    path('ml-train/', views.ml_train_view, name='ml_train_view'),
    path('ml-dashboard/', views.ml_dashboard, name='ml_dashboard'),
    path('llm-chat/', views.llm_chat, name='llm_chat'),
]