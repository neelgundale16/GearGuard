from django.contrib import admin
from .models import Equipment, WorkCenter, MaintenanceRequest

@admin.register(WorkCenter)
class WorkCenterAdmin(admin.ModelAdmin):
    list_display = ['name', 'location', 'manager', 'created_at']
    search_fields = ['name', 'location']

@admin.register(Equipment)
class EquipmentAdmin(admin.ModelAdmin):
    list_display = ['name', 'equipment_id', 'work_center', 'status', 'next_maintenance']
    list_filter = ['status', 'work_center']
    search_fields = ['name', 'equipment_id']

@admin.register(MaintenanceRequest)
class MaintenanceRequestAdmin(admin.ModelAdmin):
    list_display = ['title', 'equipment', 'priority', 'status', 'requested_by', 'scheduled_date']
    list_filter = ['status', 'priority', 'request_type']
    search_fields = ['title', 'description']