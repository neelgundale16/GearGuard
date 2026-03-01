from rest_framework import serializers
from .models import Equipment, MaintenanceRequest, WorkCenter

class EquipmentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Equipment
        fields = '__all__'

class MaintenanceRequestSerializer(serializers.ModelSerializer):
    equipment_name = serializers.CharField(source='equipment.name', read_only=True)
    
    class Meta:
        model = MaintenanceRequest
        fields = '__all__'

class WorkCenterSerializer(serializers.ModelSerializer):
    equipment_count = serializers.IntegerField(
        source='equipment.count', 
        read_only=True
    )
    
    class Meta:
        model = WorkCenter
        fields = '__all__'