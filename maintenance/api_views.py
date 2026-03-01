from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import Equipment, MaintenanceRequest, WorkCenter
from .serializers import (
    EquipmentSerializer, 
    MaintenanceRequestSerializer,
    WorkCenterSerializer
)
from .ml_models import PredictiveMaintenanceModel

class EquipmentViewSet(viewsets.ModelViewSet):
    queryset = Equipment.objects.all()
    serializer_class = EquipmentSerializer
    permission_classes = [permissions.IsAuthenticated]
    
    @action(detail=True, methods=['get'])
    def predict_maintenance(self, request, pk=None):
        """Predict next maintenance date for equipment"""
        equipment = self.get_object()
        predictor = PredictiveMaintenanceModel()
        
        prediction = predictor.predict(equipment)
        priority = predictor.get_maintenance_priority(equipment)
        
        return Response({
            'equipment_id': equipment.id,
            'days_until_maintenance': prediction,
            'priority_score': priority
        })

class MaintenanceRequestViewSet(viewsets.ModelViewSet):
    queryset = MaintenanceRequest.objects.all()
    serializer_class = MaintenanceRequestSerializer
    permission_classes = [permissions.IsAuthenticated]

class WorkCenterViewSet(viewsets.ModelViewSet):
    queryset = WorkCenter.objects.all()
    serializer_class = WorkCenterSerializer
    permission_classes = [permissions.IsAuthenticated]