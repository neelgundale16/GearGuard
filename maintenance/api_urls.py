from rest_framework.routers import DefaultRouter
from .api_views import (
    EquipmentViewSet,
    MaintenanceRequestViewSet,
    WorkCenterViewSet
)

router = DefaultRouter()
router.register(r'equipment', EquipmentViewSet)
router.register(r'requests', MaintenanceRequestViewSet)
router.register(r'workcenters', WorkCenterViewSet)

urlpatterns = router.urls