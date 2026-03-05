from django.shortcuts import redirect
from django.contrib import messages

class RoleBasedAccessMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
    
    def __call__(self, request):
        # Admin-only paths
        admin_paths = ['/admin/', '/maintenance/ml-dashboard/', '/maintenance/train-model/']
        
        if request.user.is_authenticated:
            if any(request.path.startswith(path) for path in admin_paths):
                if not request.user.is_staff:
                    messages.error(request, 'Admin access required.')
                    return redirect('maintenance:dashboard')
        
        response = self.get_response(request)
        return response