from django.core.management.base import BaseCommand
from maintenance.ml_engine import RealMLEngine
from maintenance.models import Equipment

class Command(BaseCommand):
    help = 'Train ML model on equipment data'
    
    def handle(self, *args, **kwargs):
        self.stdout.write('Starting ML training...')
        
        engine = RealMLEngine()
        equipment = Equipment.objects.all()
        
        self.stdout.write(f'Training on {equipment.count()} equipment items...')
        
        result = engine.train_models(equipment)
        
        if result['success']:
            self.stdout.write(self.style.SUCCESS(f'\n✅ Training complete!'))
            self.stdout.write(f"Best Model: {result['best_model']}")
            self.stdout.write(f"MAE: {result['results'][result['best_model']]['mae']:.2f} days")
            self.stdout.write(f"R² Score: {result['results'][result['best_model']]['r2']:.4f}")
        else:
            self.stdout.write(self.style.ERROR(f"Training failed: {result.get('error')}"))