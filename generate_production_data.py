import os
import django
import random
from datetime import datetime, timedelta
import requests
import json

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gearguard_project.settings')
django.setup()

from django.contrib.auth.models import User
from maintenance.models import Equipment, WorkCenter, MaintenanceRequest
from django.utils import timezone

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
LLAMA_MODEL = os.getenv('LLAMA_MODEL', 'llama3.2')

def call_llama(prompt):
    """Call Llama via Ollama"""
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        response = requests.post(url, json={
            "model": LLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }, timeout=60)
        return response.json()['response']
    except Exception as e:
        print(f"Llama error: {e}")
        return None

def generate_equipment_types():
    """Generate realistic equipment types using Llama"""
    prompt = """Generate EXACTLY 20 industrial equipment types.
Return ONLY a Python list like: ['CNC Machine', 'Hydraulic Press', ...]
No explanations, just the list."""
    
    response = call_llama(prompt)
    if response:
        try:
            equipment_types = eval(response.strip())
            if isinstance(equipment_types, list) and len(equipment_types) >= 20:
                return equipment_types[:20]
        except:
            pass
    
    # Fallback if Llama fails
    return [
        'CNC Lathe', 'Hydraulic Press', 'Assembly Robot', 'Conveyor Belt',
        'Industrial Oven', 'Paint Booth', 'Welding Station', 'Grinding Machine',
        '3D Printer', 'Injection Molding Machine', 'Laser Cutter', 'Plasma Cutter',
        'Forklift', 'Overhead Crane', 'Packaging Machine', 'Quality Scanner',
        'Air Compressor', 'Water Pump', 'Cooling Tower', 'Generator'
    ]

def generate_failure_patterns():
    """Generate realistic failure patterns using Llama"""
    prompt = """Generate EXACTLY 30 equipment failure patterns.
Return ONLY a Python list like: ['Bearing wear', 'Seal leak', ...]
No explanations, just the list."""
    
    response = call_llama(prompt)
    if response:
        try:
            patterns = eval(response.strip())
            if isinstance(patterns, list) and len(patterns) >= 30:
                return patterns[:30]
        except:
            pass
    
    # Fallback
    return [
        'Bearing wear', 'Seal leak', 'Belt misalignment', 'Overheating',
        'Vibration excessive', 'Lubrication failure', 'Electrical fault',
        'Sensor malfunction', 'Hydraulic pressure drop', 'Motor burnout',
        'Coupling failure', 'Gear tooth damage', 'Shaft misalignment',
        'Corrosion damage', 'Fatigue crack', 'Bolt loosening',
        'Filter clog', 'Valve stuck', 'Hose rupture', 'Gasket failure',
        'Circuit breaker trip', 'Fuse blown', 'Wiring damage',
        'Software error', 'Calibration drift', 'Contamination',
        'Wear plate erosion', 'Coolant leak', 'Oil contamination', 'Chain stretch'
    ]

def create_users():
    """Create admin and technician users"""
    print("Creating users...")
    
    # Admin
    if not User.objects.filter(username='admin').exists():
        User.objects.create_superuser('admin', 'admin@gearguard.com', 'admin123')
        print("✓ Admin created")
    
    # Technicians
    technicians = []
    tech_names = [
        'john_tech', 'sarah_maint', 'mike_eng', 'lisa_tech', 'david_mech',
        'emma_elec', 'james_hydr', 'olivia_auto', 'william_fab', 'sophia_qual',
        'robert_weld', 'ava_paint', 'michael_assy', 'isabella_pack', 'daniel_ship'
    ]
    
    for name in tech_names:
        if not User.objects.filter(username=name).exists():
            user = User.objects.create_user(
                username=name,
                email=f"{name}@gearguard.com",
                password='tech123',
                first_name=name.split('_')[0].capitalize(),
                last_name='Technician'
            )
            technicians.append(user)
    
    print(f"✓ {len(technicians)} technicians created")
    return User.objects.filter(username__in=tech_names)

def create_work_centers():
    """Create work centers"""
    print("Creating work centers...")
    
    centers = [
        ('Machining', 'CNC and precision machining area', 'Building A'),
        ('Assembly', 'Final product assembly line', 'Building B'),
        ('Welding', 'Welding and fabrication zone', 'Building C'),
        ('Quality Control', 'Testing and inspection area', 'Building A'),
        ('Packaging', 'Product packaging and shipping prep', 'Building D'),
    ]
    
    work_centers = []
    admin = User.objects.get(username='admin')
    
    for name, desc, loc in centers:
        wc, created = WorkCenter.objects.get_or_create(
            name=name,
            defaults={'description': desc, 'location': loc, 'manager': admin}
        )
        work_centers.append(wc)
    
    print(f"✓ {len(work_centers)} work centers created")
    return work_centers

def create_equipment_and_maintenance(equipment_types, failure_patterns, work_centers, technicians):
    """Create equipment with REALISTIC NOISY maintenance data"""
    print("Generating equipment and maintenance records with REAL NOISE...")
    
    equipment_count = 200
    created_equipment = []
    
    for i in range(1, equipment_count + 1):
        eq_type = random.choice(equipment_types)
        eq_number = random.randint(1, 200)
        
        # Random purchase date (1-5 years ago)
        purchase_date = timezone.now() - timedelta(days=random.randint(365, 1825))
        
        equipment = Equipment.objects.create(
            name=f"{eq_type} {eq_number}",
            equipment_id=f"EQ{i:05d}",
            equipment_type=eq_type,
            manufacturer=random.choice(['Siemens', 'ABB', 'Fanuc', 'Mitsubishi', 'Bosch', 'Schneider']),
            model_number=f"MDL-{random.randint(1000, 9999)}",
            serial_number=f"SN{random.randint(100000, 999999)}",
            purchase_date=purchase_date,
            status=random.choice(['operational'] * 7 + ['maintenance'] * 2 + ['down'] * 1),
            work_center=random.choice(work_centers),
            location=f"Bay {random.randint(1, 20)}-{random.randint(1, 10)}"
        )
        created_equipment.append(equipment)
    
    print(f"✓ {len(created_equipment)} equipment created")
    
    # Create REALISTIC NOISY maintenance records
    print("Creating REALISTIC maintenance records with NOISE...")
    
    total_records = 0
    technician_list = list(technicians)
    
    for equipment in created_equipment:
        # Random number of maintenance events (2-15 per equipment)
        num_maintenances = random.randint(2, 15)
        
        # Start from purchase date
        current_date = equipment.purchase_date
        
        for j in range(num_maintenances):
            # ADD REALISTIC NOISE - days between maintenance is NOT constant!
            # Base interval: 20-60 days
            base_interval = random.randint(20, 60)
            
            # Add RANDOM VARIATION (±50% - this is the KEY to avoid overfitting)
            noise_factor = random.uniform(0.5, 1.5)
            actual_interval = int(base_interval * noise_factor)
            
            # Add occasional LONG gaps (unexpected delays)
            if random.random() < 0.15:  # 15% chance of long delay
                actual_interval += random.randint(30, 90)
            
            # Add occasional EMERGENCY maintenance (very short interval)
            if random.random() < 0.10:  # 10% chance of emergency
                actual_interval = random.randint(1, 10)
            
            current_date += timedelta(days=actual_interval)
            
            # Stop if we've gone past today
            if current_date > timezone.now():
                break
            
            # Random failure pattern
            failure = random.choice(failure_patterns)
            
            # Determine request type and priority with REALISTIC VARIATION
            if actual_interval < 15:  # Emergency
                request_type = 'emergency'
                priority = random.choice(['critical'] * 3 + ['high'])  # Mostly critical, sometimes high
            elif actual_interval < 30:  # Corrective
                request_type = 'corrective'
                priority = random.choice(['high'] * 2 + ['medium'])
            else:  # Preventive
                request_type = 'preventive'
                priority = random.choice(['medium'] * 3 + ['low'])
            
            # Create maintenance request
            request = MaintenanceRequest.objects.create(
                equipment=equipment,
                title=f"{failure} - {equipment.name}",
                description=f"Maintenance required: {failure}. Equipment showing signs of wear. {random.choice(['Routine check', 'Urgent repair', 'Scheduled maintenance', 'Unexpected failure'])}.",
                request_type=request_type,
                priority=priority,
                status='completed',  # Mark as completed for training
                requested_by=random.choice(technician_list),
                assigned_to=random.choice(technician_list),
                created_at=current_date,
                scheduled_date=current_date + timedelta(days=random.randint(1, 3)),
                completed_date=current_date + timedelta(days=random.randint(1, 7)),
                estimated_hours=random.uniform(1, 8),
                actual_hours=random.uniform(1, 10)
            )
            total_records += 1
    
    print(f"✓ {total_records} REALISTIC maintenance records created with NOISE")
    return created_equipment

def main():
    print("=" * 60)
    print("GearGuard Production Data Generation with REAL NOISE")
    print("=" * 60)
    
    # Clear existing data
    print("\nClearing existing data...")
    MaintenanceRequest.objects.all().delete()
    Equipment.objects.all().delete()
    WorkCenter.objects.all().delete()
    User.objects.filter(is_superuser=False).delete()
    print("✓ Data cleared")
    
    # Generate templates using Llama
    print("\n[1/5] Generating equipment types with Llama...")
    equipment_types = generate_equipment_types()
    print(f"✓ {len(equipment_types)} equipment types generated")
    
    print("\n[2/5] Generating failure patterns with Llama...")
    failure_patterns = generate_failure_patterns()
    print(f"✓ {len(failure_patterns)} failure patterns generated")
    
    # Create base data
    print("\n[3/5] Creating users...")
    technicians = create_users()
    
    print("\n[4/5] Creating work centers...")
    work_centers = create_work_centers()
    
    print("\n[5/5] Creating equipment and REALISTIC maintenance records...")
    equipment = create_equipment_and_maintenance(
        equipment_types, failure_patterns, work_centers, technicians
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("PRODUCTION DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Equipment: {Equipment.objects.count()}")
    print(f"Maintenance Records: {MaintenanceRequest.objects.count()}")
    print(f"Users: {User.objects.count()}")
    print(f"Work Centers: {WorkCenter.objects.count()}")
    print("\nLogin credentials:")
    print("  Admin: admin / admin123")
    print("  Tech:  john_tech / tech123")
    print("\nNOTE: Data includes REALISTIC NOISE to prevent overfitting!")
    print("Expected ML accuracy: 70-85% (GOOD for real-world data)")
    print("=" * 60)

if __name__ == '__main__':
    main()