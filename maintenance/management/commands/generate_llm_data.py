from django.core.management.base import BaseCommand
import random
from datetime import datetime, timedelta
from maintenance.models import Equipment, WorkCenter, MaintenanceRequest
from django.contrib.auth.models import User
from django.utils import timezone

class Command(BaseCommand):
    help = 'Generate realistic maintenance data - NO SENSORS'
    
    def handle(self, *args, **kwargs):
        self.stdout.write("=" * 80)
        self.stdout.write(self.style.SUCCESS("GENERATING MAINTENANCE DATASET"))
        self.stdout.write("=" * 80)
        
        random.seed(42)
        
        # Work Centers
        self.stdout.write("\n[1/4] Creating Work Centers...")
        work_centers = [
            ('Production Line A', 'Building 1'),
            ('Production Line B', 'Building 2'),
            ('Assembly Station', 'Building 3'),
            ('Quality Control', 'Building 4'),
            ('Packaging Unit', 'Building 5')
        ]
        
        wc_objects = []
        for name, loc in work_centers:
            wc, _ = WorkCenter.objects.get_or_create(name=name, defaults={'location': loc})
            wc_objects.append(wc)
            self.stdout.write(f"  ✓ {name}")
        
        # Users
        self.stdout.write("\n[2/4] Creating Users...")
        admin, created = User.objects.get_or_create(
            username='admin',
            defaults={'is_staff': True, 'is_superuser': True, 'email': 'admin@gearguard.com'}
        )
        if created:
            admin.set_password('admin123')
            admin.save()
            self.stdout.write("  ✓ Admin (admin/admin123)")
        
        tech_names = [
            ('Alex', 'Martinez'), ('Jamie', 'Chen'), ('Sam', 'Patel'),
            ('Taylor', 'Kim'), ('Jordan', 'Singh'), ('Casey', 'Lee')
        ]
        
        technicians = []
        for i, (first, last) in enumerate(tech_names, 1):
            tech, created = User.objects.get_or_create(
                username=f'tech{i}',
                defaults={'first_name': first, 'last_name': last, 'email': f'{first.lower()}@gearguard.com'}
            )
            if created:
                tech.set_password('tech123')
                tech.save()
            technicians.append(tech)
        
        self.stdout.write(f"  ✓ {len(technicians)} technicians")
        
        # Equipment
        self.stdout.write("\n[3/4] Generating Equipment...")
        
        equipment_types = [
            {
                'type': 'CNC Machine', 'prefix': 'CNC', 'count': 30,
                'issues': [
                    'Spindle motor failure', 'Coolant pump malfunction', 'Tool changer error',
                    'Control system freeze', 'Servo motor overheating', 'Chuck gripping issue'
                ]
            },
            {
                'type': 'Conveyor Belt', 'prefix': 'CVB', 'count': 25,
                'issues': [
                    'Belt wear and tear', 'Motor bearing failure', 'Belt misalignment',
                    'Roller damage', 'Drive belt slipping', 'Emergency stop malfunction'
                ]
            },
            {
                'type': 'Robotic Arm', 'prefix': 'ROB', 'count': 20,
                'issues': [
                    'Joint stiffness', 'Gripper malfunction', 'Position accuracy loss',
                    'Cable damage', 'Power supply fluctuation', 'Software error'
                ]
            },
            {
                'type': 'Hydraulic Press', 'prefix': 'HYP', 'count': 15,
                'issues': [
                    'Hydraulic seal leak', 'Pressure drop', 'Valve malfunction',
                    'Oil contamination', 'Cylinder rod damage', 'Control panel error'
                ]
            },
            {
                'type': 'Air Compressor', 'prefix': 'ACP', 'count': 25,
                'issues': [
                    'Compressor overheating', 'Air filter clogged', 'Pressure valve stuck',
                    'Oil leak', 'Motor failure', 'Excessive noise'
                ]
            }
        ]
        
        equipment_list = []
        eq_id = 1
        
        for et in equipment_types:
            for i in range(et['count']):
                days_old = random.randint(30, 300)
                health = max(20, 100 - (days_old / 250 * 100))
                
                status = 'maintenance' if health < 40 else 'operational'
                
                eq = Equipment.objects.create(
                    name=f"{et['type']} {eq_id}",
                    equipment_id=f"{et['prefix']}{eq_id:04d}",
                    description=f"{et['type']} - Health: {health:.1f}%",
                    work_center=random.choice(wc_objects),
                    status=status,
                    purchase_date=timezone.now().date() - timedelta(days=days_old),
                    last_maintenance=timezone.now().date() - timedelta(days=random.randint(5, 45)),
                    next_maintenance=timezone.now().date() + timedelta(days=int(30 * health / 100))
                )
                
                equipment_list.append({
                    'equipment': eq,
                    'type': et['type'],
                    'issues': et['issues'],
                    'days_old': days_old,
                    'health': health
                })
                
                eq_id += 1
                if eq_id % 20 == 0:
                    self.stdout.write(f"  ✓ {eq_id-1} equipment...")
        
        self.stdout.write(f"\n  ✓ TOTAL: {len(equipment_list)}")
        
        # Maintenance History
        self.stdout.write("\n[4/4] Generating Maintenance History...")
        
        total = 0
        priority_dist = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for ed in equipment_list:
            eq, days_old = ed['equipment'], ed['days_old']
            num_maint = max(3, int(days_old / 40))
            
            for m in range(num_maint):
                days_ago = days_old - (m * (days_old // num_maint))
                
                # Select random issue
                issue = random.choice(ed['issues'])
                
                # Determine severity
                if ed['health'] < 40 or random.random() < 0.15:
                    priority = 'critical'
                    req_type = 'emergency'
                    impact = 'Production stopped. Immediate attention required.'
                elif ed['health'] < 60 or random.random() < 0.25:
                    priority = 'high'
                    req_type = 'corrective'
                    impact = 'Performance degraded. Schedule repair within 24 hours.'
                elif ed['health'] < 80:
                    priority = 'medium'
                    req_type = 'corrective'
                    impact = 'Minor issues detected. Address in next maintenance window.'
                else:
                    priority = 'low'
                    req_type = 'preventive'
                    impact = 'Routine preventive maintenance.'
                
                description = f"""
ISSUE: {issue}

EQUIPMENT: {ed['type']}
AGE: {days_old} days
HEALTH SCORE: {ed['health']:.1f}%

IMPACT: {impact}

OBSERVED SYMPTOMS:
- {random.choice(['Unusual noise during operation', 'Reduced efficiency', 'Frequent stops', 'Error messages on display', 'Physical damage visible', 'Abnormal operation'])}
- {random.choice(['Safety concern reported', 'Quality issues in output', 'Increased energy consumption', 'Intermittent failures', 'Operator complaints', 'Maintenance indicator activated'])}

ROOT CAUSE: {random.choice([
    'Component wear due to age and usage',
    'Lack of proper lubrication',
    'Environmental factors (dust, temperature)',
    'Incorrect operation or overload',
    'Manufacturing defect in part',
    'End of component lifecycle'
])}

RECOMMENDED ACTION:
{random.choice([
    'Replace worn component immediately',
    'Perform complete system inspection',
    'Adjust operational parameters',
    'Clean and lubricate mechanism',
    'Recalibrate control systems',
    'Schedule component replacement'
])}
"""
                
                notes = f"{req_type.title()} maintenance completed. " + random.choice([
                    f"Replaced faulty component. Tested successfully. Downtime: {random.randint(1, 8)}h. Cost: ${random.randint(200, 5000)}.",
                    f"Adjusted and recalibrated. Performance restored. Downtime: {random.randint(1, 4)}h. Cost: ${random.randint(150, 2000)}.",
                    f"Cleaned and serviced. Running normally. Downtime: {random.randint(1, 3)}h. Cost: ${random.randint(100, 1000)}."
                ])
                
                req_date = timezone.now() - timedelta(days=days_ago)
                
                MaintenanceRequest.objects.create(
                    equipment=eq,
                    title=issue[:100],
                    description=description.strip(),
                    request_type=req_type,
                    priority=priority,
                    status='completed',
                    assigned_to=random.choice(technicians),
                    scheduled_date=req_date.date(),
                    estimated_hours=random.uniform(1, 12),
                    created_at=req_date,
                    notes=notes
                )
                
                total += 1
                priority_dist[priority] += 1
            
            if total % 500 == 0:
                self.stdout.write(f"  ✓ {total} records...")
        
        # Stats
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write(self.style.SUCCESS("✅ COMPLETE!"))
        self.stdout.write("=" * 80)
        self.stdout.write(f"\n• Equipment: {len(equipment_list)}")
        self.stdout.write(f"• Maintenance Records: {total}")
        self.stdout.write(f"• Work Centers: {len(wc_objects)}")
        
        self.stdout.write("\nPRIORITY DISTRIBUTION:")
        for p, c in priority_dist.items():
            pct = (c / total * 100) if total > 0 else 0
            self.stdout.write(f"  • {p.upper():<10} {c:>5} ({pct:.1f}%)")
        
        self.stdout.write("\n✅ Run: python manage.py shell")
        self.stdout.write(">>> from maintenance.ml_engine import RealMLEngine")
        self.stdout.write(">>> from maintenance.models import Equipment")
        self.stdout.write(">>> engine = RealMLEngine()")
        self.stdout.write(">>> result = engine.train_models(Equipment.objects.all())")
        self.stdout.write("=" * 80)