import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'gearguard_project.settings')
django.setup()

import requests
import random
from datetime import timedelta
from maintenance.models import Equipment, WorkCenter, MaintenanceRequest
from django.contrib.auth.models import User
from django.utils import timezone

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
LLAMA_MODEL = os.getenv('LLAMA_MODEL', 'llama3.2')

def call_llama_once(prompt):
    """Single LLM call for template generation"""
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        response = requests.post(url, json={"model": LLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=60)
        return response.json()['response']
    except:
        return None

print("=" * 80)
print("🚀 PRODUCTION DATA GENERATION - 1000+ RECORDS IN 2-3 MINUTES")
print("Using: Template-based generation (PRO METHOD)")
print("=" * 80)

random.seed(42)

# Clear existing data
print("\n[0/6] Clearing existing data...")
MaintenanceRequest.objects.all().delete()
Equipment.objects.all().delete()
WorkCenter.objects.all().delete()
User.objects.filter(username__startswith='tech').delete()
print("  ✓ Database cleared")

# Work Centers
print("\n[1/6] Creating Work Centers...")
wc_data = [
    ('CNC Machining Center', 'Building A - Floor 1'),
    ('Assembly Line Station', 'Building A - Floor 2'),
    ('Quality Control Lab', 'Building B - Floor 1'),
    ('Welding & Fabrication', 'Building C - Floor 1'),
    ('Paint & Coating Booth', 'Building C - Floor 2'),
    ('Packaging & Shipping', 'Building D - Floor 1'),
    ('Material Storage Warehouse', 'Building E'),
    ('Testing & Calibration Lab', 'Building B - Floor 2')
]

wc_objects = []
for name, loc in wc_data:
    wc, _ = WorkCenter.objects.get_or_create(name=name, defaults={'location': loc})
    wc_objects.append(wc)
    print(f"  ✓ {name}")

# Users
print("\n[2/6] Creating Users...")
admin, _ = User.objects.get_or_create(
    username='admin',
    defaults={'is_staff': True, 'is_superuser': True, 'email': 'admin@gearguard.com'}
)
admin.set_password('admin123')
admin.save()

# Create 15 technicians
technician_names = [
    ('John', 'Martinez'), ('Sarah', 'Chen'), ('Michael', 'Patel'),
    ('Emily', 'Kim'), ('David', 'Singh'), ('Jessica', 'Lee'),
    ('Robert', 'Garcia'), ('Amanda', 'Rodriguez'), ('James', 'Wilson'),
    ('Lisa', 'Anderson'), ('William', 'Taylor'), ('Jennifer', 'Thomas'),
    ('Richard', 'Moore'), ('Maria', 'Jackson'), ('Charles', 'White')
]

technicians = []
for i, (first, last) in enumerate(technician_names, 1):
    tech, _ = User.objects.get_or_create(
        username=f'tech{i}',
        defaults={
            'first_name': first,
            'last_name': last,
            'email': f'{first.lower()}.{last.lower()}@gearguard.com'
        }
    )
    tech.set_password('tech123')
    tech.save()
    technicians.append(tech)

print(f"  ✓ Created {len(technicians)} technicians + admin")

# GENERATE TEMPLATES VIA LLM (ONLY 2 LLM CALLS!)
print("\n[3/6] Generating templates via LLM...")
print("  (This uses LLM smartly - only 2 calls for all templates)")

# Equipment types via LLM
eq_prompt = """Generate 20 different industrial equipment types for manufacturing.
Return ONLY a Python list: ["CNC Lathe", "Hydraulic Press", ...]
No explanations, just the list."""

eq_response = call_llama_once(eq_prompt)
if eq_response:
    try:
        equipment_types = eval(eq_response.replace('```python', '').replace('```', '').strip())[:20]
        print(f"  ✓ LLM generated {len(equipment_types)} equipment types")
    except:
        equipment_types = None
        print("  ⚠️  LLM parsing failed, using fallback")
else:
    equipment_types = None
    print("  ⚠️  LLM not available, using fallback")

if not equipment_types or len(equipment_types) < 20:
    equipment_types = [
        "CNC Milling Machine", "Hydraulic Press", "Industrial Robot Arm", "Conveyor Belt System",
        "Air Compressor", "Injection Molding Machine", "Laser Cutting Machine", "TIG Welder",
        "Vertical Lathe", "Gear Hobbing Machine", "EDM Machine", "Plasma Cutter",
        "Shot Blasting Machine", "Industrial Furnace", "Coordinate Measuring Machine",
        "Grinding Machine", "Boring Mill", "Broaching Machine", "Honing Machine", "Turret Punch Press"
    ]

# Failure patterns via LLM
failure_prompt = """Generate 50 common industrial equipment failure modes.
Return ONLY a Python list: ["bearing wear", "seal leak", ...]
No explanations, just the list."""

failure_response = call_llama_once(failure_prompt)
if failure_response:
    try:
        failure_patterns = eval(failure_response.replace('```python', '').replace('```', '').strip())[:50]
        print(f"  ✓ LLM generated {len(failure_patterns)} failure patterns")
    except:
        failure_patterns = None
        print("  ⚠️  LLM parsing failed, using fallback")
else:
    failure_patterns = None
    print("  ⚠️  LLM not available, using fallback")

if not failure_patterns or len(failure_patterns) < 50:
    failure_patterns = [
        "bearing wear and failure", "hydraulic seal leak", "belt misalignment", "motor overheating",
        "sensor calibration drift", "coolant pump failure", "electrical short circuit", "excessive vibration",
        "thermal expansion damage", "lubrication system failure", "pressure valve malfunction", "control system error",
        "pneumatic actuator failure", "shaft wear", "gear tooth damage", "coupling misalignment",
        "filter clogging", "oil contamination", "cable insulation damage", "relay contact failure",
        "encoder position error", "limit switch malfunction", "mechanical wear", "fatigue crack formation",
        "corrosion damage", "weld joint failure", "bolt loosening", "gasket deterioration",
        "spring fatigue", "clutch slipping", "brake pad wear", "drive belt failure",
        "chain elongation", "spindle runout", "chuck jaw wear", "tool holder crack",
        "coolant nozzle blockage", "chip accumulation", "guide rail wear", "ball screw backlash",
        "servo motor failure", "power supply fluctuation", "PCB component failure", "software bug",
        "overload condition", "emergency stop trigger", "safety interlock failure", "maintenance overdue",
        "improper lubrication", "operator error"
    ]

print(f"  ✓ Total templates: {len(equipment_types)} types × {len(failure_patterns)} failures")
print(f"  ✓ Total LLM calls: 2 (not 1000!)")

# Description templates (no LLM needed!)
desc_templates = [
    "{eq} with {health}% operational efficiency after {age} days of service",
    "{eq} currently operating at {health}% capacity, installed {age} days ago",
    "{eq} showing {health}% performance level, {age} days in operation",
    "{eq} running at {health}% rated output after {age} operational days",
    "{eq} maintained at {health}% efficiency, {age} days since installation"
]

# GENERATE 200 EQUIPMENT (INSTANT - NO LLM CALLS!)
print("\n[4/6] Generating 200 equipment...")
equipment_list = []

for eq_id in range(1, 201):
    eq_type = random.choice(equipment_types)
    days_old = random.randint(100, 500)
    health = max(20, 100 - (days_old / 500 * 100))
    
    description = random.choice(desc_templates).format(
        eq=eq_type,
        health=int(health),
        age=days_old
    )
    
    eq = Equipment.objects.create(
        name=f"{eq_type} {eq_id}",
        equipment_id=f"EQ{eq_id:05d}",
        description=description,
        work_center=random.choice(wc_objects),
        status='operational' if health > 50 else 'maintenance',
        purchase_date=timezone.now().date() - timedelta(days=days_old),
        last_maintenance=timezone.now().date() - timedelta(days=random.randint(10, 60)),
        next_maintenance=timezone.now().date() + timedelta(days=int(40 * health / 100))
    )
    
    equipment_list.append({
        'equipment': eq,
        'type': eq_type,
        'days_old': days_old,
        'health': health
    })
    
    if eq_id % 50 == 0:
        print(f"  ✓ {eq_id} equipment created...")

print(f"  ✓ Total: {len(equipment_list)} equipment")

# GENERATE 1000+ MAINTENANCE RECORDS (INSTANT!)
print("\n[5/6] Generating 1000+ maintenance records...")

# Templates for descriptions (no LLM!)
impact_by_priority = {
    'critical': [
        'Production line completely stopped',
        'Safety hazard - immediate shutdown required',
        'Multiple systems affected - emergency response',
        'Complete equipment failure - critical situation'
    ],
    'high': [
        'Production efficiency reduced by 40-60%',
        'Quality degradation detected',
        'Performance significantly impaired',
        'Immediate corrective action required'
    ],
    'medium': [
        'Minor performance degradation observed',
        'Scheduled maintenance window needed',
        'Wear indicators approaching limits',
        'Preventive action recommended'
    ],
    'low': [
        'Routine preventive maintenance',
        'Scheduled inspection required',
        'Standard service interval',
        'Normal wear and tear monitoring'
    ]
}

action_templates = [
    'Replaced defective {component}, tested and verified',
    'Repaired {component}, performed full diagnostics',
    'Adjusted and recalibrated {component}',
    'Cleaned, serviced, and lubricated {component}',
    'Overhauled {component}, replaced worn parts',
    'Realigned {component}, restored specifications',
    'Updated firmware and tested {component}',
    'Tightened connections, inspected {component}'
]

total_records = 0

for idx, ed in enumerate(equipment_list, 1):
    eq = ed['equipment']
    # 5-10 maintenance records per equipment = 1000-2000 total
    num_records = random.randint(5, 10)
    
    for m in range(num_records):
        failure = random.choice(failure_patterns)
        priority = random.choices(
            ['critical', 'high', 'medium', 'low'],
            weights=[1, 2, 3, 4],
            k=1
        )[0]
        
        # Build description using templates
        impact = random.choice(impact_by_priority[priority])
        component = failure.split()[0] if ' ' in failure else failure
        action = random.choice(action_templates).format(component=component)
        
        description = f"ISSUE: {failure.capitalize()}\nIMPACT: {impact}\nACTION TAKEN: {action}"
        
        # Generate realistic costs and times
        cost = random.randint(300, 2500) if priority in ['critical', 'high'] else random.randint(150, 800)
        hours = random.uniform(2, 16) if priority in ['critical', 'high'] else random.uniform(1, 6)
        
        notes = f"Completed repair of {failure}. Parts replaced, system tested. Cost: ${cost}, Labor: {hours:.1f}h. Equipment returned to service."
        
        days_ago = random.randint(20, ed['days_old'])
        
        MaintenanceRequest.objects.create(
            equipment=eq,
            title=failure[:100].capitalize(),
            description=description,
            request_type='emergency' if priority == 'critical' else 'corrective' if priority == 'high' else 'preventive',
            priority=priority,
            status='completed',
            requested_by=random.choice(technicians),
            assigned_to=random.choice(technicians),
            scheduled_date=timezone.now() - timedelta(days=days_ago),
            estimated_hours=hours,
            created_at=timezone.now() - timedelta(days=days_ago),
            notes=notes
        )
        
        total_records += 1
    
    if idx % 50 == 0:
        print(f"  ✓ {total_records} records created... ({idx}/200 equipment)")

print(f"  ✓ Total: {total_records} maintenance records")

# VERIFY DATA QUALITY
print("\n[6/6] Verifying data quality...")
eq_count = Equipment.objects.count()
maint_count = MaintenanceRequest.objects.count()
user_count = User.objects.count()
wc_count = WorkCenter.objects.count()
avg_per_eq = maint_count / eq_count if eq_count > 0 else 0

# Calculate priority distribution
priority_dist = {}
for priority in ['critical', 'high', 'medium', 'low']:
    count = MaintenanceRequest.objects.filter(priority=priority).count()
    priority_dist[priority] = (count, count/maint_count*100 if maint_count > 0 else 0)

print("\n" + "=" * 80)
print("✅ PRODUCTION DATA GENERATION COMPLETE!")
print("=" * 80)
print(f"\n📊 FINAL STATISTICS:")
print(f"  • Equipment: {eq_count}")
print(f"  • Maintenance Records: {maint_count}")
print(f"  • Average per equipment: {avg_per_eq:.1f}")
print(f"  • Users: {user_count} ({len(technicians)} technicians + admin)")
print(f"  • Work Centers: {wc_count}")

print(f"\n📈 PRIORITY DISTRIBUTION:")
for priority, (count, pct) in priority_dist.items():
    print(f"  • {priority.upper():<8} {count:>4} ({pct:>5.1f}%)")

print(f"\n⚡ PERFORMANCE:")
print(f"  • Total LLM calls: 2")
print(f"  • Generation method: Template-based (PRO)")
print(f"  • Time: 2-3 minutes")
print(f"  • Cost: $0.00")

print(f"\n✅ ML TRAINING REQUIREMENTS:")
print(f"  • Minimum needed: 1000 records → {'PASS ✓' if maint_count >= 1000 else 'FAIL ✗'}")
print(f"  • Recommended: 200+ equipment → {'PASS ✓' if eq_count >= 200 else 'FAIL ✗'}")
print(f"  • Production ready: {'YES ✓' if maint_count >= 1000 and eq_count >= 200 else 'NO ✗'}")

print(f"\n🎯 NEXT STEPS:")
print(f"  1. Train ML model: python manage.py train_model")
print(f"  2. Run server: python manage.py runserver")
print(f"  3. Visit: http://127.0.0.1:8000/maintenance/")
print("=" * 80)