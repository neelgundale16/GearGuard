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
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
LLAMA_MODEL = os.getenv('LLAMA_MODEL', 'llama3.2')

def call_llama(prompt):
    """Call local Llama via Ollama"""
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": LLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()['response']
    except:
        return None

print("=" * 80)
print("🦙 LLAMA-POWERED DATA GENERATION (100% FREE!)")
print("=" * 80)

random.seed(42)

# CLEAR EXISTING DATA
print("\n[0/5] Clearing existing data...")
try:
    MaintenanceRequest.objects.all().delete()
    Equipment.objects.all().delete()
    WorkCenter.objects.all().delete()
    User.objects.filter(username__startswith='tech').delete()
    print("  ✓ Database cleared!")
except Exception as e:
    print(f"  ⚠️  Clear error (likely empty): {e}")

# Work Centers
print("\n[1/5] Creating Work Centers...")
work_centers = [
    ('Production Line A', 'Building 1'),
    ('Production Line B', 'Building 2'),
    ('Assembly Station', 'Building 3'),
]

wc_objects = []
for name, loc in work_centers:
    wc, _ = WorkCenter.objects.get_or_create(name=name, defaults={'location': loc})
    wc_objects.append(wc)
    print(f"  ✓ {name}")

# Users
print("\n[2/5] Creating Users...")
admin, created = User.objects.get_or_create(
    username='admin',
    defaults={'is_staff': True, 'is_superuser': True, 'email': 'admin@gearguard.com'}
)
if created:
    admin.set_password('admin123')
    admin.save()

technicians = []
for i in range(1, 4):
    tech, created = User.objects.get_or_create(
        username=f'tech{i}',
        defaults={'first_name': f'Tech', 'last_name': f'User{i}', 'email': f'tech{i}@gearguard.com'}
    )
    if created:
        tech.set_password('tech123')
        tech.save()
    technicians.append(tech)
print(f"  ✓ {len(technicians)} technicians + admin")

# Llama: Generate Equipment Types
print("\n[3/5] Using Llama to generate equipment types...")

prompt = """List 5 industrial equipment types for manufacturing. For each, give 3 common failures.
Format as Python list like:
[{"type": "CNC Machine", "failures": ["spindle failure", "coolant leak", "tool jam"]}, ...]

Return ONLY the Python list, no explanation."""

response = call_llama(prompt)

if not response:
    print("  ⚠️  Llama not responding, using fallback data")
    equipment_types = [
        {"type": "CNC Machine", "failures": ["spindle failure", "coolant leak", "tool jam"]},
        {"type": "Conveyor Belt", "failures": ["belt wear", "motor failure", "alignment issue"]},
        {"type": "Robotic Arm", "failures": ["joint stiffness", "gripper jam", "sensor error"]},
        {"type": "Hydraulic Press", "failures": ["seal leak", "pressure drop", "valve stuck"]},
        {"type": "Air Compressor", "failures": ["overheating", "filter clogged", "pressure loss"]}
    ]
else:
    try:
        response = response.replace('```python', '').replace('```', '').strip()
        equipment_types = eval(response)
        print(f"  ✓ Llama generated {len(equipment_types)} equipment types!")
    except:
        print("  ⚠️  Parsing error, using fallback")
        equipment_types = [
            {"type": "CNC Machine", "failures": ["spindle failure", "coolant leak", "tool jam"]},
            {"type": "Conveyor Belt", "failures": ["belt wear", "motor failure", "alignment issue"]},
        ]

# Generate Equipment
print("\n[4/5] Generating equipment with Llama descriptions...")

equipment_list = []
eq_id = 1

for et in equipment_types:
    for i in range(6):  # 6 each = 30 total
        days_old = random.randint(50, 200)
        health = max(30, 100 - (days_old / 200 * 100))
        
        # Llama: Generate description
        desc_prompt = f"Write 1 technical sentence describing a {et['type']} with {health:.0f}% health."
        description = call_llama(desc_prompt)
        if not description:
            description = f"{et['type']} operating at {health:.0f}% capacity"
        
        eq = Equipment.objects.create(
            name=f"{et['type']} {eq_id}",
            equipment_id=f"EQ{eq_id:04d}",
            description=description[:200],
            work_center=random.choice(wc_objects),
            status='operational' if health > 50 else 'maintenance',
            purchase_date=timezone.now().date() - timedelta(days=days_old),
            last_maintenance=timezone.now().date() - timedelta(days=random.randint(5, 30)),
            next_maintenance=timezone.now().date() + timedelta(days=int(30 * health / 100))
        )
        
        equipment_list.append({
            'equipment': eq,
            'type': et['type'],
            'failures': et['failures'],
            'days_old': days_old,
            'health': health
        })
        
        eq_id += 1
        if eq_id % 10 == 0:
            print(f"  ✓ Generated {eq_id-1} equipment...")

print(f"  ✓ Total: {len(equipment_list)} equipment")

# Generate Maintenance History
print("\n[5/5] Generating maintenance records with Llama...")

total = 0

for ed in equipment_list:
    eq = ed['equipment']
    num_maint = random.randint(3, 6)
    
    for m in range(num_maint):
        failure = random.choice(ed['failures'])
        priority = random.choice(['critical', 'high', 'medium', 'low'])
        
        # Llama: Generate maintenance report
        report_prompt = f"""Equipment {ed['type']} had {failure}. Priority: {priority}.
Write 3 lines:
ISSUE: [describe issue]
IMPACT: [operational impact]
ACTION: [what was done]"""
        
        description = call_llama(report_prompt)
        if not description:
            description = f"ISSUE: {failure}\nIMPACT: Equipment degraded\nACTION: Repaired"
        
        # Llama: Generate resolution
        notes_prompt = f"One sentence: How to fix {failure}? Include cost and hours."
        notes = call_llama(notes_prompt)
        if not notes:
            notes = f"Fixed {failure}. Cost: ${random.randint(500, 1500)}, Time: {random.randint(2, 6)}h"
        
        days_ago = random.randint(10, ed['days_old'])
        
        MaintenanceRequest.objects.create(
            equipment=eq,
            title=failure[:100],
            description=description[:500],
            request_type='emergency' if priority == 'critical' else 'preventive',
            priority=priority,
            status='completed',
            requested_by=random.choice(technicians),
            assigned_to=random.choice(technicians),
            scheduled_date=(timezone.now() - timedelta(days=days_ago)),
            estimated_hours=random.uniform(2, 8),
            created_at=timezone.now() - timedelta(days=days_ago),
            notes=notes[:200]
        )
        
        total += 1
        if total % 20 == 0:
            print(f"  ✓ {total} records...")

print(f"  ✓ Total: {total} maintenance records")

print("\n" + "=" * 80)
print("LLAMA DATA GENERATION COMPLETE!")
print("=" * 80)
print(f"\nSTATISTICS:")
print(f"  • Equipment: {len(equipment_list)}")
print(f"  • Maintenance Records: {total}")
print(f"  • Users: {len(technicians) + 1}")
print("\nReady to train model!")
print("=" * 80)