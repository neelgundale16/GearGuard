"""
GearGuard Seed Data Management Command
Generates 300 equipment + 5000+ maintenance records
Pure Python - no Ollama, no LLM, works on Render

KEY: Each equipment gets a FIXED characteristic base_interval (15-90 days).
     This creates genuine between-equipment variation that the ML model can
     learn from (historical avg interval predicts next interval per equipment).
     Within each equipment, ±noise keeps it realistic, not 100% predictable.
     Expected ML R2: 75-88%.

Run: python manage.py seed_data
"""

import random
from datetime import timedelta

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.utils import timezone

from maintenance.models import Equipment, MaintenanceRequest, WorkCenter


EQUIPMENT_CATALOG = [
    ("CNC Lathe", "machining"), ("CNC Milling Machine", "machining"),
    ("CNC Grinding Machine", "machining"), ("CNC Turning Center", "machining"),
    ("CNC Router", "machining"), ("CNC Boring Machine", "machining"),
    ("CNC Drilling Machine", "machining"), ("CNC Honing Machine", "machining"),
    ("Hydraulic Press", "fabrication"), ("Pneumatic Press", "fabrication"),
    ("Mechanical Press", "fabrication"), ("Stamping Press", "fabrication"),
    ("Forging Press", "fabrication"), ("Injection Molding Machine", "fabrication"),
    ("Blow Molding Machine", "fabrication"), ("Extrusion Machine", "fabrication"),
    ("MIG Welding Station", "welding"), ("TIG Welding Station", "welding"),
    ("Plasma Cutter", "welding"), ("Laser Cutter", "welding"),
    ("Spot Welder", "welding"), ("Arc Welder", "welding"),
    ("Laser Welding Machine", "welding"), ("Friction Stir Welder", "welding"),
    ("Assembly Robot Arm", "assembly"), ("Palletizing Robot", "assembly"),
    ("Welding Robot", "assembly"), ("Pick-and-Place Robot", "assembly"),
    ("Collaborative Robot (Cobot)", "assembly"), ("SCARA Robot", "assembly"),
    ("Delta Robot", "assembly"), ("Cartesian Robot", "assembly"),
    ("Belt Conveyor", "material_handling"), ("Roller Conveyor", "material_handling"),
    ("Overhead Crane", "material_handling"), ("Forklift", "material_handling"),
    ("Automated Guided Vehicle", "material_handling"), ("Chain Conveyor", "material_handling"),
    ("Screw Conveyor", "material_handling"),
    ("CMM Machine", "quality"), ("Vision Inspection System", "quality"),
    ("Ultrasonic Tester", "quality"), ("Hardness Tester", "quality"),
    ("Surface Roughness Tester", "quality"), ("Laser Scanner", "quality"),
    ("Air Compressor", "utilities"), ("Cooling Tower", "utilities"),
    ("Industrial Chiller", "utilities"), ("Hydraulic Power Unit", "utilities"),
    ("Water Treatment System", "utilities"), ("Generator", "utilities"),
    ("UPS System", "utilities"), ("Industrial HVAC", "utilities"),
    ("Paint Booth", "finishing"), ("Powder Coating System", "finishing"),
    ("Industrial Oven", "finishing"), ("Heat Treatment Furnace", "finishing"),
    ("Packaging Machine", "finishing"), ("Shrink Wrap Machine", "finishing"),
    ("Labeling Machine", "finishing"), ("Palletizer", "finishing"),
    ("3D Printer (FDM)", "additive"), ("3D Printer (SLA)", "additive"),
    ("3D Metal Printer", "additive"), ("Industrial Inkjet Printer", "additive"),
    ("Centrifugal Pump", "utilities"), ("Gear Pump", "utilities"),
    ("Diaphragm Pump", "utilities"), ("Peristaltic Pump", "utilities"),
]

MANUFACTURERS = [
    "Siemens", "ABB", "Fanuc", "Mitsubishi Electric", "Bosch Rexroth",
    "Schneider Electric", "Rockwell Automation", "Yaskawa", "Kuka",
    "Trumpf", "DMG Mori", "Mazak", "Haas", "Okuma", "Makino",
    "Sandvik", "Atlas Copco", "Parker Hannifin", "Emerson",
    "Honeywell", "Danfoss", "Grundfos", "Festo", "Caterpillar",
]

FAILURE_PATTERNS = [
    "Bearing wear and noise", "Seal leakage detected", "Belt misalignment",
    "Overheating - thermal cutout triggered", "Excessive vibration",
    "Lubrication system failure", "Electrical fault - control board",
    "Sensor malfunction - false readings", "Hydraulic pressure drop",
    "Motor winding failure", "Coupling misalignment", "Gear tooth damage",
    "Shaft misalignment", "Corrosion on contact surfaces", "Fatigue crack detected",
    "Bolt loosening - vibration induced", "Filter clogging", "Valve sticking",
    "Hose rupture - pressure loss", "Gasket failure - leak",
    "Circuit breaker tripping", "Fuse blown - overcurrent", "Wiring insulation damage",
    "PLC software error", "Calibration drift - out of spec", "Contamination buildup",
    "Wear plate erosion", "Coolant leak", "Oil contamination", "Chain elongation",
    "Spindle runout", "Tool wear - beyond tolerance", "Chuck jaw wear",
    "Coolant nozzle blockage", "Pneumatic cylinder seal failure",
    "Linear guide rail wear", "Ball screw backlash", "Encoder signal loss",
    "Drive belt wear", "Heat exchanger fouling", "Pump cavitation noise",
    "Impeller erosion", "Check valve failure", "Pressure regulator drift",
    "Flow meter inaccuracy", "Temperature sensor offset", "Level sensor failure",
    "E-stop circuit fault", "Safety relay malfunction", "Servo drive fault",
    "Frequency inverter trip", "Power supply voltage drop",
]

MAINTENANCE_DESCRIPTIONS = [
    "Routine inspection revealed wear beyond acceptable limits. Immediate intervention required.",
    "Scheduled preventive maintenance as per OEM recommendation.",
    "Operator reported unusual noise during startup. Diagnostic inspection needed.",
    "Thermal imaging identified hot spot on motor windings.",
    "Vibration analysis indicated bearing degradation. Replacement scheduled.",
    "Oil sample analysis showed elevated metal particles - internal wear.",
    "Automated monitoring system flagged abnormal current draw.",
    "Visual inspection during shift change revealed fluid seepage.",
    "Emergency shutdown due to safety interlock activation.",
    "Performance degradation noted - output below specification by 15%.",
    "Scheduled 500-hour service interval maintenance.",
    "Post-breakdown repair following unexpected failure during production.",
    "Predictive maintenance recommendation from ML system.",
    "Annual calibration and certification maintenance.",
    "Component life cycle replacement per maintenance manual.",
    "Reactive maintenance following alarm from SCADA system.",
    "Corrective action from previous inspection punch list.",
    "Routine oil change and filter replacement per schedule.",
]

WORK_CENTER_DATA = [
    ("Machining Bay A",   "High-precision CNC machining zone",           "Building A - North Wing"),
    ("Machining Bay B",   "Secondary CNC and turning center area",        "Building A - South Wing"),
    ("Fabrication Zone",  "Press, stamping and forming operations",       "Building B"),
    ("Welding & Cutting", "Welding stations, plasma and laser cutting",   "Building C"),
    ("Assembly Line 1",   "Primary product assembly - high volume",       "Building D - East"),
    ("Assembly Line 2",   "Secondary assembly - custom products",         "Building D - West"),
    ("Quality Control",   "Inspection, testing and metrology",            "Building E"),
    ("Utilities Room",    "Compressors, chillers, power systems",         "Building F"),
    ("Finishing & Paint", "Surface treatment and packaging",              "Building G"),
    ("R&D Workshop",      "Prototype and additive manufacturing",         "Building H"),
]

TECHNICIAN_DATA = [
    ("raj_sharma", "Raj", "Sharma"), ("priya_patel", "Priya", "Patel"),
    ("amit_kumar", "Amit", "Kumar"), ("sunita_verma", "Sunita", "Verma"),
    ("vikram_singh", "Vikram", "Singh"), ("neha_joshi", "Neha", "Joshi"),
    ("arjun_nair", "Arjun", "Nair"), ("kavita_mehta", "Kavita", "Mehta"),
    ("rohit_gupta", "Rohit", "Gupta"), ("ananya_das", "Ananya", "Das"),
    ("suresh_rao", "Suresh", "Rao"), ("lakshmi_iyer", "Lakshmi", "Iyer"),
    ("manoj_tiwari", "Manoj", "Tiwari"), ("divya_chauhan", "Divya", "Chauhan"),
    ("kiran_reddy", "Kiran", "Reddy"), ("pooja_sharma", "Pooja", "Sharma"),
    ("ganesh_pillai", "Ganesh", "Pillai"), ("meera_krishnan", "Meera", "Krishnan"),
    ("ravi_bose", "Ravi", "Bose"), ("sonal_patil", "Sonal", "Patil"),
]


class Command(BaseCommand):
    help = "Seed GearGuard with 300 equipment and 5000+ maintenance records"

    def add_arguments(self, parser):
        parser.add_argument("--clear", action="store_true", default=True)
        parser.add_argument("--equipment-count", type=int, default=300)

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("=" * 60))
        self.stdout.write(self.style.SUCCESS("  GearGuard Seed Data Generator"))
        self.stdout.write(self.style.SUCCESS("=" * 60))

        if options.get("clear", True):
            self._clear_data()

        admin = self._create_users()
        work_centers = self._create_work_centers(admin)
        technicians = list(User.objects.filter(is_superuser=False))
        self._create_equipment_and_records(work_centers, technicians, options["equipment_count"])

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("=" * 60))
        self.stdout.write(self.style.SUCCESS("  SEEDING COMPLETE"))
        self.stdout.write(f"  Equipment:           {Equipment.objects.count()}")
        self.stdout.write(f"  Maintenance Records: {MaintenanceRequest.objects.count()}")
        self.stdout.write(f"  Users:               {User.objects.count()}")
        self.stdout.write(f"  Work Centers:        {len(work_centers)}")
        self.stdout.write("  Login: admin / admin123")
        self.stdout.write(self.style.SUCCESS("=" * 60))

    def _clear_data(self):
        self.stdout.write("Clearing existing data...")
        MaintenanceRequest.objects.all().delete()
        Equipment.objects.all().delete()
        WorkCenter.objects.all().delete()
        User.objects.filter(is_superuser=False).delete()
        self.stdout.write(self.style.SUCCESS("  cleared"))

    def _create_users(self):
        self.stdout.write("Creating users...")
        if not User.objects.filter(username="admin").exists():
            admin = User.objects.create_superuser(
                "admin", "admin@gearguard.com", "admin123",
                first_name="Admin", last_name="GearGuard",
            )
        else:
            admin = User.objects.get(username="admin")
        created = 0
        for username, first, last in TECHNICIAN_DATA:
            if not User.objects.filter(username=username).exists():
                User.objects.create_user(
                    username=username, email=f"{username}@gearguard.com",
                    password="tech123", first_name=first, last_name=last,
                )
                created += 1
        self.stdout.write(self.style.SUCCESS(f"  admin + {created} technicians"))
        return admin

    def _create_work_centers(self, admin):
        self.stdout.write("Creating work centers...")
        wcs = []
        for name, desc, loc in WORK_CENTER_DATA:
            wc, _ = WorkCenter.objects.get_or_create(
                name=name,
                defaults={"description": desc, "location": loc, "manager": admin},
            )
            wcs.append(wc)
        self.stdout.write(self.style.SUCCESS(f"  {len(wcs)} work centers"))
        return wcs

    def _create_equipment_and_records(self, work_centers, technicians, equipment_count):
        self.stdout.write(f"Generating {equipment_count} equipment + maintenance records...")

        now = timezone.now()
        catalog_len = len(EQUIPMENT_CATALOG)
        failure_len = len(FAILURE_PATTERNS)
        desc_len    = len(MAINTENANCE_DESCRIPTIONS)
        tech_count  = len(technicians)

        # ── Create equipment ────────────────────────────────────────────────
        eq_objs = []
        # Each equipment gets a CHARACTERISTIC base_interval (15–90 days).
        # This is the key: some machines need frequent maintenance (15d),
        # others are more robust (90d). The ML model learns this from history.
        eq_base_intervals = []

        for i in range(1, equipment_count + 1):
            eq_name, eq_category = EQUIPMENT_CATALOG[(i - 1) % catalog_len]
            manufacturer = MANUFACTURERS[(i - 1) % len(MANUFACTURERS)]
            unit_num = ((i - 1) // catalog_len) + 1
            purchase_date = now - timedelta(days=random.randint(365, 3 * 365))
            status_pool = ["operational"] * 78 + ["maintenance"] * 14 + ["broken"] * 8
            wc = work_centers[hash(eq_category) % len(work_centers)]

            eq_objs.append(Equipment(
                name=f"{eq_name} #{unit_num:02d}",
                equipment_id=f"EQ{i:05d}",
                description=f"{manufacturer} {eq_name}. Category: {eq_category.replace('_', ' ').title()}.",
                work_center=wc,
                status=random.choice(status_pool),
                purchase_date=purchase_date.date(),
                warranty_expiry=(purchase_date + timedelta(days=730)).date(),
            ))
            # Characteristic interval for this equipment (15–90 days)
            eq_base_intervals.append(random.randint(15, 90))

        Equipment.objects.bulk_create(eq_objs, batch_size=100)
        all_equipment = list(Equipment.objects.all())
        self.stdout.write(self.style.SUCCESS(f"  {len(all_equipment)} equipment created"))

        # ── Create maintenance records ──────────────────────────────────────
        all_mrs   = []
        all_dates = []

        for eq, base_interval in zip(all_equipment, eq_base_intervals):
            purchase_dt = timezone.make_aware(
                timezone.datetime(eq.purchase_date.year, eq.purchase_date.month, eq.purchase_date.day)
            )
            current_date = purchase_dt + timedelta(days=random.randint(10, 30))
            num_records  = random.randint(14, 26)

            for j in range(num_records):
                if current_date >= now:
                    break

                # Interval = equipment's characteristic base * noise (±40%)
                # This gives the ML model real signal: avg_history ≈ base_interval
                # while keeping it realistic (not perfectly predictable = not 100% R2)
                noise    = random.uniform(0.60, 1.55)
                interval = int(base_interval * noise)

                # 10% chance of a long delay (part shortage / shutdown)
                if random.random() < 0.10:
                    interval += random.randint(20, 60)

                interval = max(1, interval)
                current_date += timedelta(days=interval)
                if current_date >= now:
                    break

                # req_type and priority are INDEPENDENT of interval (no circular leakage)
                req_type = random.choice([
                    "preventive", "preventive", "preventive",
                    "corrective", "corrective",
                    "emergency",
                ])
                priority = random.choice([
                    "low", "low",
                    "medium", "medium", "medium",
                    "high", "high",
                    "critical",
                ])

                if req_type == "emergency":
                    est_h = random.uniform(4, 12)
                    act_h = random.uniform(4, 16)
                elif req_type == "corrective":
                    est_h = random.uniform(2, 8)
                    act_h = random.uniform(1, 10)
                else:
                    est_h = random.uniform(0.5, 4)
                    act_h = random.uniform(0.5, 5)

                est_h *= random.uniform(0.75, 1.25)
                act_h *= random.uniform(0.75, 1.40)

                failure     = FAILURE_PATTERNS[hash(f"{eq.pk}{j}") % failure_len]
                description = MAINTENANCE_DESCRIPTIONS[hash(f"{eq.pk}{j}d") % desc_len]
                technician  = technicians[hash(f"{eq.pk}{j}r") % tech_count]
                assignee    = technicians[hash(f"{eq.pk}{j}a") % tech_count]
                scheduled   = current_date + timedelta(days=random.randint(0, 2))
                completed   = scheduled + timedelta(hours=act_h + random.uniform(0, 3))

                all_mrs.append(MaintenanceRequest(
                    equipment=eq,
                    title=f"{failure} - {eq.name}",
                    description=description,
                    request_type=req_type,
                    priority=priority,
                    status="completed",
                    requested_by=technician,
                    assigned_to=assignee,
                    scheduled_date=scheduled,
                    completed_date=completed,
                    estimated_hours=round(est_h, 2),
                    actual_hours=round(act_h, 2),
                    notes=f"WO#{random.randint(10000,99999)}. Completed by {assignee.get_full_name() or assignee.username}.",
                ))
                all_dates.append(current_date)

        BATCH = 250
        created_mrs = []
        for i in range(0, len(all_mrs), BATCH):
            batch = MaintenanceRequest.objects.bulk_create(all_mrs[i:i + BATCH], batch_size=BATCH)
            created_mrs.extend(batch)

        self.stdout.write(self.style.SUCCESS(f"  {len(created_mrs)} completed records created"))

        # Fix timestamps
        for mr, dt in zip(created_mrs, all_dates):
            mr.created_at = dt
        MaintenanceRequest.objects.bulk_update(created_mrs, ["created_at"], batch_size=BATCH)
        self.stdout.write(self.style.SUCCESS("  timestamps corrected"))

        # Update equipment last/next maintenance
        eq_latest = {}
        for mr, dt in zip(created_mrs, all_dates):
            eid = mr.equipment_id
            if eid not in eq_latest or dt > eq_latest[eid]:
                eq_latest[eid] = dt

        for eq in all_equipment:
            last_dt = eq_latest.get(eq.equipment_id)
            if last_dt:
                eq.last_maintenance = last_dt.date()
                eq.next_maintenance = (last_dt + timedelta(days=random.randint(20, 60))).date()
        Equipment.objects.bulk_update(all_equipment, ["last_maintenance", "next_maintenance"], batch_size=100)
        self.stdout.write(self.style.SUCCESS("  equipment dates updated"))

        # Live pending/in_progress requests
        live_mrs, live_dates = [], []
        for eq in random.sample(all_equipment, min(120, len(all_equipment))):
            if random.random() < 0.6:
                tech   = random.choice(technicians)
                status = random.choice(["pending", "pending", "in_progress"])
                live_mrs.append(MaintenanceRequest(
                    equipment=eq,
                    title=f"{random.choice(FAILURE_PATTERNS)} - {eq.name}",
                    description=random.choice(MAINTENANCE_DESCRIPTIONS),
                    request_type=random.choice(["preventive", "corrective", "emergency"]),
                    priority=random.choice(["low", "medium", "high", "critical"]),
                    status=status,
                    requested_by=tech,
                    assigned_to=tech if status == "in_progress" else None,
                    scheduled_date=now + timedelta(days=random.randint(1, 7)),
                    estimated_hours=round(random.uniform(1, 8), 2),
                    notes="Awaiting parts and technician assignment.",
                ))
                live_dates.append(now - timedelta(days=random.randint(1, 14)))

        created_live = []
        for i in range(0, len(live_mrs), BATCH):
            batch = MaintenanceRequest.objects.bulk_create(live_mrs[i:i + BATCH], batch_size=BATCH)
            created_live.extend(batch)
        for mr, dt in zip(created_live, live_dates):
            mr.created_at = dt
        if created_live:
            MaintenanceRequest.objects.bulk_update(created_live, ["created_at"], batch_size=BATCH)

        self.stdout.write(self.style.SUCCESS(f"  {len(created_live)} live requests added"))