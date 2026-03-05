#!/usr/bin/env bash
set -o errexit

pip install -r requirements.txt
python manage.py collectstatic --no-input
python manage.py migrate

# Create superuser if needed
python manage.py shell -c "
from django.contrib.auth.models import User
if not User.objects.filter(username='admin').exists():
    User.objects.create_superuser('admin', 'admin@gearguard.com', 'changeme123')
    print('Admin user created')
"

# Generate data if database is empty
python manage.py shell -c "
from maintenance.models import Equipment
if Equipment.objects.count() == 0:
    print('Database empty - generating data...')
    import subprocess
    subprocess.run(['python', 'generate_production_data.py'])
"