from django.core.management.base import BaseCommand

class Command(BaseCommand):
    def handle(self, *args, **kwargs):
        print("=" * 50)
        print("TEST COMMAND EXECUTED!")
        print("=" * 50)