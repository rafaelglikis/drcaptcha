# Generated by Django 2.1.3 on 2018-11-20 14:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('captcha', '0010_auto_20181118_1454'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='captcha',
            name='used',
        ),
        migrations.AlterField(
            model_name='digit',
            name='value',
            field=models.SmallIntegerField(blank=True, null=True),
        ),
    ]
