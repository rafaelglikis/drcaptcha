# Generated by Django 2.1.3 on 2019-03-07 15:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('captcha', '0019_auto_20190307_1703'),
    ]

    operations = [
        migrations.AddField(
            model_name='digit',
            name='value',
            field=models.CharField(blank=True, max_length=5, null=True),
        ),
    ]
