# Generated by Django 2.1.3 on 2018-11-20 15:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('captcha', '0011_auto_20181120_1659'),
    ]

    operations = [
        migrations.AlterField(
            model_name='captcha',
            name='path',
            field=models.ImageField(blank=True, null=True, upload_to=''),
        ),
    ]
