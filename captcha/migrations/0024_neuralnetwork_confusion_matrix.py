# Generated by Django 2.1.3 on 2019-03-08 16:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('captcha', '0023_remove_neuralnetwork_confusion_matrix'),
    ]

    operations = [
        migrations.AddField(
            model_name='neuralnetwork',
            name='confusion_matrix',
            field=models.ImageField(blank=True, null=True, upload_to=''),
        ),
    ]
