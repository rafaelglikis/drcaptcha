# Generated by Django 2.1.3 on 2018-11-20 23:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('captcha', '0013_auto_20181120_1712'),
    ]

    operations = [
        migrations.AddField(
            model_name='neuralnetwork',
            name='accuracy_graph',
            field=models.ImageField(blank=True, null=True, upload_to=''),
        ),
        migrations.AddField(
            model_name='neuralnetwork',
            name='loss_graph',
            field=models.ImageField(blank=True, null=True, upload_to=''),
        ),
        migrations.AddField(
            model_name='neuralnetwork',
            name='model_summary',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='neuralnetwork',
            name='model_summary_image',
            field=models.ImageField(blank=True, null=True, upload_to=''),
        ),
        migrations.AlterField(
            model_name='neuralnetwork',
            name='create_date',
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]
