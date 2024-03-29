# Generated by Django 2.1.3 on 2019-03-15 16:38

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('captcha', '0024_neuralnetwork_confusion_matrix'),
    ]

    operations = [
        migrations.CreateModel(
            name='Ensemble',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('accuracy', models.FloatField()),
            ],
        ),
        migrations.CreateModel(
            name='NeuralNetworkEnsembles',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('weight', models.FloatField()),
                ('ensemble', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='captcha.Ensemble')),
                ('neural_network', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='captcha.NeuralNetwork')),
            ],
        ),
        migrations.AddField(
            model_name='ensemble',
            name='neural_networks',
            field=models.ManyToManyField(through='captcha.NeuralNetworkEnsembles', to='captcha.NeuralNetwork'),
        ),
    ]
