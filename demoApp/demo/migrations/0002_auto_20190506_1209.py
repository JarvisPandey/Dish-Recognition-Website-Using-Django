# Generated by Django 2.2.1 on 2019-05-06 06:39

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('demo', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='sweets',
            name='att_name',
        ),
        migrations.RemoveField(
            model_name='sweets',
            name='value',
        ),
        migrations.CreateModel(
            name='AttrValue',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('attr_name', models.CharField(max_length=10)),
                ('value', models.IntegerField()),
                ('sweet_name', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='demo.Sweets')),
            ],
        ),
    ]