# Generated by Django 2.2.1 on 2019-05-31 10:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('demo', '0007_remove_images_swet'),
    ]

    operations = [
        migrations.AlterField(
            model_name='attrvalue',
            name='value',
            field=models.FloatField(),
        ),
    ]
