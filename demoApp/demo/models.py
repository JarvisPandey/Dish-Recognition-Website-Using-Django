from django.db import models

class Sweets(models.Model):
	name = models.CharField(max_length=10)
	def __str__(self):
		return self.name

class AttrValue(models.Model):
	sweet_name = models.ForeignKey(Sweets, on_delete=models.CASCADE)
	attr_name = models.CharField(max_length=10)
	value = models.FloatField()

	def __str__(self):
		return self.sweet_name.name+ ' '+self.attr_name

class Images(models.Model):
    #swet = models.ForeignKey(Sweets, on_delete=models.CASCADE)
    file = models.ImageField()