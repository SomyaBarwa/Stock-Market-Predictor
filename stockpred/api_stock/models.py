from django.db import models

# Create your models here.

class Stock(models.Model):
    stock_id = models.CharField(max_length=10, primary_key=True)
    stock_name = models.CharField(max_length=100, blank=True, null=True)
    # field that captures the last date of stock data that was fetched
    last_stock_data_date = models.DateField(blank=True, null=True)

    def __str__(self):
        return self.stock_id
    
    
class StockPrice(models.Model):
    stock_id = models.ForeignKey(Stock, on_delete=models.CASCADE)
    date = models.DateField()
    close_price = models.FloatField()
    
    # open_price = models.FloatField()
    # high_price = models.FloatField()
    # low_price = models.FloatField()
    # volume = models.IntegerField()
    # adj_close_price = models.FloatField()

    class Meta:
        unique_together = ('stock_id', 'date')

    def __str__(self):
        return self.stock_id.stock_id + ' ' + str(self.date) + ' ' + str(self.close_price)
