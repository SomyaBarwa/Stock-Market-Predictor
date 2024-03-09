from rest_framework import serializers

from .models import Stock, StockPrice

class StockSerializer(serializers.ModelSerializer):
    class Meta:
        model = Stock
        fields = '__all__'

# class StockPriceSerializer(serializers.ModelSerializer):
#     stock_id = serializers.ReadOnlyField(source='stock_id.stock_id')

#     class Meta:
#         model = StockPrice
#         lookupfield = 'id'
#         fields = '__all__'




class StockPriceSerializer(serializers.HyperlinkedModelSerializer):
    stock_id = serializers.SlugRelatedField(queryset=Stock.objects.all(), slug_field='stock_id')
    class Meta:
        model = StockPrice
        fields = ['stock_id', 'date', 'close_price']