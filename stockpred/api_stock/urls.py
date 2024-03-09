from django.urls import path, include
from rest_framework import routers
from . import views

router = routers.DefaultRouter()
router.register(r'api/stock', views.StockViewSet, 'stock')
router.register(r'api/stockprice', views.StockPriceViewSet, 'stockprice')

urlpatterns = [
    path('api/stockprice/<str:stock_id>/', views.StockPriceView.as_view(), name='stockprice'),
] + router.urls

