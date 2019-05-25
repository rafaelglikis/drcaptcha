from django.urls import path

from . import views

urlpatterns = [
    # Dr Captcha
    path('', views.index, name='index'),
    path('test-captcha', views.test_captcha, name='test_captcha'),
    path('test-captcha/<int:id>', views.test_captcha_value, name='test_captcha_value'),
    path('canvas', views.canvas, name='canvas'),
    path('api', views.api_docs, name='api_docs'),
    path('methodology', views.methodology, name='methodology'),

    # Dr Captcha API
    path('_do_ocr', views.ocr_prediction, name='ocr_prediction'),
    path('api/get-captcha', views.api_get_captcha, name='api_get_captcha'),
    path('api/test-captcha', views.api_test_captcha_value, name='api_test_captcha')
]