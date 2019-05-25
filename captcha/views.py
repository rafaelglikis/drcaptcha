from django.http.response import JsonResponse
from django.shortcuts import render

from .models import Captcha
import captcha.services.ocr as ocr
import captcha.services.captcha as cs


def index(request):
    """ Home page """
    return render(request, 'index.html', {
        'index': 'active'
    })


def methodology(request):
    """ Methodology page """
    return render(request, 'methodology.html', {
        'methodology': 'active'
    })


def test_captcha(request):
    """ Captcha testing page """
    captcha = Captcha.get_random()
    return render(request, 'test_captcha.html', {
        'test_captcha': 'active',
        'captcha': captcha
    })


def test_captcha_value(request, id):
    """ Captcha testing page -- test a value """
    value = request.GET.get('value', None)
    captcha = Captcha.objects.filter(pk=id).first()

    return render(request, 'test_captcha.html', {
        'test_captcha': 'active',
        'captcha': captcha,
        'correct': cs.check_captcha_value(captcha, value),
        'value': value
    })


def canvas(request):
    """ OCR service page """
    return render(request, 'canvas.html', {
        'canvas': 'active'
    })


def ocr_prediction(request):
    """ OCR prediction in ajax form """
    image_data = request.GET.get('imgURI', '')
    image = ocr.preprocess(image_data)

    return JsonResponse({
        'result': ocr.predict(image)
    })


def api_docs(request):
    """ API docs page """
    return render(request, 'api_docs.html', {
        'api': 'active'
    })


def api_get_captcha(request):
    """ API get a catpcha """
    captcha = Captcha.get_random()
    return JsonResponse({
        "captcha": captcha.id,
        "image": captcha.image.url
    })


def api_test_captcha_value(request):
    """ API: test catpcha value """
    captcha_id = request.GET.get('captcha', None)
    value = request.GET.get('value', None)
    if captcha_id is None or value is None:
        return JsonResponse({
            "message": "You must specify captcha id and value"
        })

    captcha = Captcha.objects.filter(pk=captcha_id).first()

    return JsonResponse({
        "test": cs.check_captcha_value(captcha, value)
    })
