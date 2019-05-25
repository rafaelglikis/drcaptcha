import tensorflow as tf
from django.contrib import admin
from django.urls import reverse
from django.utils.html import format_html
from django.utils.safestring import mark_safe

from captcha.services.ml.emnist import load_test_data_from_db
from captcha.services.ml.models import EnsembleModel
from .models import Digit, NeuralNetworkGroup
from .models import Captcha
from .models import NeuralNetwork
from .models import Ensemble


def linkify(field_name):
    """
    Converts a foreign key value into clickable links.

    If field_name is 'parent', link text will be str(obj.parent)
    Link will be admin url for the admin url for obj.parent.id:change
    """
    def _linkify(obj):
        app_label = obj._meta.app_label
        linked_obj = getattr(obj, field_name)
        model_name = linked_obj._meta.model_name
        view_name = "admin:{}_{}_change".format(app_label, model_name)
        link_url = reverse(view_name, args=[linked_obj.id])
        return format_html('<a href="{}">{}</a>', link_url, linked_obj.ascii_value)

    _linkify.short_description = field_name # Sets column name
    return _linkify


@admin.register(Captcha)
class CaptchaAdmin(admin.ModelAdmin):
    list_display = ['value', 'image_show_small']

    exclude = [
        'digit1',
        'digit2',
        'digit3',
        'digit4',
        'digit5',
        'digit6'
    ]

    readonly_fields = [
        'image_show',
        linkify('digit1'),
        linkify('digit2'),
        linkify('digit3'),
        linkify('digit4'),
        linkify('digit5'),
        linkify('digit6')
    ]

    search_fields = ['value']

    @staticmethod
    def image_show(obj):
        return mark_safe('<img src="{url}" width="600px" height="100px"/>'
                         .format(url=obj.image.path))

    @staticmethod
    def image_show_small(obj):
        return mark_safe('<img src="{url}" width="150px" height="25px"/>'
                         .format(url=obj.image.path))


@admin.register(Digit)
class DigitAdmin(admin.ModelAdmin):
    list_display = [
        'ascii_value',
        'value',
        'dataset',
        'image_show_small'
    ]

    readonly_fields = ["image_show"]

    search_fields = [
        'ascii_value',
        'value',
        'dataset'
    ]

    @staticmethod
    def image_show(obj):
        return mark_safe('<img src="{url}" width="100px" height="100px"/>'
                         .format(url=obj.image.path))

    @staticmethod
    def image_show_small(obj):
        return mark_safe('<img src="{url}" width="25px" height="25px"/>'
                         .format(url=obj.image.path))


@admin.register(NeuralNetwork)
class NeuralNetworkAdmin(admin.ModelAdmin):
    list_display = [
        'pk',
        'accuracy',
        'create_date',
        'training_set_size',
        'test_set_size',
        'epochs',
        'batch_size'
    ]


    exclude = [
        'accuracy_graph',
        'loss_graph',
        'model_summary_image',
        'confusion_matrix'
    ]

    readonly_fields = [
        'path',
        'create_date',
        'epochs',
        'batch_size',
        'accuracy',
        'training_set_size',
        'test_set_size',
        'accuracy_plot_show',
        'loss_plot_show',
        'model_summary_graph_show',
        'confusion_matrix_show'
    ]

    @staticmethod
    def accuracy_plot_show(obj):
        return mark_safe('<img src="{url}"/>'.format(url=obj.accuracy_graph.path))

    @staticmethod
    def loss_plot_show(obj):
        return mark_safe('<img src="{url}"/>'.format(url=obj.loss_graph.path))

    @staticmethod
    def model_summary_graph_show(obj):
        return mark_safe('<img src="{url}"/>'.format(url=obj.model_summary_image.path))

    @staticmethod
    def confusion_matrix_show(obj):
        return mark_safe('<img src="{url}"/>'.format(url=obj.confusion_matrix.path))


class NeuralNetworkGroupInline(admin.TabularInline):
    readonly_fields = ["accuracy"]
    model = NeuralNetworkGroup
    extra = 1

    @staticmethod
    def accuracy(obj):
        return "{}".format(obj.neural_network.accuracy)


@admin.register(Ensemble)
class EnsembleAdmin(admin.ModelAdmin):
    list_display = ['pk', 'accuracy']

    readonly_fields = ["accuracy"]

    inlines = [NeuralNetworkGroupInline]

    def save_formset(self, request, form, formset, change):
        models = formset.save(commit=False)
        for model in models:
            model.save()
        formset.save_m2m()

        # Normalize weights if data altered
        if change and models:
            ensemble = Ensemble.objects.get(id=models[0].ensemble.id)
            ensemble.normalize_weights()

            # Re-evaluate model with the new weights
            data, labels = load_test_data_from_db()
            ensemble_model = EnsembleModel(ensemble)
            ensemble.accuracy = ensemble_model.evaluate(data, labels)
            ensemble.save()

    def save_model(self, request, obj, form, change):
        super(EnsembleAdmin, self).save_model(request, obj, form, change)

        if change:
            tf.keras.backend.clear_session()
            obj.save()


