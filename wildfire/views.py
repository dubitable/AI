from django.http import JsonResponse, HttpRequest, HttpResponseBadRequest, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from . import model
import json

@csrf_exempt
def predict(request: HttpRequest):
    if request.method != "POST": return HttpResponse("Wildfire Detector")

    body = json.loads(request.body)
    if "image" not in body: return HttpResponseBadRequest()

    image = model.load_image(body["image"])
    return JsonResponse(model.detect(image))