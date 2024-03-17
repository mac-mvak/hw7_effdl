import requests
import torch
import io
import grpc
import inference_pb2_grpc
import inference_pb2


from flask import Flask, request, jsonify
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from prometheus_flask_exporter import PrometheusMetrics


from PIL import Image
from torchvision.transforms import ToTensor



app = Flask(__name__, static_url_path="")
metrics = PrometheusMetrics(app)


model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)
names = MaskRCNN_ResNet50_FPN_Weights.COCO_V1.meta['categories']
model.eval()
to_tensor = ToTensor()


@app.route("/predict", methods=['POST'])
@metrics.counter("app_http_inference_count_total", "Multiprocess metric")
def predict():
    data = request.get_json(force=True)
    url_str = data['url']
    with grpc.insecure_channel('127.0.0.1:9090') as channel:
        service = inference_pb2_grpc.InstanceDetectorStub(channel)
        r = service.Predict(inference_pb2.InstanceDetectorInput(
            url=url_str
        ))
    named_labels = list(r.objects)
    return jsonify({
        "objects": named_labels
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
