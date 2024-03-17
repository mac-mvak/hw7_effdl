import requests
import io

from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import ToTensor
from PIL import Image



from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics




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
    image_url = data['url']
    image_data = requests.get(image_url).content

    image = Image.open(io.BytesIO(image_data))
    image = to_tensor(image)
    ans = model([image])
    ans = ans[0]

    scores = ans['scores']
    labels = ans['labels']

    named_labels = []

    for i in range(len(scores)):
        if scores[i] < 0.75:
            continue
        named_labels.append(names[labels[i]])

    named_labels.sort()

    return jsonify({
        "objects": named_labels
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
