import grpc
import inference_pb2_grpc
import inference_pb2


from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics





app = Flask(__name__, static_url_path="")
metrics = PrometheusMetrics(app)



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
