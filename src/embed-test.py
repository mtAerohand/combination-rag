import boto3
import json

client = boto3.client(service_name='bedrock-runtime')

input_text = "NVIDIAの新しく出たGPUには何がありますか？名前を教えてください。"
model_id="amazon.titan-embed-text-v2:0"

request_body = {
    "inputText": input_text,
    "dimensions": 256,
    "normalize": True
}

try:
    response = client.invoke_model(
        body=json.dumps(request_body),
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
except Exception as e:
    print(f"error: {e}")

response_body = json.loads(response.get('body').read())
print(response_body)