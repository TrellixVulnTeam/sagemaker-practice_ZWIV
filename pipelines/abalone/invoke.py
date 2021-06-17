import os
import io
import boto3
import json
import csv

# grab environment variables
ENDPOINT_NAME = 'demo-project-staging' #os.environ['ENDPOINT_NAME']
boto_session = boto3.Session(region_name='us-east-1')
sagemaker_client = boto_session.client("sagemaker")
runtime_client = boto_session.client("sagemaker-runtime")

def lambda_handler(payload):
    print(payload)
    
    response = runtime_client.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=payload)
    print(response)
    result = json.loads(response['Body'].read().decode())
    print(result)

print('hi')

