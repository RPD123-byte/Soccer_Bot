#!/usr/bin/env python3.11
import os
import aws_cdk as cdk
from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_apigateway as apigateway,
    aws_dynamodb as dynamodb,
    RemovalPolicy,
    Duration
)
from constructs import Construct


class MyApplicationStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create an S3 bucket
        bucket = s3.Bucket(
            self, "MyAppBucket",
            bucket_name=f"my-app-bucket-{self.account}-{self.region}",
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY,  # For dev/test only
            auto_delete_objects=True,  # For dev/test only
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL
        )

        # Create a DynamoDB table
        table = dynamodb.Table(
            self, "MyAppTable",
            table_name="my-app-table",
            partition_key=dynamodb.Attribute(
                name="id",
                type=dynamodb.AttributeType.STRING
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.DESTROY  # For dev/test only
        )

        # Create a Lambda function
        lambda_function = lambda_.Function(
            self, "MyAppFunction",
            function_name="my-app-function",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="index.handler",
            code=lambda_.Code.from_inline("""
import json
import boto3
import os

def handler(event, context):
    # Example Lambda function
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Hello from Lambda!',
            'bucket': os.environ.get('BUCKET_NAME'),
            'table': os.environ.get('TABLE_NAME')
        })
    }
            """),
            timeout=Duration.seconds(30),
            memory_size=256,
            environment={
                "BUCKET_NAME": bucket.bucket_name,
                "TABLE_NAME": table.table_name
            }
        )

        # Grant permissions
        bucket.grant_read_write(lambda_function)
        table.grant_read_write_data(lambda_function)

        # Create an API Gateway
        api = apigateway.RestApi(
            self, "MyAppApi",
            rest_api_name="my-app-api",
            description="API for my application",
            deploy_options={
                "stage_name": "dev"
            }
        )

        # Add Lambda integration
        lambda_integration = apigateway.LambdaIntegration(
            lambda_function,
            request_templates={
                "application/json": '{ "statusCode": "200" }'
            }
        )

        # Add API resources and methods
        api_resource = api.root.add_resource("api")
        api_resource.add_method("GET", lambda_integration)
        api_resource.add_method("POST", lambda_integration)

        # Output important values
        cdk.CfnOutput(
            self, "BucketName",
            value=bucket.bucket_name,
            description="Name of the S3 bucket"
        )

        cdk.CfnOutput(
            self, "TableName", 
            value=table.table_name,
            description="Name of the DynamoDB table"
        )

        cdk.CfnOutput(
            self, "ApiUrl",
            value=api.url,
            description="URL of the API Gateway"
        )

        cdk.CfnOutput(
            self, "FunctionName",
            value=lambda_function.function_name,
            description="Name of the Lambda function"
        )


app = cdk.App()

# Get environment from context or use defaults
env_name = app.node.try_get_context("env") or "dev"
account = app.node.try_get_context("account") or os.environ.get("CDK_DEFAULT_ACCOUNT")
region = app.node.try_get_context("region") or os.environ.get("CDK_DEFAULT_REGION")

MyApplicationStack(
    app, 
    f"MyApplicationStack-{env_name}",
    env=cdk.Environment(account=account, region=region),
    description=f"My Application Stack for {env_name} environment"
)

app.synth()