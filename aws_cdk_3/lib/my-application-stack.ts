import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';

export class MyApplicationStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // Create an S3 bucket
    const bucket = new s3.Bucket(this, 'MyAppBucket', {
      bucketName: `my-app-bucket-${this.account}-${this.region}`,
      versioned: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY, // For dev/test only
      autoDeleteObjects: true, // For dev/test only
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
    });

    // Create a DynamoDB table
    const table = new dynamodb.Table(this, 'MyAppTable', {
      tableName: 'my-app-table',
      partitionKey: {
        name: 'id',
        type: dynamodb.AttributeType.STRING,
      },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.DESTROY, // For dev/test only
    });

    // Create a Lambda function
    const lambdaFunction = new lambda.Function(this, 'MyAppFunction', {
      functionName: 'my-app-function',
      runtime: lambda.Runtime.PYTHON_3_11,
      handler: 'index.handler',
      code: lambda.Code.fromInline(`
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
      `),
      timeout: cdk.Duration.seconds(30),
      memorySize: 256,
      environment: {
        BUCKET_NAME: bucket.bucketName,
        TABLE_NAME: table.tableName,
      },
    });

    // Grant permissions
    bucket.grantReadWrite(lambdaFunction);
    table.grantReadWriteData(lambdaFunction);

    // Create an API Gateway
    const api = new apigateway.RestApi(this, 'MyAppApi', {
      restApiName: 'my-app-api',
      description: 'API for my application',
      deployOptions: {
        stageName: 'dev',
      },
      defaultCorsPreflightOptions: {
        allowOrigins: apigateway.Cors.ALL_ORIGINS,
        allowMethods: apigateway.Cors.ALL_METHODS,
      },
    });

    // Add Lambda integration
    const lambdaIntegration = new apigateway.LambdaIntegration(lambdaFunction, {
      requestTemplates: {
        'application/json': '{ "statusCode": "200" }',
      },
    });

    // Add API resources and methods
    const apiResource = api.root.addResource('api');
    apiResource.addMethod('GET', lambdaIntegration);
    apiResource.addMethod('POST', lambdaIntegration);

    // Output important values
    new cdk.CfnOutput(this, 'BucketName', {
      value: bucket.bucketName,
      description: 'Name of the S3 bucket',
    });

    new cdk.CfnOutput(this, 'TableName', {
      value: table.tableName,
      description: 'Name of the DynamoDB table',
    });

    new cdk.CfnOutput(this, 'ApiUrl', {
      value: api.url,
      description: 'URL of the API Gateway',
    });

    new cdk.CfnOutput(this, 'FunctionName', {
      value: lambdaFunction.functionName,
      description: 'Name of the Lambda function',
    });
  }
}