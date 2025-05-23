using Amazon.CDK;
using Amazon.CDK.AWS.APIGateway;
using Amazon.CDK.AWS.DynamoDB;
using Amazon.CDK.AWS.Lambda;
using Amazon.CDK.AWS.S3;
using Constructs;
using System;
using System.Collections.Generic;
using DynamoAttribute = Amazon.CDK.AWS.DynamoDB.Attribute;

namespace MyCdkApp
{
    public class MyApplicationStack : Stack
    {
        internal MyApplicationStack(Construct scope, string id, IStackProps? props = null) : base(scope, id, props)
        {
            // Create an S3 bucket
            var bucket = new Bucket(this, "MyAppBucket", new BucketProps
            {
                BucketName = $"my-app-bucket-{Account}-{Region}",
                Versioned = true,
                RemovalPolicy = RemovalPolicy.DESTROY,
                AutoDeleteObjects = true,
                Encryption = BucketEncryption.S3_MANAGED,
                BlockPublicAccess = BlockPublicAccess.BLOCK_ALL,
                EnforceSSL = true
            });

            // Create a DynamoDB table
            var table = new Table(this, "MyAppTable", new TableProps
            {
                TableName = "my-app-table",
                PartitionKey = new DynamoAttribute
                {
                    Name = "id",
                    Type = AttributeType.STRING
                },
                BillingMode = BillingMode.PAY_PER_REQUEST,
                RemovalPolicy = RemovalPolicy.DESTROY
            });

            // Lambda function code
            var lambdaCode = @"
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
";

            // Create a Lambda function
            var lambdaFunction = new Function(this, "MyAppFunction", new FunctionProps
            {
                FunctionName = "my-app-function",
                Runtime = Runtime.PYTHON_3_11,
                Handler = "index.handler",
                Code = Code.FromInline(lambdaCode),
                Timeout = Duration.Seconds(30),
                MemorySize = 256,
                Environment = new Dictionary<string, string>
                {
                    { "BUCKET_NAME", bucket.BucketName },
                    { "TABLE_NAME", table.TableName }
                }
            });

            // Grant permissions
            bucket.GrantReadWrite(lambdaFunction);
            table.GrantReadWriteData(lambdaFunction);

            // Create an API Gateway
            var api = new RestApi(this, "MyAppApi", new RestApiProps
            {
                RestApiName = "my-app-api",
                Description = "API for my application",
                DeployOptions = new StageOptions
                {
                    StageName = "dev"
                },
                DefaultCorsPreflightOptions = new CorsOptions
                {
                    AllowOrigins = Cors.ALL_ORIGINS,
                    AllowMethods = Cors.ALL_METHODS
                }
            });

            // Add Lambda integration
            var lambdaIntegration = new LambdaIntegration(lambdaFunction, new LambdaIntegrationOptions
            {
                RequestTemplates = new Dictionary<string, string>
                {
                    { "application/json", "{ \"statusCode\": \"200\" }" }
                }
            });

            // Add API resources and methods
            var apiResource = api.Root.AddResource("api");
            apiResource.AddMethod("GET", lambdaIntegration);
            apiResource.AddMethod("POST", lambdaIntegration);

            // Output important values
            new CfnOutput(this, "BucketName", new CfnOutputProps
            {
                Value = bucket.BucketName,
                Description = "Name of the S3 bucket"
            });

            new CfnOutput(this, "TableName", new CfnOutputProps
            {
                Value = table.TableName,
                Description = "Name of the DynamoDB table"
            });

            new CfnOutput(this, "ApiUrl", new CfnOutputProps
            {
                Value = api.Url,
                Description = "URL of the API Gateway"
            });

            new CfnOutput(this, "FunctionName", new CfnOutputProps
            {
                Value = lambdaFunction.FunctionName,
                Description = "Name of the Lambda function"
            });
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var app = new App();
            
            var envName = app.Node.TryGetContext("env")?.ToString() ?? "dev";
            
            // Create environment - CDK will use defaults if not specified
            Amazon.CDK.Environment? environment = null;
            var account = app.Node.TryGetContext("account")?.ToString() ?? System.Environment.GetEnvironmentVariable("CDK_DEFAULT_ACCOUNT");
            var region = app.Node.TryGetContext("region")?.ToString() ?? System.Environment.GetEnvironmentVariable("CDK_DEFAULT_REGION");
            
            if (!string.IsNullOrEmpty(account) || !string.IsNullOrEmpty(region))
            {
                environment = new Amazon.CDK.Environment
                {
                    Account = account,
                    Region = region
                };
            }

            new MyApplicationStack(app, $"MyApplicationStack-{envName}", new StackProps
            {
                Env = environment,
                Description = $"My Application Stack for {envName} environment"
            });

            app.Synth();
        }
    }
}