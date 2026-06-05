package com.myorg;

import software.amazon.awscdk.CfnOutput;
import software.amazon.awscdk.CfnOutputProps;
import software.amazon.awscdk.Duration;
import software.amazon.awscdk.RemovalPolicy;
import software.amazon.awscdk.Stack;
import software.amazon.awscdk.StackProps;
import software.amazon.awscdk.services.apigateway.*;
import software.amazon.awscdk.services.dynamodb.*;
import software.amazon.awscdk.services.lambda.*;
import software.amazon.awscdk.services.s3.*;
import software.constructs.Construct;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class MyApplicationStack extends Stack {
    public MyApplicationStack(final Construct scope, final String id) {
        this(scope, id, null);
    }

    public MyApplicationStack(final Construct scope, final String id, final StackProps props) {
        super(scope, id, props);

        // Create an S3 bucket
        Bucket bucket = Bucket.Builder.create(this, "MyAppBucket")
                .bucketName("my-app-bucket-" + this.getAccount() + "-" + this.getRegion())
                .versioned(true)
                .removalPolicy(RemovalPolicy.DESTROY)
                .autoDeleteObjects(true)
                .encryption(BucketEncryption.S3_MANAGED)
                .blockPublicAccess(BlockPublicAccess.BLOCK_ALL)
                .enforceSSL(true)
                .build();

        // Create a DynamoDB table
        Table table = Table.Builder.create(this, "MyAppTable")
                .tableName("my-app-table")
                .partitionKey(Attribute.builder()
                        .name("id")
                        .type(AttributeType.STRING)
                        .build())
                .billingMode(BillingMode.PAY_PER_REQUEST)
                .removalPolicy(RemovalPolicy.DESTROY)
                .build();

        // Lambda function code
        String lambdaCode = """
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
""";

        // Create a Lambda function
        Map<String, String> environment = new HashMap<>();
        environment.put("BUCKET_NAME", bucket.getBucketName());
        environment.put("TABLE_NAME", table.getTableName());

        Function lambdaFunction = Function.Builder.create(this, "MyAppFunction")
                .functionName("my-app-function")
                .runtime(Runtime.PYTHON_3_11)
                .handler("index.handler")
                .code(Code.fromInline(lambdaCode))
                .timeout(Duration.seconds(30))
                .memorySize(256)
                .environment(environment)
                .build();

        // Grant permissions
        bucket.grantReadWrite(lambdaFunction);
        table.grantReadWriteData(lambdaFunction);

        // Create an API Gateway
        RestApi api = RestApi.Builder.create(this, "MyAppApi")
                .restApiName("my-app-api")
                .description("API for my application")
                .deployOptions(StageOptions.builder()
                        .stageName("dev")
                        .build())
                .defaultCorsPreflightOptions(CorsOptions.builder()
                        .allowOrigins(Cors.ALL_ORIGINS)
                        .allowMethods(Cors.ALL_METHODS)
                        .build())
                .build();

        // Add Lambda integration
        Map<String, String> requestTemplates = new HashMap<>();
        requestTemplates.put("application/json", "{ \"statusCode\": \"200\" }");
        
        LambdaIntegration lambdaIntegration = LambdaIntegration.Builder.create(lambdaFunction)
                .requestTemplates(requestTemplates)
                .build();

        // Add API resources and methods
        Resource apiResource = api.getRoot().addResource("api");
        apiResource.addMethod("GET", lambdaIntegration);
        apiResource.addMethod("POST", lambdaIntegration);

        // Output important values
        new CfnOutput(this, "BucketName", CfnOutputProps.builder()
                .value(bucket.getBucketName())
                .description("Name of the S3 bucket")
                .build());

        new CfnOutput(this, "TableName", CfnOutputProps.builder()
                .value(table.getTableName())
                .description("Name of the DynamoDB table")
                .build());

        new CfnOutput(this, "ApiUrl", CfnOutputProps.builder()
                .value(api.getUrl())
                .description("URL of the API Gateway")
                .build());

        new CfnOutput(this, "FunctionName", CfnOutputProps.builder()
                .value(lambdaFunction.getFunctionName())
                .description("Name of the Lambda function")
                .build());
    }
}