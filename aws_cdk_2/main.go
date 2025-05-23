package main

import (
	"fmt"
	"os"

	"github.com/aws/aws-cdk-go/awscdk/v2"
	"github.com/aws/aws-cdk-go/awscdk/v2/awsapigateway"
	"github.com/aws/aws-cdk-go/awscdk/v2/awsdynamodb"
	"github.com/aws/aws-cdk-go/awscdk/v2/awslambda"
	"github.com/aws/aws-cdk-go/awscdk/v2/awss3"
	"github.com/aws/constructs-go/constructs/v10"
	"github.com/aws/jsii-runtime-go"
)

type MyApplicationStackProps struct {
	awscdk.StackProps
}

func NewMyApplicationStack(scope constructs.Construct, id string, props *MyApplicationStackProps) awscdk.Stack {
	var sprops awscdk.StackProps
	if props != nil {
		sprops = props.StackProps
	}
	stack := awscdk.NewStack(scope, &id, &sprops)

	// Create an S3 bucket
	bucket := awss3.NewBucket(stack, jsii.String("MyAppBucket"), &awss3.BucketProps{
		BucketName:           jsii.String(fmt.Sprintf("my-app-bucket-%s-%s", *stack.Account(), *stack.Region())),
		Versioned:            jsii.Bool(true),
		RemovalPolicy:        awscdk.RemovalPolicy_DESTROY,        // For dev/test only
		AutoDeleteObjects:    jsii.Bool(true),                     // For dev/test only
		Encryption:           awss3.BucketEncryption_S3_MANAGED,
		BlockPublicAccess:    awss3.BlockPublicAccess_BLOCK_ALL(),
		EnforceSSL:          jsii.Bool(true),
	})

	// Create a DynamoDB table
	table := awsdynamodb.NewTable(stack, jsii.String("MyAppTable"), &awsdynamodb.TableProps{
		TableName: jsii.String("my-app-table"),
		PartitionKey: &awsdynamodb.Attribute{
			Name: jsii.String("id"),
			Type: awsdynamodb.AttributeType_STRING,
		},
		BillingMode:   awsdynamodb.BillingMode_PAY_PER_REQUEST,
		RemovalPolicy: awscdk.RemovalPolicy_DESTROY, // For dev/test only
	})

	// Lambda function code
	lambdaCode := `
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
`

	// Create a Lambda function
	lambdaFunction := awslambda.NewFunction(stack, jsii.String("MyAppFunction"), &awslambda.FunctionProps{
		FunctionName: jsii.String("my-app-function"),
		Runtime:      awslambda.Runtime_PYTHON_3_11(),
		Handler:      jsii.String("index.handler"),
		Code:         awslambda.Code_FromInline(jsii.String(lambdaCode)),
		Timeout:      awscdk.Duration_Seconds(jsii.Number(30)),
		MemorySize:   jsii.Number(256),
		Environment: &map[string]*string{
			"BUCKET_NAME": bucket.BucketName(),
			"TABLE_NAME":  table.TableName(),
		},
	})

	// Grant permissions
	bucket.GrantReadWrite(lambdaFunction, nil)
	table.GrantReadWriteData(lambdaFunction)

	// Create an API Gateway
	api := awsapigateway.NewRestApi(stack, jsii.String("MyAppApi"), &awsapigateway.RestApiProps{
		RestApiName: jsii.String("my-app-api"),
		Description: jsii.String("API for my application"),
		DeployOptions: &awsapigateway.StageOptions{
			StageName: jsii.String("dev"),
		},
		DefaultCorsPreflightOptions: &awsapigateway.CorsOptions{
			AllowOrigins: awsapigateway.Cors_ALL_ORIGINS(),
			AllowMethods: awsapigateway.Cors_ALL_METHODS(),
		},
	})

	// Add Lambda integration
	lambdaIntegration := awsapigateway.NewLambdaIntegration(lambdaFunction, &awsapigateway.LambdaIntegrationOptions{
		RequestTemplates: &map[string]*string{
			"application/json": jsii.String(`{ "statusCode": "200" }`),
		},
	})

	// Add API resources and methods
	apiResource := api.Root().AddResource(jsii.String("api"), nil)
	apiResource.AddMethod(jsii.String("GET"), lambdaIntegration, nil)
	apiResource.AddMethod(jsii.String("POST"), lambdaIntegration, nil)

	// Output important values
	awscdk.NewCfnOutput(stack, jsii.String("BucketName"), &awscdk.CfnOutputProps{
		Value:       bucket.BucketName(),
		Description: jsii.String("Name of the S3 bucket"),
	})

	awscdk.NewCfnOutput(stack, jsii.String("TableName"), &awscdk.CfnOutputProps{
		Value:       table.TableName(),
		Description: jsii.String("Name of the DynamoDB table"),
	})

	awscdk.NewCfnOutput(stack, jsii.String("ApiUrl"), &awscdk.CfnOutputProps{
		Value:       api.Url(),
		Description: jsii.String("URL of the API Gateway"),
	})

	awscdk.NewCfnOutput(stack, jsii.String("FunctionName"), &awscdk.CfnOutputProps{
		Value:       lambdaFunction.FunctionName(),
		Description: jsii.String("Name of the Lambda function"),
	})

	return stack
}

func main() {
	defer jsii.Close()

	app := awscdk.NewApp(nil)

	// Get environment from context or use defaults
	envName := getContext(app, "env", "dev")
	account := getContext(app, "account", os.Getenv("CDK_DEFAULT_ACCOUNT"))
	region := getContext(app, "region", os.Getenv("CDK_DEFAULT_REGION"))

	NewMyApplicationStack(app, fmt.Sprintf("MyApplicationStack-%s", envName), &MyApplicationStackProps{
		awscdk.StackProps{
			Env: env(account, region),
			Description: jsii.String(fmt.Sprintf("My Application Stack for %s environment", envName)),
		},
	})

	app.Synth(nil)
}

// Helper function to get context value with default
func getContext(app awscdk.App, key string, defaultValue string) string {
	value := app.Node().TryGetContext(jsii.String(key))
	if value == nil {
		return defaultValue
	}
	return fmt.Sprintf("%v", value)
}

// Helper function to create environment
func env(account, region string) *awscdk.Environment {
	// If account/region are not specified, return nil (use defaults)
	if account == "" && region == "" {
		return nil
	}
	
	return &awscdk.Environment{
		Account: jsii.String(account),
		Region:  jsii.String(region),
	}
}