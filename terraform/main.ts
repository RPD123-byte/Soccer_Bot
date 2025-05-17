// main.ts

import { Construct } from "constructs";
import { App, TerraformStack, TerraformOutput, TerraformProvider } from "cdktf";
import { AwsProvider, s3, dynamodb } from "@cdktf/provider-aws";

class MyStack extends TerraformStack {
  constructor(scope: Construct, id: string) {
    super(scope, id);

    // 1️⃣ Configure AWS provider
    new AwsProvider(this, "aws", {
      region: "us-east-1",
    });

    // 2️⃣ Create an S3 bucket
    const bucket = new s3.S3Bucket(this, "MyBucket", {
      bucket: "cdktf-example-bucket-12345",
      acl: "private",
      versioning: {
        enabled: true,
      },
      tags: {
        Environment: "Dev",
        Project: "CDKTFDemo",
      },
    });

    // 3️⃣ Create a DynamoDB table
    const table = new dynamodb.DynamodbTable(this, "MyTable", {
      name: "cdktf-example-table",
      billingMode: "PAY_PER_REQUEST",
      hashKey: "id",
      attribute: [
        {
          name: "id",
          type: "S",
        },
      ],
      tags: {
        Environment: "Dev",
        Project: "CDKTFDemo",
      },
    });

    // 4️⃣ Expose outputs
    new TerraformOutput(this, "bucketName", {
      value: bucket.bucket,
    });
    new TerraformOutput(this, "tableName", {
      value: table.name,
    });
  }
}

const app = new App();
new MyStack(app, "cdktf-aws-demo");
app.synth();
