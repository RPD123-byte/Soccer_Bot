# __main__.py
import pulumi
import pulumi_aws as aws

# Create an S3 bucket
bucket = aws.s3.Bucket(
    "my-bucket",
    acl="private",
    tags={
        "project": "pulumi-sample",
    },
)

# Export the bucket name
pulumi.export("bucket_name", bucket.bucket)
