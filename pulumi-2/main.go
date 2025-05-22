package main

import (
	"github.com/pulumi/pulumi-aws/sdk/v6/go/aws/s3"
	"github.com/pulumi/pulumi/sdk/v3/go/pulumi"
)

func main() {
	pulumi.Run(func(ctx *pulumi.Context) error {
		bucket, err := s3.NewBucket(ctx, "myBucket", &s3.BucketArgs{
			Acl: pulumi.String("private"),
			Tags: pulumi.StringMap{
				"project": pulumi.String("pulumi-sample"),
			},
		})
		if err != nil {
			return err
		}
		ctx.Export("bucketName", bucket.ID())
		return nil
	})
}
