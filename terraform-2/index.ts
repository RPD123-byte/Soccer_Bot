import * as pulumi from "@pulumi/pulumi";
import * as aws from "@pulumi/aws";

const bucket = new aws.s3.Bucket("myBucket", {
    acl: "private",
    tags: {
        project: "pulumi-sample",
    },
});

export const bucketName = bucket.id;
