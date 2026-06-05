const pulumi = require("@pulumi/pulumi");
const aws = require("@pulumi/aws");

const bucket = new aws.s3.Bucket("myBucket", {
    acl: "private",
    tags: {
        project: "pulumi-sample",
    },
});

exports.bucketName = bucket.id;
