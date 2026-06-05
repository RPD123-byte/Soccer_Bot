using Pulumi;
using Pulumi.Aws.S3;
using System.Threading.Tasks;

class MyStack : Stack
{
    public MyStack()
    {
        var bucket = new Bucket("myBucket", new BucketArgs
        {
            Acl = "private",
            Tags = 
            {
                { "project", "pulumi-sample" },
            },
        });

        this.BucketName = bucket.Id;
    }

    [Output] public Output<string> BucketName { get; set; }
}

class Program
{
    static Task<int> Main() => Deployment.RunAsync<MyStack>();
}
