aws_region      = "ap-south-1"
cluster_name    = "example-eks-cluster"
cluster_version = "1.21"
environment     = "dev"

# Default is false, only set to true if you want to deploy Prometheus
deploy_prometheus = false

# Existing IAM role ARN
existing_role_arn = "arn:aws:iam::980921723213:role/example-eks-cluster-rbac-role"

# OIDC provider information
oidc_provider_arn = "arn:aws:iam::980921723213:oidc-provider/oidc.eks.ap-south-1.amazonaws.com/id/CE335C9BE726DBAE4CBE476E53821C7F"
oidc_provider_url = "oidc.eks.ap-south-1.amazonaws.com/id/CE335C9BE726DBAE4CBE476E53821C7F"
