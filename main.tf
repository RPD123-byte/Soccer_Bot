#########################################
# File: main.tf
#########################################

provider "aws" {
  region = var.aws_region
}

# Create a VPC for the EKS cluster
resource "aws_vpc" "eks_vpc" {
  cidr_block = "10.0.0.0/16"
}

# Create subnets in two availability zones
resource "aws_subnet" "eks_subnet" {
  count             = 2
  vpc_id            = aws_vpc.eks_vpc.id
  cidr_block        = cidrsubnet(aws_vpc.eks_vpc.cidr_block, 8, count.index)
  availability_zone = "${var.aws_region}${count.index == 0 ? "a" : "b"}"
}

# Provision the EKS Cluster using the terraform-aws-modules/eks/aws module
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "~> 20.0"
  
  cluster_name    = var.cluster_name
  cluster_version = var.cluster_version
  subnet_ids      = aws_subnet.eks_subnet[*].id
  vpc_id          = aws_vpc.eks_vpc.id
  eks_managed_node_groups = {
    example = {
      instance_types   = var.instance_types
      desired_capacity = var.node_count
      min_capacity     = var.node_count
      max_capacity     = var.node_count
    }
  }
}

# Create an ECR repository for container images
resource "aws_ecr_repository" "app_repo" {
  name = "example-eks-repo"
}

# New: Fetch EKS cluster details using aws eks describe-cluster (data block)

data "aws_eks_cluster" "example" {
  name = var.cluster_name
}

# Configure the Kubernetes provider to interact with the EKS cluster using the data block
provider "kubernetes" {
  host                   = data.aws_eks_cluster.example.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.example.certificate_authority.0.data)
  exec {
    api_version = "client.authentication.k8s.io/v1alpha1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", var.cluster_name]
  }
}

# Example Horizontal Pod Autoscaler (HPA) for a deployment named "example-deployment"
resource "kubernetes_horizontal_pod_autoscaler" "example_hpa" {
  metadata {
    name      = "example-hpa"
    namespace = "default"
  }
  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = "soccer-bot-deployment"
    }
    min_replicas = 2
    max_replicas = 5
    metric {
      type = "Resource"
      resource {
        name = "cpu"
        target {
          type                = "Utilization"
          average_utilization = 50
        }
      }
    }
  }
}

# RBAC Configuration: Create a Role, Service Account, and RoleBinding
resource "kubernetes_role" "example_role" {
  metadata {
    name = "example-role"
  }
  rule {
    api_groups = [""]
    resources  = ["pods"]
    verbs      = ["get", "list", "watch"]
  }
}


resource "kubernetes_role_binding" "example_rb" {
  metadata {
    name = "example-role-binding"
  }
  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "Role"
    name      = kubernetes_role.example_role.metadata[0].name
  }
  subject {
    kind      = "ServiceAccount"
    name      = kubernetes_service_account.example_sa.metadata[0].name
    namespace = "default"
  }
}

# Network Policy: Restrict ingress to pods with label app=example-app from pods labeled allowed-app
resource "kubernetes_network_policy" "example_np" {
  metadata {
    name = "deny-all-except-allowed"
    namespace = "default"
  }
  spec {
    pod_selector {
      match_labels = {
        app = "example-app"
      }
    }
    ingress {
      from {
        pod_selector {
          match_labels = {
            app = "allowed-app"
          }
        }
      }
      ports {
        port = 80
      }
    }
    policy_types = ["Ingress"]
  }
}

# Prometheus Monitoring: Deploy kube-prometheus-stack via a Helm release
resource "helm_release" "prometheus" {
  name             = "prometheus"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  namespace        = "monitoring"
  create_namespace = true
  values = [
    <<EOF
alertmanager:
  alertmanagerSpec:
    route:
      group_by: ['alertname']
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 1h
    config: |
      global:
        resolve_timeout: 5m
      route:
        receiver: 'default-receiver'
        group_wait: 30s
        group_interval: 5m
        repeat_interval: 1h
      receivers:
      - name: 'default-receiver'
        email_configs:
        - to: 'your-email@example.com'

prometheus:
  prometheusSpec:
    ruleSelectorNilUsesHelmValues: false

additionalPrometheusRules:
  custom:
    groups:
      - name: custom.rules
        rules:
          - alert: HighCpuUsage
            expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 75
            for: 5m
            labels:
              severity: critical
            annotations:
              summary: "High CPU usage detected"
              description: "CPU usage is over 75% on instance {{ $labels.instance }}"
          - alert: HighMemoryUsage
            expr: (node_memory_Active_bytes / node_memory_MemTotal_bytes * 100) > 75
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "High Memory usage detected"
              description: "Memory usage is over 75% on instance {{ $labels.instance }}"
EOF
  ]
}

# RBAC IAM Role and Policy for EKS Cluster

resource "aws_iam_role" "example_eks_cluster_rbac_role" {
  name = "example-eks-cluster-rbac-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Principal = {
          Federated = "REPLACE_WITH_OIDC_PROVIDER_ARN"
        },
        Action = "sts:AssumeRoleWithWebIdentity",
        Condition = {
          StringEquals = {
            "REPLACE_WITH_OIDC_PROVIDER_URL:sub": "system:serviceaccount:default:example-sa"
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "example_eks_cluster_rbac_policy" {
  name = "example-eks-cluster-rbac-policy"
  role = aws_iam_role.example_eks_cluster_rbac_role.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "s3:GetObject"
        ],
        Resource = "*"
      }
    ]
  })
}

# Update Kubernetes ServiceAccount to link with the IAM role using annotation
resource "kubernetes_service_account" "example_sa" {
  metadata {
    name = "example-sa"
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.example_eks_cluster_rbac_role.arn
    }
  }
}

#########################################
# End of main.tf
#########################################


