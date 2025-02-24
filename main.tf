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

  managed_node_groups = {
    example = {
      instance_types   = var.instance_types
      desired_capacity = var.node_count
    }
  }
}

# Create an ECR repository for container images
resource "aws_ecr_repository" "app_repo" {
  name = "${var.cluster_name}-repo"
}

# Configure the Kubernetes provider to interact with the EKS cluster
provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data[0].data)
  exec {
    api_version = "client.authentication.k8s.io/v1alpha1"
    command     = "aws"
    args        = ["eks", "get-token", "--cluster-name", module.eks.cluster_id]
  }
}

# Example Horizontal Pod Autoscaler (HPA) for a deployment named "example-deployment"
resource "kubernetes_horizontal_pod_autoscaler" "example_hpa" {
  metadata {
    name = "example-hpa"
  }
  spec {
    scale_target_ref {
      api_version = "apps/v1"
      kind        = "Deployment"
      name        = "example-deployment"
    }
    min_replicas = 2
    max_replicas = 5
    metrics {
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

resource "kubernetes_service_account" "example_sa" {
  metadata {
    name = "example-sa"
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
  }
}

# Prometheus Monitoring: Deploy kube-prometheus-stack via a Helm release
resource "helm_release" "prometheus" {
  name             = "prometheus"
  repository       = "https://prometheus-community.github.io/helm-charts"
  chart            = "kube-prometheus-stack"
  namespace        = "monitoring"
  create_namespace = true
}

#########################################
# End of main.tf
#########################################


