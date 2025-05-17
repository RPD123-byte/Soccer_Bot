terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 4.0, < 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.50"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.14"
    }
  }
}

# AWS provider
provider "aws" {
  region = "us-east-1"
}

# AzureRM provider
provider "azurerm" {
  features {}
  subscription_id = var.azure_subscription_id
  tenant_id       = var.azure_tenant_id
}

# Google provider
provider "google" {
  project = var.gcp_project
  region  = "us-central1"
  zone    = "us-central1-a"
}

# Kubernetes provider
provider "kubernetes" {
  host                   = var.k8s_host
  token                  = var.k8s_token
  cluster_ca_certificate = base64decode(var.k8s_ca_cert)
}
