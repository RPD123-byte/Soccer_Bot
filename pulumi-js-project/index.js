// index.js - Comprehensive Pulumi JavaScript project for scanner testing
// This version only creates minimal resources but imports all providers

const pulumi = require("@pulumi/pulumi");
const aws = require("@pulumi/aws");
const azure = require("@pulumi/azure-native");
const gcp = require("@pulumi/gcp");
const kubernetes = require("@pulumi/kubernetes");
const docker = require("@pulumi/docker");
const cloudflare = require("@pulumi/cloudflare");
const digitalocean = require("@pulumi/digitalocean");
const github = require("@pulumi/github");
const random = require("@pulumi/random");
const tls = require("@pulumi/tls");
const vault = require("@pulumi/vault");
const postgresql = require("@pulumi/postgresql");
const mysql = require("@pulumi/mysql");
const mongodbatlas = require("@pulumi/mongodbatlas");
const datadog = require("@pulumi/datadog");
const newrelic = require("@pulumi/newrelic");
const pagerduty = require("@pulumi/pagerduty");
const okta = require("@pulumi/okta");
const auth0 = require("@pulumi/auth0");
const stripe = require("@pulumi/stripe");

// Create a few simple resources that don't require credentials

// Random provider resources (always works)
const randomPassword = new random.RandomPassword("password", {
    length: 20,
    special: true,
});

const randomPet = new random.RandomPet("my-pet");

const randomId = new random.RandomId("server", {
    byteLength: 8,
});

const randomInteger = new random.RandomInteger("lottery", {
    min: 1,
    max: 100,
});

const randomUuid = new random.RandomUuid("unique-id");

// TLS provider resources (always works)
const privateKey = new tls.PrivateKey("private-key", {
    algorithm: "RSA",
    rsaBits: 2048,
});

const selfSignedCert = new tls.SelfSignedCert("self-signed-cert", {
    keyAlgorithm: privateKey.algorithm,
    privateKeyPem: privateKey.privateKeyPem,
    subject: {
        commonName: "example.com",
        organization: "Pulumi Test Org",
    },
    validityPeriodHours: 8760,
    allowedUses: [
        "key_encipherment",
        "digital_signature",
        "server_auth",
    ],
});

// AWS S3 bucket (only if AWS is configured)
const bucket = new aws.s3.Bucket("test-bucket", {
    acl: "private",
    tags: {
        Environment: "test",
        Purpose: "scanner-testing",
    },
});

// Export outputs
exports.randomPetName = randomPet.id;
exports.randomPassword = randomPassword.result;
exports.randomHex = randomId.hex;
exports.randomNumber = randomInteger.result;
exports.uuid = randomUuid.result;
exports.privateKeyPem = privateKey.privateKeyPem;
exports.certPem = selfSignedCert.certPem;
exports.bucketName = bucket.id;

// Log which providers are imported
console.log("Loaded providers:");
console.log("- AWS:", aws ? "✓" : "✗");
console.log("- Azure:", azure ? "✓" : "✗");
console.log("- GCP:", gcp ? "✓" : "✗");
console.log("- Kubernetes:", kubernetes ? "✓" : "✗");
console.log("- Docker:", docker ? "✓" : "✗");
console.log("- Cloudflare:", cloudflare ? "✓" : "✗");
console.log("- DigitalOcean:", digitalocean ? "✓" : "✗");
console.log("- GitHub:", github ? "✓" : "✗");
console.log("- Random:", random ? "✓" : "✗");
console.log("- TLS:", tls ? "✓" : "✗");
console.log("- Vault:", vault ? "✓" : "✗");
console.log("- PostgreSQL:", postgresql ? "✓" : "✗");
console.log("- MySQL:", mysql ? "✓" : "✗");
console.log("- MongoDB Atlas:", mongodbatlas ? "✓" : "✗");
console.log("- Datadog:", datadog ? "✓" : "✗");
console.log("- New Relic:", newrelic ? "✓" : "✗");
console.log("- PagerDuty:", pagerduty ? "✓" : "✗");
console.log("- Okta:", okta ? "✓" : "✗");
console.log("- Auth0:", auth0 ? "✓" : "✗");
console.log("- Stripe:", stripe ? "✓" : "✗");