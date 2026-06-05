#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { MyApplicationStack } from '../lib/my-application-stack';

const app = new cdk.App();

// Get environment from context or use defaults
const envName = app.node.tryGetContext('env') || 'dev';
const account = app.node.tryGetContext('account') || process.env.CDK_DEFAULT_ACCOUNT;
const region = app.node.tryGetContext('region') || process.env.CDK_DEFAULT_REGION;

new MyApplicationStack(app, `MyApplicationStack-${envName}`, {
  env: {
    account: account,
    region: region,
  },
  description: `My Application Stack for ${envName} environment`,
});