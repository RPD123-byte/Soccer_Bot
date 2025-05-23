#!/usr/bin/env node
"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
require("source-map-support/register");
const cdk = __importStar(require("aws-cdk-lib"));
const my_application_stack_1 = require("../lib/my-application-stack");
const app = new cdk.App();
// Get environment from context or use defaults
const envName = app.node.tryGetContext('env') || 'dev';
const account = app.node.tryGetContext('account') || process.env.CDK_DEFAULT_ACCOUNT;
const region = app.node.tryGetContext('region') || process.env.CDK_DEFAULT_REGION;
new my_application_stack_1.MyApplicationStack(app, `MyApplicationStack-${envName}`, {
    env: {
        account: account,
        region: region,
    },
    description: `My Application Stack for ${envName} environment`,
});
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiYXBwLmpzIiwic291cmNlUm9vdCI6IiIsInNvdXJjZXMiOlsiYXBwLnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiI7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7O0FBQ0EsdUNBQXFDO0FBQ3JDLGlEQUFtQztBQUNuQyxzRUFBaUU7QUFFakUsTUFBTSxHQUFHLEdBQUcsSUFBSSxHQUFHLENBQUMsR0FBRyxFQUFFLENBQUM7QUFFMUIsK0NBQStDO0FBQy9DLE1BQU0sT0FBTyxHQUFHLEdBQUcsQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLEtBQUssQ0FBQyxJQUFJLEtBQUssQ0FBQztBQUN2RCxNQUFNLE9BQU8sR0FBRyxHQUFHLENBQUMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxTQUFTLENBQUMsSUFBSSxPQUFPLENBQUMsR0FBRyxDQUFDLG1CQUFtQixDQUFDO0FBQ3JGLE1BQU0sTUFBTSxHQUFHLEdBQUcsQ0FBQyxJQUFJLENBQUMsYUFBYSxDQUFDLFFBQVEsQ0FBQyxJQUFJLE9BQU8sQ0FBQyxHQUFHLENBQUMsa0JBQWtCLENBQUM7QUFFbEYsSUFBSSx5Q0FBa0IsQ0FBQyxHQUFHLEVBQUUsc0JBQXNCLE9BQU8sRUFBRSxFQUFFO0lBQzNELEdBQUcsRUFBRTtRQUNILE9BQU8sRUFBRSxPQUFPO1FBQ2hCLE1BQU0sRUFBRSxNQUFNO0tBQ2Y7SUFDRCxXQUFXLEVBQUUsNEJBQTRCLE9BQU8sY0FBYztDQUMvRCxDQUFDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyIjIS91c3IvYmluL2VudiBub2RlXG5pbXBvcnQgJ3NvdXJjZS1tYXAtc3VwcG9ydC9yZWdpc3Rlcic7XG5pbXBvcnQgKiBhcyBjZGsgZnJvbSAnYXdzLWNkay1saWInO1xuaW1wb3J0IHsgTXlBcHBsaWNhdGlvblN0YWNrIH0gZnJvbSAnLi4vbGliL215LWFwcGxpY2F0aW9uLXN0YWNrJztcblxuY29uc3QgYXBwID0gbmV3IGNkay5BcHAoKTtcblxuLy8gR2V0IGVudmlyb25tZW50IGZyb20gY29udGV4dCBvciB1c2UgZGVmYXVsdHNcbmNvbnN0IGVudk5hbWUgPSBhcHAubm9kZS50cnlHZXRDb250ZXh0KCdlbnYnKSB8fCAnZGV2JztcbmNvbnN0IGFjY291bnQgPSBhcHAubm9kZS50cnlHZXRDb250ZXh0KCdhY2NvdW50JykgfHwgcHJvY2Vzcy5lbnYuQ0RLX0RFRkFVTFRfQUNDT1VOVDtcbmNvbnN0IHJlZ2lvbiA9IGFwcC5ub2RlLnRyeUdldENvbnRleHQoJ3JlZ2lvbicpIHx8IHByb2Nlc3MuZW52LkNES19ERUZBVUxUX1JFR0lPTjtcblxubmV3IE15QXBwbGljYXRpb25TdGFjayhhcHAsIGBNeUFwcGxpY2F0aW9uU3RhY2stJHtlbnZOYW1lfWAsIHtcbiAgZW52OiB7XG4gICAgYWNjb3VudDogYWNjb3VudCxcbiAgICByZWdpb246IHJlZ2lvbixcbiAgfSxcbiAgZGVzY3JpcHRpb246IGBNeSBBcHBsaWNhdGlvbiBTdGFjayBmb3IgJHtlbnZOYW1lfSBlbnZpcm9ubWVudGAsXG59KTsiXX0=