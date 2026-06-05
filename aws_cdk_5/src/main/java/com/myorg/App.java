package com.myorg;

import software.amazon.awscdk.App;
import software.amazon.awscdk.Environment;
import software.amazon.awscdk.StackProps;

public class App {
    public static void main(final String[] args) {
        App app = new App();

        String envName = (String) app.getNode().tryGetContext("env");
        if (envName == null) envName = "dev";
        
        String account = (String) app.getNode().tryGetContext("account");
        if (account == null) account = System.getenv("CDK_DEFAULT_ACCOUNT");
        
        String region = (String) app.getNode().tryGetContext("region");
        if (region == null) region = System.getenv("CDK_DEFAULT_REGION");

        StackProps stackProps = StackProps.builder()
                .env(Environment.builder()
                        .account(account)
                        .region(region)
                        .build())
                .description("My Application Stack for " + envName + " environment")
                .build();

        new MyApplicationStack(app, "MyApplicationStack-" + envName, stackProps);

        app.synth();
    }
}