# ── 1. Move into the project ─────────────────────────────────────
cd pulumi-2                       # directory that contains main.go

# 1) See what region the AWS CLI is using (prints nothing if unset)
aws configure get region

# 2) Tell Pulumi to use that same region
#    – if the line above printed, for example, us-west-2:
pulumi config set aws:region us-west-2

# (If the previous command printed nothing, pick a region you actually use
#  and substitute it in the command above.)

# 3) Clean out the bad blank value, just in case
pulumi config rm aws:region --path 2>/dev/null || true   # ignore “not found” warning
pulumi config set aws:region us-west-2                   # repeat with your region

# 4) Try the deployment again
pulumi up
