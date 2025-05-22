# ── 0. Install tools once per machine ───────────────────────────────
brew install python@3.11 pulumi              # CPython 3.11 + Pulumi CLI
aws sts get-caller-identity                  # confirms AWS creds are set

# ── 1. Jump into your project folder ────────────────────────────────
cd pulumi                                    # the dir that holds main.py

# ── 2. Isolate deps ─────────────────────────────────────────────────
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet pulumi==3.* pulumi-aws==6.*

# ── 3. Add Pulumi project metadata (if it isn’t there) ──────────────
cat > Pulumi.yaml <<'EOF'
name: pulumi-python-bucket
runtime: python
description: Creates a private S3 bucket and exports its name.
EOF


# ── 4. Log in to the Pulumi backend you prefer ──────────────────────
pulumi login                                  # pick local or cloud

# ── 5. Create / select a stack called “dev” ─────────────────────────
pulumi stack init dev || pulumi stack select dev

# ── 6. Point Pulumi at your AWS region (uses CLI default) ───────────
pulumi config set aws:region $(aws configure get region)

# ── 7. Deploy the bucket ────────────────────────────────────────────
pulumi up                                     # review ➜ “yes”
