# 1  Add the .NET 8 runtime/SDK
brew install dotnet@8                 # formula is keg-only :contentReference[oaicite:0]{index=0}

# 2  Put its tools first on your PATH
echo 'export DOTNET_ROOT="/opt/homebrew/opt/dotnet@8/libexec"' >> ~/.zshrc
echo 'export PATH="$DOTNET_ROOT:$PATH"'                        >> ~/.zshrc
source ~/.zshrc

# 3  (If the 9-series binary is still first on PATH, unlink it)
brew unlink dotnet 2>/dev/null || true
brew link --force --overwrite dotnet@8      # symlinks dotnet 8 into /opt/homebrew/bin

# 4  Retry the build & deploy
cd ~/Soccer_Bot/pulumi-3
dotnet --list-runtimes     # should now show Microsoft.NETCore.App 8.0.x
dotnet build -nologo       # compile succeeds
pulumi up                  # preview → yes
