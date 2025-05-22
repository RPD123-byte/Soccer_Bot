"""
ProviderScanner
───────────────
Walks a checked-out codebase **inside the DevOps VM** and extracts any
provider + version pairs it can see.

Currently implemented:
• Terraform HCL (`*.tf`)   →  provider block + `version = "..."`
• CDKTF  (`*.py`, `*.ts`) →  `cdktf_cdktf_provider_*`  imports

Returned structure (now partitioned by folder):
{
  "terraform": {
    ".":              { "aws": "v5.97.0" },
    "prod":           { "aws": "v5.97.0", "helm": "v2.12.1" },
    "staging":        { "aws": "v5.90.0" },
    "infra/modules":  { "azurerm": "v3.92.0" }
  }
}
"""

from __future__ import annotations

import os
import re
import json
from collections import defaultdict
import requests
from packaging import version as _semver
from functools import lru_cache

_TF_PROVIDER_RE = re.compile(
    r'provider\s+"(?P<name>[^"]+)"[^{]*{[^}]*?\n[^}]*?version\s*=\s*["\']\s*(?P<ver>[^"\']+)',
    re.S,
)

_TF_SIMPLE_PROVIDER_RE = re.compile(
    r'provider\s+"(?P<name>[^"]+)"', re.M
)

# classic js/ts CDKTF ≤0.19 and the new scoped package ≥0.20
_TF_CDK_RE = re.compile(
    r"""
        (?:                        # either old ..
           cdktf_cdktf_provider_(?P<name1>\w+) .*? ["'] (?P<ver1>\d+\.\d+\.\d+)
        |                          # …or new scoped package
           @cdktf/provider-(?P<name2>\w+)        # no version in import
        )
    """,
    re.S | re.VERBOSE,
)

# new: terraform { required_providers { aws = { version = "~> 5.97" } } }
_TF_REQUIRED_RE = re.compile(
    r"required_providers\s*{[^}]*?(?P<name>\w+)\s*=\s*{[^}]*?version\s*=\s*\"(?P<ver>[^\"]+)\"",
    re.S,
)

# Pulumi-go imports: "github.com/pulumi/pulumi-aws/sdk/v6/go/aws/s3"
_PULUMI_IMPORT_RE = re.compile(
    r'"github\.com/pulumi/pulumi-(?P<prov>\w+)/sdk/v(?P<major>\d+)"'
)

# go.mod provider lines (may be indented with tabs/spaces)
_GO_MOD_PROVIDER_RE = re.compile(
    r'^\s*github\.com/pulumi/pulumi-(?P<prov>\w+)/sdk/v\d+\s+(?P<ver>v[^\s]+)',
    re.M,
)

# Pulumi ∙ .NET  ────────────────────────────────────────────────────
# <PackageReference Include="Pulumi.Aws" Version="6.*" />
# group “prov” captures the bit after the first dot (Aws, AzureNative…)
_DOTNET_PKG_REF_RE = re.compile(
    r'<PackageReference\s+Include\s*=\s*"Pulumi\.(?P<prov>[A-Za-z0-9]+)"\s+Version\s*=\s*"(?P<ver>[^"]+)"',
    re.I | re.S,
)

# ── Pulumi-Python regexes ────────────────────────────────────────────
#   import  pulumi_aws   → "aws"
#   pulumi-aws==6.81.0   → "aws"  +  "v6.81.0"
_PY_IMPORT_RE = re.compile(r'\bimport\s+pulumi_(?P<prov>\w+)\b')
_REQ_LINE_RE  = re.compile(
    r'pulumi-(?P<prov>[a-z0-9_]+)[-_~=<>!*]*\s*(?P<ver>[0-9][^\s;]+)?',
    re.I,
)

# ── Pulumi-TypeScript/Node.js regexes ─────────────────────────────────
# import * as aws from "@pulumi/aws"
_TS_IMPORT_RE = re.compile(
    r'from\s+["\']@pulumi/(?P<prov>[a-z0-9-]+)["\']',
    re.I
)
# const aws = require("@pulumi/aws")
_JS_REQUIRE_RE = re.compile(
    r'require\s*\(\s*["\']@pulumi/(?P<prov>[a-z0-9-]+)["\']',
    re.I
)
# package.json dependencies: "@pulumi/aws": "^6.81.0"
_PKG_JSON_DEP_RE = re.compile(
    r'"@pulumi/(?P<prov>[a-z0-9-]+)"\s*:\s*"(?P<ver>[^"]+)"',
    re.I
)

# new: .terraform.lock.hcl lines like:
# provider "registry.terraform.io/hashicorp/aws" {
#   version = "5.97.0"
# }
_LOCKFILE_RE = re.compile(
    r'provider\s+"[^"]+/(?P<name>[^"]+)"\s*{[^}]*?version\s*=\s*"(?P<ver>[^"]+)"',
    re.S,
)

class ProviderScanner:
    """
    Parameters
    ----------
    vm_manager : VMManager
        The singleton that can exec commands inside pods.
    workspace_path : str
        Where the repo lives inside the container (`/workspace` by default).
    """

    def __init__(self, vm_manager: VMManager, workspace_path: str = "/workspace"):
        self.vm_manager = vm_manager
        self.workspace_path = workspace_path
        # paths we’ll search for collections
        self._venv_bin       = os.path.join(self.workspace_path, ".venv", "bin")
        self._collection_dir = os.path.join(self.workspace_path, ".ansible", "collections")

        # all known search roots, in priority order
        self._collection_paths: list[str] = [
            os.environ.get("ANSIBLE_COLLECTIONS_PATHS", ""),     # user-defined
            self._collection_dir,                                 # repo-local (CI)
            "$HOME/.ansible/collections",                         # per-user default
            "/usr/share/ansible/collections",                     # system default
        ]
        print(f"[ProviderScanner] initialised:")
        print(f"   workspace_path   = {self.workspace_path}")
        print(f"   _venv_bin        = {self._venv_bin}")
        print(f"   _collection_dir  = {self._collection_dir}")

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def scan_state(self, state: dict,) -> dict[str, dict[str, list[str]]]:
        """
        Choose the right pod (workflow-level first, then project-level) and
        return the aggregated provider map *limited to the current repo*.
        If no pod is available, returns an empty dict.
        """
        inst = None
        if state.get("workflow_id"):
            vm = self.vm_manager.get_workflow_vm(state["workflow_id"])
            inst = vm.instance_id if vm else None
        if not inst and state.get("project_id"):
            vm = self.vm_manager.get_project_vm(state["project_id"])
            inst = vm.instance_id if vm else None

        if not inst:
            return {}

        # ── limit the scan to the repo we’re actually interested in ──
        # The orchestrator puts the repo’s absolute path in state["repo_path"],
        # e.g.  /workspace/RPD123-byte/Soccer_Bot
        repo_root = state.get("repo_path")
        if repo_root:
            # drop any trailing “/” so os.path.relpath() behaves nicely
            self.workspace_path = repo_root.rstrip("/")
            # ── keep helper paths in sync ───────────────────────────────
            self._venv_bin       = os.path.join(self.workspace_path, ".venv", "bin")
            self._collection_dir = os.path.join(self.workspace_path, ".ansible", "collections")
            self._collection_paths[1] = self._collection_dir 

        return self.scan_vm(instance_id=inst)

    # ------------------------------------------------------------------ #
    # core scan                                                          #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # main entry – run inside a VM                                       #
    # ------------------------------------------------------------------ #
    def scan_vm(self, *, instance_id: str) -> dict[str, dict[str, dict[str, str]]]:
        """
        Return a map **partitioned by Terraform-root folder**:

        {
          "terraform": {
              ".":              {"aws": "v5.97.0"},
              "prod":           {"aws": "v5.97.0", "helm": "v2.12.1"},
              "infra/modules":  {"azurerm": "v3.92.0"}
          }
        }
        """
        providers: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)

        # ──────────────────────────────────────────────────────────────
        # 🐍  Detect ansible-core + installed collections (once per scan)
        # ──────────────────────────────────────────────────────────────
        # ① core version ------------------------------------------------
        core_line = (
            self._run(instance_id, "ansible --version 2>/dev/null | head -n1 || true")
            .strip()
        )
        m = re.search(r"\[core\s+([\d.]+)\]", core_line)
        if m:
            providers["ansible_core"] = m.group(1)

        # ② installed collections --------------------------------------
        # ── build a colon-separated path list from unique, non-empty entries
        paths_arg = ":".join(p for p in self._collection_paths if p)

        print(f"\n[ProviderScanner] ansible-galaxy list for paths: {paths_arg}")
        galaxy_json = self._run(
            instance_id,
            f'ansible-galaxy collection list -p "{paths_arg}" --format json 2>/dev/null || true',
        )

        # ── DEBUG: see exactly what the CLI returned ───────────────────
        print("\n[DEBUG] raw ansible-galaxy JSON:")
        print(galaxy_json)
        print("--------------------------------------------------------\n")

        try:
            data = json.loads(galaxy_json)
        except Exception:
            data = None

        coll_map: dict[str, str] = {}

        if isinstance(data, list):                       # old schema
            for c in data:
                fqcn = f"{c.get('namespace')}.{c.get('name')}"
                ver  = c.get("version")
                if fqcn and ver:
                    coll_map[fqcn] = f"v{ver.lstrip('v')}"

        elif isinstance(data, dict):                    # new schemas
            # A) ansible-core ≥ 2.12 default: { "/path": { "fqcn": {version:…} } }
            for base, colls in data.items():
                if isinstance(colls, dict) and all(isinstance(v, dict) for v in colls.values()):
                    for fqcn, meta in colls.items():
                        ver = meta.get("version") or meta.get("installed_version")
                        if ver:
                            coll_map[fqcn] = f"v{ver.lstrip('v')}"
                    continue   # handled this entry; go to next

            # B) flat fqcn → meta (value has version field)
            for fqcn, meta in data.items():
                if isinstance(meta, dict):
                    ver = meta.get("version") or meta.get("installed_version")
                    if ver:
                        coll_map[fqcn] = f"v{ver.lstrip('v')}"

            # B) ansible-galaxy –-list JSON from AWX/older plugins: { "collections": { … } }
            for base, colls in data.get("collections", {}).items():
                for fqcn, meta in colls.items():
                    ver = meta.get("version") or meta.get("installed_version")
                    if ver:
                        coll_map[fqcn] = f"v{ver.lstrip('v')}"

        if coll_map: 
            providers["ansible"]["."] = coll_map

        # ── 1️⃣ find every directory that contains at least one *.tf file ──
        #     but **ignore** any Terraform files shipped inside downloaded
        #     Ansible collections to avoid noise.
        # ── Get files first, then use Python's os.path.dirname ─────────
        hcl_files_cmd = (
            f"find {self.workspace_path} "
            "! -path '*/.ansible/collections/*' "
            "-type f -name '*.tf'"
        )
        hcl_files = self._run(instance_id, hcl_files_cmd).splitlines()
        hcl_dirs = {os.path.dirname(f) for f in hcl_files if f}

        # ── 2️⃣ find every directory that contains at least one CDKTF file ──
        cdk_dirs_cmd = (
            f"grep -R -l -E '@cdktf/provider-|cdktf_cdktf_provider_' {self.workspace_path} "
            "--include='*.ts' --include='*.py' || true"
        )
        cdk_dirs = {os.path.dirname(p) for p in self._run(instance_id, cdk_dirs_cmd).splitlines() if p}


        all_dirs: set[str] = hcl_dirs | cdk_dirs                          

        # ── 2b️⃣ detect Pulumi-Go project roots ───────────────────────────
        pulumi_dirs_cmd = (
            f"grep -R -l -E 'github.com/pulumi/pulumi-[^/]+/sdk/v[0-9]+' "
            f"{self.workspace_path} --include='*.go' || true"
        )
        pulumi_dirs = {
            os.path.dirname(p)
            for p in self._run(instance_id, pulumi_dirs_cmd).splitlines()
            if p and ".pulumi" not in p
        }

        all_dirs |= pulumi_dirs

        # ── 2c️⃣ Pulumi ∙ .NET roots ─────────────────────────────────────
        #   * -iname  → case-insensitive ("*.CSPROJ", "*.csProj", …)
        #   * Get files first, then use Python's dirname
        dotnet_files_cmd = (
            f"find {self.workspace_path} "
            "-type f \\( -iname '*.csproj' -o -iname '*.fsproj' \\)"
        )
        dotnet_files = self._run(instance_id, dotnet_files_cmd).splitlines()
        pulumi_dotnet_dirs = {os.path.dirname(f) for f in dotnet_files if f}

        # ADD THIS LINE - it's missing in your code!
        all_dirs |= pulumi_dotnet_dirs

        # ── 2d️⃣ detect Pulumi-Python project roots ───────────────────────
        py_dirs_cmd = (
            f"grep -R -l -E '\\bimport\\s+pulumi_(\\w+)' {self.workspace_path} "
            "--include='*.py' || true"
        )
        pulumi_py_dirs = {
            os.path.dirname(p)
            for p in self._run(instance_id, py_dirs_cmd).splitlines()
            if p
        }
        all_dirs |= pulumi_py_dirs

        # ── 2e️⃣ detect Pulumi-TypeScript/Node.js project roots ────────────
        # Look for Pulumi.yaml files (most reliable)
        pulumi_yaml_cmd = f"find {self.workspace_path} -name 'Pulumi.yaml' -o -name 'Pulumi.yml'"
        pulumi_ts_dirs = {
            os.path.dirname(p)
            for p in self._run(instance_id, pulumi_yaml_cmd).splitlines()
            if p
        }
        all_dirs |= pulumi_ts_dirs

        if not all_dirs:
            print("[ProviderScanner] 🔍 no IaC roots (Terraform/CDKTF/Pulumi) found")
            return {}

        # pre-create a *writable* plugin cache inside the repo
        plugin_cache = os.path.join(self.workspace_path, ".terraform-plugin-cache")
        self._run(instance_id, f"mkdir -p {plugin_cache} || true")

        # ── 2️⃣ scan every TF root ───────────────────────────────────────
        for abs_dir in sorted(all_dirs):
            rel_dir = os.path.relpath(abs_dir, self.workspace_path)  # "prod", ".", …
            print(f"[ProviderScanner] ▶ scanning dir: {rel_dir}")

            # prepare an empty CLI config so any global ~/.terraformrc is ignored
            cli_rc = os.path.join(abs_dir, ".terraform.empty.rc")
            self._run(instance_id, f"touch {cli_rc} || true")

            env = (
                f"TF_DATA_DIR=.terraform-data "
                f"TF_PLUGIN_CACHE_DIR={plugin_cache} "
                f"TF_CLI_CONFIG_FILE={cli_rc} "
            )

            has_hcl   = abs_dir in hcl_dirs
            is_cdk_dir = abs_dir in cdk_dirs

            # ── build provider map ───────────────────────────────────────
            prov_map: dict[str, str] = {}

            # ① lock-file parse (only for real HCL roots)
            if has_hcl:
                lock_cmd = (
                    f"cd {abs_dir} && "
                    f"{env} terraform providers lock "
                    "-platform=linux_amd64 -platform=linux_arm64 "
                    "-platform=darwin_amd64 -platform=windows_amd64 "
                    "-verify-plugins=false -no-color || true"
                )
                self._run(instance_id, lock_cmd)

                lock_txt = self._run(instance_id, f"cat {abs_dir}/.terraform.lock.hcl || true")
                prov_map.update(self._parse_lockfile(lock_txt))

            # ② heuristic grep (covers required_providers *and* CDK imports)
            prov_map = {**self._grep_fallback(instance_id, abs_dir), **prov_map}

            # ③ record under the correct top-level keys
            if has_hcl and prov_map:
                providers["terraform"][rel_dir] = prov_map
            if is_cdk_dir:
                providers["terraform_cdk"][rel_dir] = prov_map

            if not prov_map and not is_cdk_dir:
                print(f"   • no providers found for dir {rel_dir}")
            
            # ── ④ Pulumi-Go scan (independent of HCL/CDK) ────────────────
            if abs_dir in pulumi_dirs:
                pulumi_map = self._scan_pulumi_go(instance_id, abs_dir)
                if pulumi_map:
                    providers.setdefault("pulumi", {}).setdefault(rel_dir, {}).update(pulumi_map)

            # ── ⑤ Pulumi-Python scan ─────────────────────────────────────
            if abs_dir in pulumi_py_dirs:
                py_map = self._scan_pulumi_python(instance_id, abs_dir)
                if py_map:
                    providers.setdefault("pulumi", {}).setdefault(rel_dir, {}).update(py_map)
 
            # ── ⑥ Pulumi ∙ .NET scan ─────────────────────────────────────
            if abs_dir in pulumi_dotnet_dirs:
                dn_map = self._scan_pulumi_dotnet(instance_id, abs_dir)
                if dn_map:
                    providers.setdefault("pulumi", {}).setdefault(rel_dir, {}).update(dn_map)
                else:
                    # Still record the directory even if no providers found
                    # (you can remove this if you only want dirs with providers)
                    providers.setdefault("pulumi", {})[rel_dir] = {}

            # ── ⑦ Pulumi-TypeScript/Node.js scan ─────────────────────────
            if abs_dir in pulumi_ts_dirs:
                ts_map = self._scan_pulumi_typescript(instance_id, abs_dir)
                if ts_map:
                    providers.setdefault("pulumi", {}).setdefault(rel_dir, {}).update(ts_map)

        return providers

    # ------------------------------------------------------------------ #
    # helper: parse .terraform.lock.hcl                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_lockfile(text: str) -> dict[str, str]:
        """
        Extract provider ⇒ version pairs from a `.terraform.lock.hcl`.
        Returns an empty dict if nothing could be parsed.
        """
        out: dict[str, str] = {}
        for m in re.finditer(
            r'provider\s+"[^"]+/(?P<name>[^"]+)"\s*{[^}]*?version\s*=\s*"(?P<ver>[^"]+)"',
            text,
            flags=re.S,
        ):
            out[m["name"]] = ProviderScanner._norm(m["ver"])
        return out

    # ------------------------------------------------------------------ #
    # Pulumi-Go helpers                                                  #
    # ------------------------------------------------------------------ #
    def _scan_pulumi_go(self, instance_id: str, dir_path: str) -> dict[str, str]:
        """
        Detect Pulumi providers in a *single* Go module directory.
        Cascade:
          1) go.mod provider lines              → exact semver
          2) import path majors                 → latest patch for that major
        """
        # ① try go.mod (authoritative once committed)
        gm = self._run(instance_id, f"cat {dir_path}/go.mod 2>/dev/null || true")
        prov_map: dict[str, str] = {}
        for m in _GO_MOD_PROVIDER_RE.finditer(gm):
            prov_map[m["prov"]] = ProviderScanner._norm(m["ver"])

        # ② if nothing yet, fall back to import scans
        if not prov_map:
            imp_cmd = (
                f"grep -R --include='*.go' -E 'github.com/pulumi/pulumi-[^/]+/sdk/v[0-9]+' "
                f"{dir_path} || true"
            )
            for line in self._run(instance_id, imp_cmd).splitlines():
                m = _PULUMI_IMPORT_RE.search(line)
                if not m:
                    continue
                name, major = m["prov"], m["major"]
                prov_map[name] = self._latest_from_pulumi_registry(name, major) or f"v{major}.0.0"
        return prov_map

    # -------------------------------------------------------------------- #
    # Pulumi-Python helpers                                                #
    # -------------------------------------------------------------------- #
    def _scan_pulumi_python(self, instance_id: str, dir_path: str) -> dict[str, str]:
        """
        Detect Pulumi providers in a *Python* Pulumi project.

        Cascade:
          1. Pulumi.lock.yaml                → exact plugin versions
          2. requirements*/poetry.lock/…     → SDK versions or ranges
          3. source import scan              → fallback to latest patch
        """
        prov_map: dict[str, str] = {}

        # ① Pulumi.lock.yaml ------------------------------------------------
        lock_txt = self._run(
            instance_id, f"cat {dir_path}/Pulumi.lock.yaml 2>/dev/null || true"
        )
        for m in re.finditer(
            r'name:\s+"?(?P<prov>\w+)"?[^\n]*\n\s*version:\s+"?(?P<ver>[^"\n]+)',
            lock_txt,
            flags=re.S,
        ):
            prov_map[m["prov"]] = self._norm(m["ver"])
        if prov_map:
            return prov_map

        # ② requirements / Pipfile / poetry.lock / pyproject.toml ----------
        req_cmd = (
            f"grep -R -h -E 'pulumi-[a-z0-9_-]+' {dir_path} "
            "--include='requirements*.txt' --include='Pipfile.lock' "
            "--include='poetry.lock' --include='pyproject.toml' || true"
        )
        for line in self._run(instance_id, req_cmd).splitlines():
            m = _REQ_LINE_RE.search(line)
            if not m:
                continue
            ver = m["ver"]
            if ver:
                prov_map[m["prov"]] = self._norm(f"v{ver.lstrip('v')}")
            else:
                # defer filling-in until after the loop
                prov_map[m["prov"]] = None

        # ③ import-scan fallback ------------------------------------------
        if not prov_map or any(v is None for v in prov_map.values()):
            src_cmd = (
                f"grep -R --include='*.py' -E '\\bimport\\s+pulumi_\\w+' {dir_path} || true"
            )
            for line in self._run(instance_id, src_cmd).splitlines():
                m = _PY_IMPORT_RE.search(line)
                if not m:
                    continue
                prov_map.setdefault(m["prov"], None)

        # fill in any still-unknown versions using PyPI (then registry) ----
        for pkg, ver in list(prov_map.items()):
            if ver:
                continue
            prov_map[pkg] = (
                self._latest_from_pypi(pkg)
                or self._latest_from_pulumi_registry(pkg, "")
                or "vlatest"
            )

        return prov_map    

    # -------------------------------------------------------------------- #
    # Pulumi-TypeScript/Node.js helpers                                    #
    # -------------------------------------------------------------------- #
    def _scan_pulumi_typescript(self, instance_id: str, dir_path: str) -> dict[str, str]:
        """
        Detect Pulumi providers in a TypeScript/Node.js Pulumi project.
        """
        prov_map: dict[str, str] = {}

        # Check package.json for @pulumi dependencies
        pkg_json = self._run(
            instance_id, f"cat {dir_path}/package.json 2>/dev/null || true"
        )
        if pkg_json:
            for m in _PKG_JSON_DEP_RE.finditer(pkg_json):
                prov = m["prov"]
                if prov == "pulumi":  # Skip base pulumi package
                    continue
                ver = m["ver"].lstrip("^~>=<")  # Remove version range operators
                prov_map[prov] = self._norm(f"v{ver.lstrip('v')}")

        # Fallback: scan imports if no package.json
        if not prov_map:
            # Look for both ES6 imports and CommonJS requires
            src_cmd = (
                f"grep -R -E '(from\\s+[\"\\']@pulumi/|require\\s*\\([\"\\']@pulumi/)' {dir_path} "
                "--include='*.ts' --include='*.tsx' --include='*.js' --include='*.jsx' || true"
            )
            for line in self._run(instance_id, src_cmd).splitlines():
                if "node_modules" in line:
                    continue
                
                # Try ES6 import first
                m = _TS_IMPORT_RE.search(line)
                if m and m["prov"] != "pulumi":
                    prov_map[m["prov"]] = "vlatest"
                    continue
                
                # Try CommonJS require
                m = _JS_REQUIRE_RE.search(line)
                if m and m["prov"] != "pulumi":
                    prov_map[m["prov"]] = "vlatest"

        return prov_map

   # -------------------------------------------------------------------- #
    # Pulumi ∙ .NET helpers                                                #
    # -------------------------------------------------------------------- #
    def _scan_pulumi_dotnet(self, instance_id: str, dir_path: str) -> dict[str, str]:
        """
        Detect Pulumi provider packages in a .NET project directory.
        Looks at all *.csproj / *.fsproj files inside *dir_path*.
        """
        proj_files_cmd = (
            f"ls {dir_path}/*.csproj {dir_path}/*.fsproj 2>/dev/null || true"
        )
        prov_map: dict[str, str] = {}
        for pf in self._run(instance_id, proj_files_cmd).split():
            if not pf:  # Skip empty strings
                continue
                
            content = self._run(instance_id, f"cat {pf}")
            
            # Look for Pulumi provider packages (skip base "Pulumi" package)
            for m in _DOTNET_PKG_REF_RE.finditer(content):
                prov_name = m["prov"]
                
                # Skip if this is just the base Pulumi package (no provider)
                if not prov_name:
                    continue
                    
                prov  = prov_name.lower()           # Aws → aws
                ver   = (m["ver"] or "").strip()
                
                if not ver or "*" in ver:           # wildcard / missing → query NuGet
                    major = ver.split(".")[0] if ver and ver[0].isdigit() else ""
                    best  = self._latest_from_nuget(prov, major) or f"v{major}.0.0"
                    prov_map[prov] = best
                else:
                    prov_map[prov] = self._norm(f"v{ver.lstrip('v')}")
        
        return prov_map

    # -------------------------------------------------------------------- #
    # NuGet helper – newest package version                                #
    # -------------------------------------------------------------------- #
    @staticmethod
    @lru_cache(maxsize=128)
    def _latest_from_nuget(provider: str, major: str | None = "") -> str | None:
        """
        Ask NuGet for the newest *Pulumi.<provider>* release.
        If *major* is given, return the newest patch that starts
        with that major (skips prereleases).
        """
        pkg = f"pulumi.{provider}".lower()
        url = f"https://api.nuget.org/v3-flatcontainer/{pkg}/index.json"
        try:
            resp = requests.get(url, timeout=4)
            resp.raise_for_status()
            versions = resp.json().get("versions", [])
            if major:
                versions = [v for v in versions if v.startswith(f"{major}.")]
            versions = [v for v in versions if "-" not in v]      # no prereleases
            if not versions:
                return None
            best = max(versions, key=_semver.parse)
            return f"v{best}"
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Pulumi registry helper                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    @lru_cache(maxsize=128)
    def _latest_from_pulumi_registry(provider: str, major: str) -> str | None:
        """
        Return the latest *patch* release for given provider *major* stream,
        e.g. ("aws","6") → "v6.81.0".
        """
        url = f"https://api.pulumi.com/api/preview/registry/packages/pulumi/pulumi/{provider}/versions"
        try:
            resp = requests.get(url, timeout=4)
            resp.raise_for_status()
            versions = [
                v for v in resp.json().get("versions", []) if v.startswith(f"{major}.")
            ] 
            if not versions:
                return None
            best = max(versions, key=_semver.parse)
            return f"v{best}"
        except Exception:
            return None

    # -------------------------------------------------------------------- #
    # PyPI helper – latest Pulumi-Python SDK                               #
    # -------------------------------------------------------------------- #
    @staticmethod
    @lru_cache(maxsize=128)
    def _latest_from_pypi(provider: str) -> str | None:
        """
        Return newest released version of *pulumi-<provider>* from PyPI,
        as ``"vX.Y.Z"`` (skips pre-releases).
        """
        url = f"https://pypi.org/pypi/pulumi-{provider}/json"
        try:
            resp = requests.get(url, timeout=4)
            resp.raise_for_status()
            data     = resp.json()
            releases = [
                v for v in data.get("releases", {}).keys()
                if not _semver.parse(v).is_prerelease
            ]
            if not releases:
                return None
            best = max(releases, key=_semver.parse)
            return f"v{best}"
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_providers_json(raw: str) -> dict[str, str]:
        """
        Pick the *highest* version per provider from terraform’s –json output.
        Returns an empty dict on any parsing issue.
        """
        try:
            data = json.loads(raw)
        except Exception:
            return {}

        result: dict[str, str] = {}
        # Terraform ≥1.6
        for entry in data.get("provider", []):
            short = entry["name"].split("/")[-1]            # hashicorp/aws → aws
            result[short] = f"v{entry['version']}"

        # Terraform <1.6 (provider_instances)
        for inst in data.get("provider_instances", []):
            short = inst["name"].split("/")[-1]
            # keep the highest version we see
            ver   = f"v{inst['version']}"
            if short not in result or ver > result[short]:
                result[short] = ver

        return result

    def _grep_fallback(self, instance_id: str, dir_path: str) -> dict[str, str]:
        """
        Original heuristic scan limited to *dir_path*.  Versions may be empty.
        """
        print(f"[ProviderScanner] ↪ fallback grep for {os.path.relpath(dir_path, self.workspace_path)}")
        found: dict[str, str] = {}

        # provider blocks in *this dir only* (no recursion)
        cmd = f"grep -n -A5 -E 'provider\\s+\"' {dir_path}/*.tf 2>/dev/null || true"
        out = self._run(instance_id, cmd)
        for line in out.splitlines():
            m = _TF_PROVIDER_RE.search(line)
            if m:
                found[m["name"]] = ProviderScanner._norm(m["ver"])
                continue
            m2 = _TF_SIMPLE_PROVIDER_RE.search(line)
            if m2 and m2["name"] not in found:
                # → resolve to the latest published version
                latest = self._latest_from_registry(m2["name"])
                if latest:
                    found[m2["name"]] = latest
        # required_providers blocks (dir only)
        cmd = f"grep -n -A5 -E 'required_providers' {dir_path}/*.tf 2>/dev/null || true"
        out = self._run(instance_id, cmd)
        for m in _TF_REQUIRED_RE.finditer(out):
            found[m["name"]] = ProviderScanner._norm(m["ver"])

        # CDK-TF imports (old & new package names)
        cmd = (
            f"grep -R -n -E '@cdktf/provider-|cdktf_cdktf_provider_' {dir_path} "
            "--include='*.py' --include='*.ts' 2>/dev/null || true"
        )
        out = self._run(instance_id, cmd)
        for m in _TF_CDK_RE.finditer(out):
            name = m["name1"] or m["name2"]
            ver  = m["ver1"] or ""
            if ver:
                found[name] = ProviderScanner._norm(ver)
            else:
                latest = self._latest_from_registry(name)
                if latest:
                    found[name] = latest

        return found

    # ------------------------------------------------------------------ #
    # version normaliser                                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _norm(raw: str) -> str:
        """
        • If *raw* already looks like a range (>=, <, ~>, ,) → return as-is.  
        • Otherwise → ensure it starts with “v”.
        """
        raw = raw.strip()
        if any(sym in raw for sym in (">", "<", "~", ",")):
            return raw                    # keep range verbatim
        return f"v{raw.lstrip('v')}"

    # ------------------------------------------------------------------ #
    # newest-version helper                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _latest_from_registry(provider: str) -> str | None:
        """
        Query the Terraform Registry for *provider* (hashicorp namespace) and
        return the most recent semver as ``"vX.Y.Z"``  
        – returns ``None`` on any network / parsing error.
        """
        url = f"https://registry.terraform.io/v1/providers/hashicorp/{provider}/versions"
        try:
            resp = requests.get(url, timeout=4)
            resp.raise_for_status()
            versions = [v["version"] for v in resp.json().get("versions", [])]
            if not versions:
                return None
            latest = max(versions, key=_semver.parse)
            return f"v{latest}"
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # helpers                                                            #
    # ------------------------------------------------------------------ #
    def _run(self, instance_id: str, command: str) -> str:
        """
        Execute *command* in the pod and return **stdout** (stderr is ignored
        but printed on failure).
        """
        # ensure we always hit the project’s venv + collections
        env = (
            f'PATH="{self._venv_bin}:$PATH" '
            f'ANSIBLE_COLLECTIONS_PATHS="{self._collection_dir}" '
        )
        full_cmd = env + command
        # print(f"\n[RUN] on {instance_id}\nCMD: {full_cmd}\n")
        res = self.vm_manager.execute_command(instance_id, full_cmd)
        # print("[RUN] exit_code:", res.get("exit_code"))
        # if res.get("stdout"):
        #     print("[RUN] stdout:\n", res["stdout"])
        # if res.get("stderr"):
        #     print("[RUN] stderr:\n", res["stderr"])
        # if res.get("exit_code", 0) != 0:
        #     print(f"[ProviderScanner] command failed ({res['exit_code']}): {command}")
        #     print(res.get("stderr", ""))
        return res.get("stdout", "")

# ──────────────────────────────────────────────────────────────────────────
# Minimal ad-hoc test
# --------------------------------------------------------------------------
# ──────────────────────────────────────────────────────────────────────────
# Quick test: run “python -m infrastructure.pipeline.services.provider_scanner”
# --------------------------------------------------------------------------
# if __name__ == "__main__":
#     import os
#     import json
#     from termcolor import colored
#     from urllib.parse import urlparse

#     from infrastructure.services.global_manager import get_vm_manager

#     # --- configuration ---------------------------------------------------
#     PROJECT_ID   = "3ce79ab4-4c52-4182-8478-7a88828559a0"
#     REPO_URL     = "https://github.com/RPD123-byte/Soccer_Bot.git"
#     BRANCH_NAME  = "test"
#     GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # must be set for private repos

#     if not GITHUB_TOKEN:
#         raise SystemExit("❌  GITHUB_TOKEN env var required for this test")

#     # --- set up / get a pod ---------------------------------------------
#     vm_mgr = get_vm_manager()
#     vm = vm_mgr.assign_vm_to_project(PROJECT_ID, user_credentials={}, env_vars={})
#     print(colored(f"🖥️  Using pod: {vm.instance_id} ({vm.public_ip})", "yellow"))

#     # --- make sure code is present in /workspace ------------------------
#     owner, repo = urlparse(REPO_URL).path.lstrip("/").split("/")[:2]
#     repo_folder = f"/workspace/{owner}/{repo.replace('.git','')}"
#     auth_url = REPO_URL.replace("https://", f"https://{GITHUB_TOKEN}@")

#     clone_cmd = (
#         f"if [ -d {repo_folder}/.git ]; then "
#         # repo exists → fetch & reset to desired branch
#         f"  echo 'repo exists, updating'; "
#         f"  cd {repo_folder} && "
#         f"  git remote set-url origin {auth_url} && "
#         f"  git fetch origin && "
#         f"  git checkout {BRANCH_NAME} && "
#         f"  git reset --hard origin/{BRANCH_NAME}; "
#         f"else "
#         # fresh clone
#         f"  echo 'cloning repo'; "
#         f"  mkdir -p {repo_folder} && "
#         f"  git clone --depth 1 --branch {BRANCH_NAME} {auth_url} {repo_folder}; "
#         f"fi"
#     )
#     res = vm_mgr.execute_command(vm.instance_id, clone_cmd)
#     if res.get("exit_code"):
#         print(res.get("stderr", ""))
#         raise SystemExit("❌  Git setup failed")

#     print(colored("✔ Repo ready in pod", "green"))

#     # --- run the provider scanner ---------------------------------------
#     scanner = ProviderScanner(vm_mgr, workspace_path=repo_folder)
#     print(colored(f"[ProviderScanner] scanning {repo_folder}", "yellow"))
#     provider_map = scanner.scan_vm(instance_id=vm.instance_id)

#     print(colored("\n=== detected providers ===", "cyan"))
#     print(json.dumps(provider_map, indent=2))

#     # --- clean up --------------------------------------------------------
#     vm_mgr.release_project_vm(PROJECT_ID)
#     print(colored("\nVM released.", "yellow"))

# ────────────────────────────────────────────────────────────────────
# LOCAL-ONLY quick-test runner
#   • python provider_scanner.py  /path/to/repo
#   • python provider_scanner.py               # scans CWD
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__" and os.environ.get("PSCAN_LOCAL") != "0":
    import subprocess
    import sys
    import json
    from pathlib import Path
    from pprint import pprint

    class LocalProviderScanner(ProviderScanner):
        """
        Minimal shim that reuses all ProviderScanner logic but runs every
        shell command directly on the host via subprocess rather than
        inside a DevOps VM.
        """
        def __init__(self, workspace_path: str = "."):
            # vm_manager isn’t needed – pass None
            super().__init__(vm_manager=None, workspace_path=workspace_path)

        # override only this one helper ----------------------------------
        def _run(self, _instance_id: str, command: str) -> str:
            """
            Execute *command* on the local shell and return stdout.
            Stderr is swallowed (just like the remote variant).
            """
            proc = subprocess.run(
                command,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            return proc.stdout

    # ------------ entry-point ------------------------------------------
    repo_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    print(f"🔍 scanning local repo: {repo_root}")

    scanner   = LocalProviderScanner(workspace_path=str(repo_root))
    providers = scanner.scan_vm(instance_id="local")     # id is unused

    print("\n=== detected providers ===")
    print(json.dumps(providers, indent=2))
