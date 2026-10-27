import docker
import tempfile
import os
from typing import Dict, Any, Optional

_docker_client: Optional[docker.DockerClient] = None

def get_docker_client() -> docker.DockerClient:
    """Lazily initialize Docker client to avoid startup crashes when Docker is stopped."""
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client

def execute_in_sandbox(repo_files: Dict[str, str], test_command: list[str]) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as host_temp_dir:
        # Ensure non-root container user can write to workspace directory
        os.chmod(host_temp_dir, 0o777)

        # Write files to host mount
        for file_path, content in repo_files.items():
            full_path = os.path.join(host_temp_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            os.chmod(full_path, 0o666)

        try:
            client = get_docker_client()
            container = client.containers.run(
                image="python:3.11-slim",
                command=test_command,
                volumes={host_temp_dir: {"bind": "/workspace", "mode": "rw"}},
                working_dir="/workspace",
                network_disabled=True,       # Prevent SSRF / exfiltration
                mem_limit="512m",             # Kill memory bombs
                nano_cpus=1_000_000_000,      # Max 1 core
                user="1000:1000",             # Non-root
                detach=False,
                stdout=True,
                stderr=True,
                remove=True                   # Ephemeral cleanup
            )
            output = container.decode("utf-8", errors="replace") if isinstance(container, bytes) else str(container)
            return {"exit_code": 0, "output": output}
        except docker.errors.ContainerError as exc:
            err_output = ""
            if getattr(exc, "stderr", None):
                err_output = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else str(exc.stderr)
            elif getattr(exc, "container", None):
                err_output = exc.container.logs().decode("utf-8", errors="replace")
            else:
                err_output = str(exc)
            return {"exit_code": exc.exit_status, "output": err_output}
        except docker.errors.DockerException as exc:
            return {"exit_code": -1, "output": f"Docker daemon unavailable: {exc}"}
        except Exception as exc:
            return {"exit_code": -1, "output": str(exc)}

        