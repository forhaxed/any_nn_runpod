"""
Talking to RunPod.

Two APIs, because RunPod needs two:

* **REST** (``https://rest.runpod.io/v1``) for the pod lifecycle.  Everything
  the launcher does day to day.
* **GraphQL** (``https://api.runpod.io/graphql``) for the catalogue of GPU
  types and their prices, which REST does not expose.

REST v1 is deprecated with a retirement date of **15 November 2026**; v2 exists
at ``https://api.runpod.io/v2`` with nested request objects and a single
``POST /v2/pods/{id}/action`` in place of the separate start/stop verbs.  The
version is confined to this file so the move is one file's worth of work.

The launcher holds the API key.  It is never put in a pod's environment: a pod
is rented from strangers and runs code from the internet, and a key that can
create and delete pods is not something to leave lying around on one.  The
consequence is that pods are ended from here, which is where §7 of the plan
puts the three triggers.
"""

from __future__ import annotations

import os
import time

import requests

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"

#: How long a freshly created pod may take to report an IP and port mappings.
READY_TIMEOUT = 600.0


class RunPodError(RuntimeError):
    """RunPod said no.  The message is theirs."""


class RunPod:
    def __init__(self, api_key: str | None = None, root: str | None = None, timeout: float = 60.0):
        self.api_key = api_key or _key_from_environment(root)
        if not self.api_key:
            raise RunPodError(
                "No RunPod API key. Put it in .env at the project root as\n"
                "    RUNPOD_API_KEY=...\n"
                "or export it. Make one at "
                "https://console.runpod.io/user/settings -> API Keys."
            )
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    # ================================================================
    #  Pods
    # ================================================================
    def create_pod(
        self,
        name: str,
        image: str,
        gpu_types: list,
        *,
        ports: list,
        start_command: list | None = None,
        env: dict | None = None,
        gpu_count: int = 1,
        container_disk_gb: int = 40,
        volume_gb: int = 0,
        volume_mount_path: str = "/workspace",
        network_volume_id: str | None = None,
        cloud_type: str = "SECURE",
        data_centers: list | None = None,
        interruptible: bool = False,
    ) -> dict:
        """Create a pod and start it.  Returns the pod as RunPod describes it."""
        body = {
            "name": name,
            "imageName": image,
            "computeType": "GPU",
            "cloudType": cloud_type,
            "gpuCount": gpu_count,
            "gpuTypeIds": list(gpu_types),
            "gpuTypePriority": "availability",
            "containerDiskInGb": container_disk_gb,
            "ports": list(ports),
            "supportPublicIp": True,
            "env": dict(env or {}),
            "interruptible": interruptible,
        }
        if start_command:
            body["dockerStartCmd"] = list(start_command)
        if network_volume_id:
            body["networkVolumeId"] = network_volume_id
        elif volume_gb:
            body["volumeInGb"] = volume_gb
            body["volumeMountPath"] = volume_mount_path
        if data_centers:
            body["dataCenterIds"] = list(data_centers)
        return self._request("POST", "/pods", json=body)

    def list_pods(self) -> list:
        answer = self._request("GET", "/pods")
        return answer if isinstance(answer, list) else answer.get("pods", [])

    def get_pod(self, pod_id: str) -> dict:
        return self._request("GET", f"/pods/{pod_id}")

    def find_pod(self, name: str) -> dict | None:
        for pod in self.list_pods():
            if pod.get("name") == name:
                return pod
        return None

    def stop_pod(self, pod_id: str) -> dict:
        """Stop the container but keep the pod and its disk.  Still billed for storage."""
        return self._request("POST", f"/pods/{pod_id}/stop")

    def start_pod(self, pod_id: str) -> dict:
        return self._request("POST", f"/pods/{pod_id}/start")

    def terminate_pod(self, pod_id: str) -> None:
        """Destroy the pod.  Irreversible: the container disk goes with it."""
        self._request("DELETE", f"/pods/{pod_id}")

    # ================================================================
    #  Waiting
    # ================================================================
    def wait_until_ready(
        self, pod_id: str, ports: list, timeout: float = READY_TIMEOUT, on_wait=None
    ) -> dict:
        """Poll until the pod has a public IP and every port is mapped.

        Neither exists at creation time -- RunPod assigns the external port for
        each container port, and there is no way to predict the pair.  This is
        what replaces reading them off the console by eye.
        """
        deadline = time.monotonic() + timeout
        wanted = {str(port) for port in ports}
        while True:
            pod = self.get_pod(pod_id)
            mappings = pod.get("portMappings") or {}
            if pod.get("publicIp") and wanted <= set(map(str, mappings)):
                return pod
            if pod.get("desiredStatus") == "TERMINATED":
                raise RunPodError(
                    f"pod {pod_id} was terminated while starting up: "
                    f"{pod.get('lastStatusChange')}"
                )
            if time.monotonic() >= deadline:
                raise RunPodError(
                    f"pod {pod_id} had no public IP and mappings for "
                    f"{', '.join(sorted(wanted))} within {timeout:.0f}s. "
                    f"Last seen: status={pod.get('desiredStatus')}, "
                    f"ip={pod.get('publicIp')!r}, mappings={mappings}. "
                    "A pod with no public IP cannot be reached -- try another "
                    "data centre, or SECURE cloud."
                )
            if on_wait is not None:
                on_wait(pod)
            time.sleep(3.0)

    # ================================================================
    #  GPU catalogue (GraphQL only)
    # ================================================================
    def gpu_types(self) -> list:
        """Available GPU types with prices, cheapest usable first."""
        query = """
        query GpuTypes {
          gpuTypes {
            id
            displayName
            memoryInGb
            secureCloud
            communityCloud
            lowestPrice(input: {gpuCount: 1}) {
              minimumBidPrice
              uninterruptablePrice
            }
          }
        }
        """
        response = self._session.post(
            GRAPHQL,
            json={"query": query},
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        payload = _json(response)
        if "errors" in payload:
            raise RunPodError(f"GraphQL: {payload['errors']}")

        types = []
        for entry in payload["data"]["gpuTypes"]:
            price = (entry.get("lowestPrice") or {}).get("uninterruptablePrice")
            types.append(
                {
                    "id": entry["id"],
                    "name": entry.get("displayName") or entry["id"],
                    "memory_gb": entry.get("memoryInGb"),
                    "price": price,
                    "secure": bool(entry.get("secureCloud")),
                    "community": bool(entry.get("communityCloud")),
                }
            )
        available = [t for t in types if t["price"]]
        return sorted(available, key=lambda t: t["price"])

    # ================================================================
    #  Plumbing
    # ================================================================
    def _request(self, method: str, path: str, **kwargs):
        response = self._session.request(
            method, REST + path, timeout=self.timeout, **kwargs
        )
        if response.status_code == 401:
            raise RunPodError("RunPod rejected the API key (401).")
        if response.status_code == 404:
            raise RunPodError(f"RunPod has no {path} (404).")
        if not response.ok:
            raise RunPodError(
                f"{method} {path} failed ({response.status_code}): "
                f"{_error_text(response)}"
            )
        if not response.content:
            return {}
        return _json(response)


def endpoint(pod: dict, port: int) -> tuple[str, int]:
    """The (host, port) to dial for a container port on this pod."""
    mappings = pod.get("portMappings") or {}
    mapped = mappings.get(str(port))
    if not pod.get("publicIp") or not mapped:
        raise RunPodError(
            f"pod {pod.get('id')} has no external mapping for port {port} yet"
        )
    return pod["publicIp"], int(mapped)


def _key_from_environment(root: str | None = None) -> str | None:
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key
    # A .env in the project directory is the ordinary place for it, and it is in
    # .gitignore for the obvious reason. The project root comes first: run.py
    # can be invoked from anywhere with --root.
    candidates = [root] if root else []
    candidates += [os.getcwd(), os.path.dirname(os.getcwd())]
    for directory in candidates:
        path = os.path.join(directory, ".env")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("RUNPOD_API_KEY"):
                    _, _, value = line.partition("=")
                    return value.strip().strip("\"'") or None
    return None


def _json(response):
    try:
        return response.json()
    except ValueError as exc:
        raise RunPodError(
            f"RunPod returned something that is not JSON: {response.text[:200]}"
        ) from exc


def _error_text(response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:400]
    # v1 uses {"error": ...}; v2 follows RFC 9457 with title/detail.
    for key in ("detail", "error", "message", "title"):
        if payload.get(key):
            return str(payload[key])
    return str(payload)[:400]
