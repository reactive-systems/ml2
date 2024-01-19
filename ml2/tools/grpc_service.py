"""Base class for gRPC services"""

import contextlib
import errno
import logging
import multiprocessing
import socket
from time import sleep
from typing import Any, Callable, Dict, List, Optional

import grpc
from docker.models.containers import Container

import docker

from ..configurable import Configurable
from ..grpc.tools.tools_pb2 import Functionality, IdentificationRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SLEEP_INTERVAL = 1


class GRPCService(Configurable):
    def __init__(
        self,
        stub: Callable,
        tool: str,
        network_mode: str = "",
        cpu_count: int = 2,
        mem_limit: str = "2g",
        port: int = None,
        start_containerized_service: bool = True,
        image: str = None,
        start_service: bool = False,
        service: Callable = None,
        start_up_timeout: int = 60,
        channel_ready_timeout: int = 15,
        nvidia_gpus: bool = False,
    ):
        self.tool = tool

        if port is not None and start_containerized_service:
            if image is None:
                raise Exception("An image needs to be given to start a containerized service")
            if self.port_is_used(port):
                raise Exception("port is already in use, cannot start container at " + str(port))
            self.start_container_port(
                port=port,
                image=image,
                start_up_timeout=start_up_timeout,
                mem_limit=mem_limit,
                cpu_count=cpu_count,
                nvidia_gpus=nvidia_gpus,
                network_mode=network_mode,
            )
            self.port = port

        elif port is not None and start_service:
            if service is None:
                raise Exception("A service function needs to be given to start a service")
            if self.port_is_used(port):
                raise Exception("port is already in use, cannot start service at " + str(port))
            self.start_grpc_subprocess_port(port=port, service=service)
            self.port = port

        elif port is not None:
            self.port = port

        elif port is None and start_containerized_service:
            if image is None:
                raise Exception("An image needs to be given to start a containerized service")
            self.port = self.start_container(
                image=image,
                start_up_timeout=start_up_timeout,
                mem_limit=mem_limit,
                cpu_count=cpu_count,
                nvidia_gpus=nvidia_gpus,
                network_mode=network_mode,
            )

        elif port is None and start_service:
            if service is None:
                raise Exception("A service function needs to be given to start a service")
            self.port = self.start_grpc_subprocess(service=service)

        elif port is None and not start_containerized_service and not start_service:
            raise Exception(
                "If no port is given, a start service parameter (start_service or start_containerized_service) needs to be set"
            )

        elif start_containerized_service and start_service:
            raise Exception("A service can either be containerized or not.")

        else:
            raise Exception("Unknown Error")

        self.connect_to_grpc(self.port, channel_ready_timeout=channel_ready_timeout)
        self.stub = stub(self.channel)
        self.check_identity()

    def config_postprocessors(self, **kwargs) -> list:
        def postprocess_callable(config: Dict[str, Any], annotations: Dict[str, type]) -> None:
            config.pop("service", None)
            annotations.pop("service", None)
            config.pop("stub", None)
            annotations.pop("stub", None)

        return [postprocess_callable] + super().config_postprocessors()

    def start_container(self, **kwargs) -> int:
        port = self.get_free_port()
        try:
            self.start_container_port(port, **kwargs)
        except docker.errors.APIError as e:
            if "Ports are not available" in str(e) or "port is already allocated" in str(e):
                print("test")
                self.start_container(**kwargs)
            else:
                raise e
        return port

    def start_grpc_subprocess(self, **kwargs) -> int:
        logger.warning(
            "Starting a subprocess without specifying ports can lead to a race condition on reserving ports. Starting the subprocess may fail."
        )
        port = self.get_free_port()
        self.start_grpc_subprocess_port(port, **kwargs)
        return port

    def connect_to_grpc(self, port: int, channel_ready_timeout: int):
        self.channel = grpc.insecure_channel(f"localhost:{port}")

        try:
            grpc.channel_ready_future(self.channel).result(timeout=channel_ready_timeout)
        except grpc.FutureTimeoutError:
            raise Exception(
                "Could not connect to %s gRPC server within %d seconds",
                self.tool,
                channel_ready_timeout,
            )
        else:
            logger.info(
                "Successfully connected to %s gRPC server on port %d",
                self.tool,
                port,
            )

    def check_identity(self):
        if self.assert_identities is not None and not self.assert_identities:
            raise Exception(
                "Tool setup for "
                + self.tool
                + " failed. \n GRPC Server Tool does not match client side"
            )

    def start_container_port(self, port: int, image: str, start_up_timeout: int, **kwargs):
        docker_client = docker.from_env()
        if image not in [(i.attrs["RepoTags"] + [None])[0] for i in docker_client.images.list()]:
            logger.info("Pulling Docker image %s", image)
            docker_client.images.pull(image)

        self.run_container(cli=docker_client, image=image, own_port=port, **kwargs)

        start_up_time = 0
        while self.container.status != "running" and start_up_time < start_up_timeout:
            sleep(SLEEP_INTERVAL)
            start_up_time += SLEEP_INTERVAL
            self.container.reload()
        if self.container.status != "running":
            raise Exception(
                "Could not run %s container within %d seconds", self.tool, start_up_timeout
            )
        else:
            logger.info(
                "%s container %s on port %d started",
                self.tool,
                self.container.name,
                port,
            )

    def start_grpc_subprocess_port(self, port: int, service: Callable):
        prc = multiprocessing.Process(target=service, args=(port,))
        prc.daemon = True
        prc.start()
        self.subprocess = prc

    def get_free_port(self) -> int:
        for p in range(50051, 50251):
            if not self.port_is_used(p):
                return p
        raise Exception("could not find free port in given range")

    @staticmethod
    def port_is_used(port: int, host: str = "127.0.0.1") -> bool:
        """
        Returns if port is used. Port is considered used if the current process
        can't bind to it or the port doesn't refuse connections.
        """

        def _refuses_connection(port: int, host: str) -> bool:
            sock = socket.socket()
            with contextlib.closing(sock):
                sock.settimeout(1)
                err = sock.connect_ex((host, port))
                return err == errno.ECONNREFUSED

        def _can_bind(port: int, host: str) -> bool:
            sock = socket.socket()
            with contextlib.closing(sock):
                try:
                    sock.bind((host, port))
                except socket.error:
                    return False
            return True

        unused = _can_bind(port, host) and _refuses_connection(port, host)
        return not unused

    def run_container(
        self,
        cli: docker.DockerClient,
        image: str,
        own_port: int,
        mem_limit: str,
        cpu_count: Optional[int] = None,
        nvidia_gpus: bool = False,
        network_mode: str = "",
    ):
        if cpu_count is not None:
            logger.warning("cpu count in docker not implemented")
        forward_ports = {own_port: own_port}
        ports_dict = {f"{k}/tcp": forward_ports[k] for k in forward_ports.keys()}
        if nvidia_gpus and network_mode == "":
            self.container: Container = cli.containers.run(
                image,
                f"-p {own_port}",
                detach=True,
                mem_limit=mem_limit,
                ports=ports_dict,
                remove=True,
                device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],  # type: ignore
            )
        elif nvidia_gpus and network_mode != "":
            self.container: Container = cli.containers.run(
                image,
                f"-p {own_port}",
                detach=True,
                mem_limit=mem_limit,
                remove=True,
                network_mode=network_mode,
                device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])],  # type: ignore
            )
        elif not nvidia_gpus and network_mode == "":
            self.container: Container = cli.containers.run(
                image,
                f"-p {own_port}",
                detach=True,
                mem_limit=mem_limit,
                ports=ports_dict,
                remove=True,
            )  # type: ignore
        else:
            self.container: Container = cli.containers.run(
                image,
                f"-p {own_port}",
                detach=True,
                mem_limit=mem_limit,
                remove=True,
                network_mode=network_mode,
            )  # type: ignore

    @property
    def server_version(self) -> Optional[str]:
        if hasattr(self.stub, "Identify") and callable(self.stub.Identify):
            return self.stub.Identify(IdentificationRequest()).version
        else:
            logger.warning("GRPC service has not implemented the identify message")
            return None

    @property
    def functionality(self) -> Optional[List[str]]:
        if hasattr(self.stub, "Identify") and callable(self.stub.Identify):
            return [
                Functionality.Name(i)
                for i in self.stub.Identify(IdentificationRequest()).functionalities
            ]
        else:
            logger.warning("GRPC service has not implemented the identify message")
            return None

    @property
    def assert_identities(self) -> Optional[bool]:
        if self.tool is None:
            logger.info("Generic Service has no identity check.")
            return None
        elif hasattr(self.stub, "Identify") and callable(self.stub.Identify):
            try:
                result = self.stub.Identify(IdentificationRequest())
                return result.tool == self.tool
            except Exception:
                return False
        else:
            logger.warning("GRPC service has not implemented the identify message")
            return None

    def __del__(self):
        self.remove_container()
        self.remove_subprocess()

    def __delete__(self, instance):
        self.remove_container()
        self.remove_subprocess()

    def remove_container(self):
        if hasattr(self, "container"):
            self.container.stop()
            logger.info(
                "Stopped and removed container %s running %s on port %d",
                self.container.name,
                self.tool,
                self.port,
            )
            delattr(self, "container")

    def remove_subprocess(self):
        if hasattr(self, "subprocess"):
            self.subprocess.terminate()
            logger.info(
                "Stopped and subprocess running %s on port %d",
                self.tool,
                self.port,
            )
            delattr(self, "subprocess")
