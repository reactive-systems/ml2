"""Base class for Containerized gRPC services"""

import logging
from time import sleep
import socket

import grpc

import docker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SLEEP_INTERVAL = 1


class ContainerizedGRPCService:
    def __init__(
        self,
        image: str,
        cpu_count: int = 2,
        mem_limit: str = "2g",
        port: int = None,
        start_up_timeout: int = 60,
        channel_ready_timeout: int = 10,
        service_name: str = "",
    ):

        if not port:
            # find a port that is not used
            self.port = None
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            for p in range(50051, 50151):
                try:
                    s.connect(("localhost", p))
                except Exception:
                    self.port = p
                    break
            s.close()
            if not self.port:
                raise Exception("Could not find unused port")
        else:
            self.port = port

        self.service_name = service_name
        docker_client = docker.from_env()
        if image not in [(i.attrs["RepoTags"] + [None])[0] for i in docker_client.images.list()]:
            logger.info("Pulling Docker image %s", image)
            docker_client.images.pull(image)

        container_same_port = next(
            (c for c in docker_client.containers.list() if f"{self.port}/tcp" in c.ports), None
        )
        if container_same_port:
            if container_same_port.image.attrs["RepoTags"][0] == image:
                self.container = container_same_port
                logger.info(
                    "Found existing %s container %s on port %d",
                    service_name,
                    self.container.name,
                    self.port,
                )
            else:
                raise Exception(f"Found container running different image on port {self.port}")
        else:
            self.container = docker_client.containers.run(
                image,
                f"-p {self.port}",
                detach=True,
                mem_limit=mem_limit,
                ports={f"{self.port}/tcp": self.port},
                remove=True,
            )

        start_up_time = 0
        while self.container.status != "running" and start_up_time < start_up_timeout:
            sleep(SLEEP_INTERVAL)
            start_up_time += SLEEP_INTERVAL
            self.container.reload()
        if self.container.status != "running":
            logger.error(
                "Could not run %s container within %d seconds", service_name, start_up_timeout
            )
        else:
            logger.info(
                "%s container %s on port %d is running",
                service_name,
                self.container.name,
                self.port,
            )
            self.channel = grpc.insecure_channel(f"localhost:{self.port}")
            try:
                grpc.channel_ready_future(self.channel).result(timeout=channel_ready_timeout)
            except grpc.FutureTimeoutError:
                logger.error(
                    "Could not connect to %s gRPC server within %d seconds",
                    service_name,
                    channel_ready_timeout,
                )
            else:
                logger.info(
                    "Successfully connected to %s gRPC server running in container %s on port %d",
                    service_name,
                    self.container.name,
                    self.port,
                )

    def __del__(self):
        if hasattr(self, "container"):
            self.container.stop()
            logger.info(
                "Stopped and removed container %s running %s on port %d",
                self.container.name,
                self.service_name,
                self.port,
            )

    def __delete__(self, instance):
        if hasattr(self, "container"):
            self.container.stop()
            logger.info(
                "Stopped and removed container %s running %s on port %d",
                self.container.name,
                self.service_name,
                self.port,
            )
