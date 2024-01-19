"""Ray actor class for tracking progress of data generation"""

from asyncio import Event

import ray


@ray.remote
class ProgressActor:
    def __init__(self):
        self.progress = {}
        self.event = Event()

    def get(self, key):
        return self.progress[key]

    def get_progress(self):
        return self.progress

    def update(self, key, delta=1):
        if key in self.progress:
            self.progress[key] += delta
            self.event.set()
        else:
            self.progress[key] = delta
            self.progress = {key: self.progress[key] for key in sorted(self.progress.keys())}
            self.event.set()

    def update_multi(self, keys, delta=1):
        for key in keys:
            self.update(key, delta)

    async def wait_for_update(self):
        await self.event.wait()
        self.event.clear()
        return self.progress
