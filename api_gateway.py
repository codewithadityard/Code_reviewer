import asyncio
import time
from langsmith import traceable

class APIGateway:
    def __init__(self, rpm_limit=15, max_concurrency=2):
        self.rpm_limit = rpm_limit
        self.max_concurrency = max_concurrency
        self.request_times = []
  
        # We just set them to None for now.
        self.concurrency_semaphore = None
        self.times_lock = None
        self.queue = None
        self.queue_counter = 0
        self._processor_task = None

    async def _initialize(self):
        """Creates asyncio primitives safely inside the running event loop."""
        # This will only run once, the very first time an agent asks for a ticket.
        if self.queue is None:
            self.concurrency_semaphore = asyncio.Semaphore(self.max_concurrency)
            self.times_lock = asyncio.Lock()
            self.queue = asyncio.PriorityQueue()
            # NOW it is safe to create the task, because the loop is running!
            self._processor_task = asyncio.create_task(self._process_queue())



    async def _process_queue(self):
        """Background loop that grants permission to waiting requests when capacity frees up."""
        while True:
            priority, _, future = await self.queue.get()
            
            async with self.times_lock:
                now = time.time()
                self.request_times = [t for t in self.request_times if now - t < 60]
                
                if len(self.request_times) >= self.rpm_limit:
                    sleep_time = 60 - (now - self.request_times[0]) + 0.1
                    if sleep_time > 0:
                        print(f" [GATEWAY] Capacity full. Holding queue for {sleep_time:.2f}s")
                        await asyncio.sleep(sleep_time)
                        
                        now = time.time()
                        self.request_times = [t for t in self.request_times if now - t < 60]

                self.request_times.append(now)

            future.set_result(True)
            self.queue.task_done()
    @traceable(name="Groq API Network Call", run_type="llm")
    async def wait_for_admission(self, priority: int = 10):
        """Agents call this to get in line. Blocks until permission is granted."""
        #  IMPORTANT: Boot up the async objects safely before waiting in line
        await self._initialize()
        
        future = asyncio.Future()
        async with self.times_lock:
            self.queue_counter += 1
            counter = self.queue_counter
            
        await self.queue.put((priority, counter, future))
        await future 

# Instantiate a single global gateway to be shared across all agents
global_gateway = APIGateway(rpm_limit=30, max_concurrency=4)