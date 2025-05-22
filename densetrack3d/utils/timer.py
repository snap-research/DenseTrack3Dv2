import torch

class CUDATimer:
    def __init__(self, name="Block", enabled=True):
        self.name = name
        self.enabled = enabled
        if not enabled:
            return
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if not self.enabled:
            return
        torch.cuda.synchronize()  # Make sure everything before is done
        self.start_event.record()

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return
        self.end_event.record()
        torch.cuda.synchronize()  # Wait for everything to finish
        elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        print(f"{self.name}: {elapsed_time_ms:.3f} ms")