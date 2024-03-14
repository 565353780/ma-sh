from time import time


class Timer(object):
    def __init__(self, auto_start: bool = True) -> None:
        self.start_time = None
        self.time_sum = 0

        if auto_start:
            self.start()
        return

    def reset(self, auto_start: bool = True) -> bool:
        self.start_time = None
        self.time_sum = 0

        if auto_start:
            self.start()
        return True

    def start(self) -> bool:
        if self.start_time is not None:
            return True

        self.start_time = time()
        return True

    def pause(self) -> bool:
        if self.start_time is None:
            return True

        self.time_sum += time() - self.start_time
        self.start_time = None
        return True

    def currentTimeSum(self) -> float:
        if self.start_time is None:
            return self.time_sum

        return time() - self.start_time + self.time_sum

    def now(self, append: str = "s") -> float:
        current_time_sum = self.currentTimeSum()

        match append:
            case "s":
                return current_time_sum
            case "m":
                return current_time_sum / 60.0
            case "h":
                return current_time_sum / 60.0 / 60.0
            case "d":
                return current_time_sum / 60.0 / 60.0 / 24.0
            case _:
                return current_time_sum
