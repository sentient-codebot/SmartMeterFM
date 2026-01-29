import random
from datetime import datetime


__all__ = ["configuration", "plot"]


def generate_time_id():
    return datetime.now().strftime("%Y%m%d" + "-" + f"{random.randint(0, 9999):04d}")
