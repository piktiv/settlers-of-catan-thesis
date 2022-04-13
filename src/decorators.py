import time


# @timer decorator to time function
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        print(f'{func.__name__} {time.time() - start}')
    return wrapper
