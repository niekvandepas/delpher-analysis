import time

def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    print(f"Function '{func.__name__}' took {time.time() - start_time:.2f} seconds")
    return result
