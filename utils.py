import time

def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    print(f"Function '{func.__name__}' took {time.time() - start_time:.2f} seconds")
    return result


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncates text to a maximum number of characters."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
