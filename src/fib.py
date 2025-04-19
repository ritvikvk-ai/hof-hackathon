def fibonacci(n):
    """Calculate the nth Fibonacci number iteratively.
    
    Args:
        n (int): The position in the Fibonacci sequence (0-based)
        
    Returns:
        int: The nth Fibonacci number
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Fibonacci sequence not defined for negative numbers")
    if n <= 1:
        return n
        
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
