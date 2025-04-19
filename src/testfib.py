def fibonacci(n):
    """Calculate the nth Fibonacci number iteratively.
    
    Args:
        n (int): The position in the Fibonacci sequence (0-based)
        
    Returns:
        int: The nth Fibonacci number
    """
    if n < 0:
        raise ValueError("Fibonacci sequence is not defined for negative numbers")
    if n <= 1:
        return n
        
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Test cases
def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(2) == 1
    assert fibonacci(3) == 2
    assert fibonacci(4) == 3
    assert fibonacci(5) == 5
    assert fibonacci(6) == 8
    assert fibonacci(7) == 13
    
    try:
        fibonacci(-1)
        assert False, "Should raise ValueError for negative numbers"
    except ValueError:
        pass
        
if __name__ == "__main__":
    test_fibonacci()
    print("All tests passed!")
