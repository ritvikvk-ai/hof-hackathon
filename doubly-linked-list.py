class Node:
    def __init__(self, data):
        self.data = data
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None

    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = new_node
        new_node.prev = curr

    def reverse(self):
        temp = None
        current = self.head

        while current:
            # Swap next and prev
            temp = current.prev
            current.prev = current.next
            current.next = temp

            # Move to next node (which is previous node before swap)
            current = current.prev
        
        if temp:
            self.head = temp.prev  # Update head

    def print_list(self):
        curr = self.head
        while curr:
            print(curr.data, end=" <-> " if curr.next else "\n")
            curr = curr.next

# Example usage:
dll = DoublyLinkedList()
for value in [1, 2, 3, 4, 5]:
    dll.append(value)

print("Original List:")
dll.print_list()

dll.reverse()

print("Reversed List:")
dll.print_list()
