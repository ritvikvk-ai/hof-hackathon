from node_dll import DoublyLinkedList

class ExtendedDoublyLinkedList(DoublyLinkedList):
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
dll = ExtendedDoublyLinkedList()
for value in [1, 2, 3, 4, 5]:
    dll.append(value)

print("Original List:")
dll.print_list()

dll.reverse()

print("Reversed List:")
dll.print_list()
