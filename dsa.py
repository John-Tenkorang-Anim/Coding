# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        current = head

        while current and current.next:
            first = current
            second = current.next
            next_pair = second.next

            #Swapping
            prev.next = second
            second.next = first
            first.next = next_pair

            #moving pointers
            prev = first
            current = next_pair

        return dummy.next