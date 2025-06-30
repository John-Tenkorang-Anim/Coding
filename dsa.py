# Definition for singly-linked list.
from typing import Optional
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
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

from collections import defaultdict
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        
        rows = defaultdict(set)
        cols = defaultdict(set)
        squares = defaultdict(set)


        for r in range(9):
            for c in range(9):

                if board[r][c] == ".":
                    continue
                
                elif (board[r][c] in rows[r] or
                    board[r][c] in cols[c] or 
                    board[r][c] in squares[(r//3,c//3)]):
                    return False
                else:
                    rows[r].add(board[r][c])
                    cols[c].add(board[r][c])
                    squares[(r//3,c//3)].add(board[r][c])

        return True

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        spiral_matrix = []

        top = left = 0
        bottom = len(matrix) - 1
        right = len(matrix[0]) - 1

        while top <= bottom and left <= right:

            for i in range(left, right+1):
                spiral_matrix += [matrix[top][i]]
            top += 1

            for j in range(top, bottom + 1):
                spiral_matrix += [matrix[j][right]]
            right -= 1

            if top <= bottom:
                for k in range(right,left-1,-1):
                    spiral_matrix += [matrix[bottom][k]]
                bottom -= 1
            if left <= right:
                for m in range(bottom,top-1,-1):
                    spiral_matrix += [matrix[m][left]]
                left += 1

        return spiral_matrix

Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        pre_idx = 0
        post_hmap = {val: i for i, val in enumerate(postorder)}

        def helper(left , right):
            if left > right:
                return None
            nonlocal pre_idx
            root_val = preorder[pre_idx]
            root = TreeNode(root_val)
            pre_idx += 1

            if left == right:
                return root 

            left_child_val = preorder[pre_idx]
            idx_in_post = post_hmap[left_child_val]

            root.left = helper(left, idx_in_post)
            root.right = helper(idx_in_post +1, right-1)
            return root

        return helper(0, len(postorder)-1)

class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False

        return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)
