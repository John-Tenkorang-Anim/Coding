# Definition for singly-linked list.
from typing import Optional, List
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

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

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

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False

        return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)


    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        
        def isSameTree(p, q):
            if not p and not q:
                return True
            if not p or not q:
                return False
            if p.val != q.val:
                return False
            return isSameTree(p.right,q.right) and isSameTree(p.left,q.left)
        
        def dfs(root, subRoot):
            if not root:
                return False

            if isSameTree(root,subRoot):
                return  True
            return dfs(root.left,subRoot) or dfs(root.right, subRoot)

        return dfs(root, subRoot)
    
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if p.val > q.val:
            p, q = q, p

        while root:
            if root.val > q.val:
                root = root.left 
            elif root.val < p.val:
                root = root.right  
            else:
                return root
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        
        if not root:
            return None

        if key < root.val:
            root.left = self.deleteNode(root.left, key)
        elif key > root.val:
            root.right = self.deleteNode(root.right, key)
        else:
            if not root.left:
                return root.right
            elif not root.right:
                return root.left
            temp = find_min(root.right)
            root.val = temp.val
            root.right = self.deleteNode(root.right, temp.val)
        return root

def find_min(node):
    while node.left:
        node = node.left
    return node

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        queue = deque([root])

        res = []

        while queue:
            lev = len(queue)
            for i in range(lev):
                node = queue.popleft()
                if i == lev -1:
                    res.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return res