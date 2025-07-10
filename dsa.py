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
    
    def findCenter(self, edges: List[List[int]]) -> int:
        a, b = edges[0]
        c, d = edges[1]
        
        if a == c or a == d:
            return a
        else:
            return b

import heapq
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        
        maxHeap = [-s for s in stones]

        heapq.heapify(maxHeap)

        while len(maxHeap) > 1:

            first = -heapq.heappop(maxHeap)
            second = -heapq.heappop(maxHeap)

            if first != second:
                heapq.heappush(maxHeap, -(first-second))
        
        return -maxHeap[0] if maxHeap else 0

import heapq
from collections import defaultdict
from typing import List, Optional
def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        
    hmap = defaultdict(list)
    heap = []
    res = []

    for point in points:
        d = point[0]**2 + point[1]**2
        hmap[d].append(point)
        heap.append(d)
        
    heapq.heapify(heap)
    visited = set()

    while k > 0:
        dist = heapq.heappop(heap)
        if dist in visited:
            continue
        visited.add(dist)
        for point in hmap[dist]:
            if k == 0:
                break
            res.append(point)
            k -= 1
        
    return res
def goodNodes(self, root: TreeNode) -> int:

    def good_count(root, max_val):
        if not root:
            return 0

        good = 1 if max_val <= root.val else 0

        new_max = max(max_val,root.val)
        left = good_count(root.left,new_max)
        right = good_count(root.right,new_max)
        return good + left + right
            
    return good_count(root, float('-inf'))  
def isValidBST(self, root: Optional[TreeNode]) -> bool:

    def valid(node,min_val,max_val):
        if not node:
            return True

        if not (min_val < node.val < max_val):
            return False

        return valid(node.left,min_val,node.val) and valid(node.right,node.val,max_val)

    return valid(root,float('-inf'),float('inf'))

def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
    res = []
    def inorder(node):
        if not node:
            return
            
        inorder(node.left)
        res.append(node.val)
        inorder(node.right)
        
    inorder(root)
    return res[k-1]
def maxPathSum(self, root: Optional[TreeNode]) -> int:
    path_sum = root.val

    def dfs(node):
        nonlocal path_sum
        if not node:
            return 0
            
        left_sum = max(0, dfs(node.left))
        right_sum = max(0, dfs(node.right))

        path_sum = max(path_sum, left_sum + right_sum + node.val)

        return max(left_sum, right_sum) + node.val
    dfs(root)
    return path_sum
class Codec:
    def __init__(self):
        self.vals = []
def serialize(root):
    """Encodes a tree to a single string.
        
    :type root: TreeNode
    :rtype: str
    """
    def dfs(node):
        if node:
            vals.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
        else:
            vals.append("#")
    vals = []
    dfs(root)  
    return " ".join(vals)
        

def deserialize(self, data):
    """Decodes your encoded data to tree.
        
    :type data: str
    :rtype: TreeNode
    """
    def dfs():
        val = next(vals)
        if val == "#":
            return None
        node = TreeNode(int(val))
        node.left = dfs()
        node.right = dfs()
        return node

    vals = iter(data.split())
    return dfs()