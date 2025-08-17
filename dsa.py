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


from collections import Counter
import heapq as hq
def leastInterval(self, tasks: List[str], n: int):
        
    if n == 0:
        return len(tasks)

    count = Counter(tasks)
    maxHeap = [-cnt for cnt in count.values()]
    hq.heapify(maxHeap)
    time = 0

    while maxHeap:
        temp = []
        cycle = 0

        for _ in range(n +1):
            if maxHeap:
                freq = hq.heappop(maxHeap)

                if (freq + 1) != 0:
                    temp.append(freq+ 1)
                cycle += 1
            elif temp:
                cycle += 1
        for tmp in temp:
            hq.heappush(maxHeap, tmp)
            
        time += cycle if not maxHeap else n + 1

        return time

class Twitter:

    def __init__(self):
        self.time = 0
        self.tweets = defaultdict(list)
        self.following = defaultdict(set)

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.time += 1
        self.tweets[userId].append((self.time, tweetId))

    def getNewsFeed(self, userId: int) -> List[int]:
        heap = []

        for tweet in self.tweets[userId][-10:]:
            hq.heappush(heap, tweet)
        
        for followee in self.following[userId]:
            for t in self.tweets[followee][-10:]:
                hq.heappush(heap, t)
        
        return [tweetId for _,tweetId in hq.nlargest(10, heap)]

    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:
            self.following[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        if followeeId in self.following[followerId]:
            self.following[followerId].remove(followeeId)

import heapq as hq
class Solution:
    def getOrder(self, tasks: List[List[int]]) -> List[int]:
        itasks = [(et,pt,i) for i , (et, pt) in enumerate(tasks)]
        itasks.sort()

        res = []
        heap = []
        time = 0
        i = 0
        n = len(tasks)

        while i < n or heap:

            while i < n and itasks[i][0] <= time: # processing tasks
                enqueue_time, proc_time, index = itasks[i]
                hq.heappush(heap,(proc_time,index))
                i += 1

            if heap:
                proc_time, index = hq.heappop(heap)
                time += proc_time
                res.append(index)
            else:
                time = itasks[i][0] # next enqueued time

        return res
    def reorganizeString(self, s: str) -> str:
        
        freq = Counter(s)
        max_allowed = (len(s) + 1)//2
        if any( val > max_allowed for val in freq.values()):
            return ""

        heap = [(-cnt, char) for char, cnt in freq.items()]
        hq.heapify(heap)
        
        ans = []
        prev_count , prev_char = 0, ""
        while heap:
            cnt , char = hq.heappop(heap)
            ans.append(char)
            if prev_count < 0:
                hq.heappush(heap, (prev_count, prev_char))
            
            prev_count , prev_char = cnt + 1, char
            
        return "".join(ans)
    def longestDiverseString(self, a: int, b: int, c: int) -> str:
        heap = []
        if a > 0:
            hq.heappush(heap, (-a,'a'))
        if b > 0:
            hq.heappush(heap, (-b,'b'))
        if c > 0:
            hq.heappush(heap, (-c,'c'))
        
        res = []
        while heap:
            cnt1, char1 = hq.heappop(heap)

            if len(res)>= 2 and res[-1] == res[-2] == char1:
                if not heap:
                    break
                cnt2, char2 = hq.heappop(heap)
                res.append(char2)
                cnt2 += 1
                if cnt2 != 0:
                    hq.heappush(heap, (cnt2,char2))
                hq.heappush(heap, (cnt1,char1))
            else:
                res.append(char1)
                cnt1 += 1
                if cnt1 != 0:
                    hq.heappush(heap, (cnt1, char1))
        return "".join(res)
    def removeSubfolders(self, folder: List[str]) -> List[str]:
        
        folder.sort()
        res = []
        for dir in folder:
            if not res or not dir.startswith(res[-1]+ '/'):
                res.append(dir)
        return res
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        visited = set()
        rec_stack = set()

        graph  = defaultdict(list)
        for course , prereq in prerequisites:
            graph[course].append(prereq)

        def dfs(course):
            if course in rec_stack:
                return True
            if course in visited:
                return False

            visited.add(course)
            rec_stack.add(course)

            for prereq in graph[course]:
                if dfs(prereq):
                    return True
            rec_stack.remove(course)
            return False

        for i in range(numCourses):
            if dfs(i):
                return False
        return True

    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        visited = set()
        rec_stack = set()
        stack = []

        graph = defaultdict(list)
        for course, prereq in prerequisites:
            graph[course].append(prereq)
        
        def dfs(course):
            if course in rec_stack:
                return False
            if course in visited:
                return True
            
            rec_stack.add(course)
            for prereq in graph[course]:
                if prereq not in visited:
                    if not dfs(prereq):
                        return False

            visited.add(course)
            stack.append(course)
            rec_stack.remove(course)
            return True
            
        for i in range(numCourses):
            if i not in visited:
                if not dfs(i):
                    return []

        return stack
    
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        
        graph = {}

        for i in range(len(equations)):
            node1, node2 = equations[i][0], equations[i][1]
            weight = values[i]

            if node1 not in graph:
                graph[node1] = []
            if node2 not in graph:
                graph[node2] = []

            graph[node1].append((node2, weight))
            graph[node2].append((node1, 1 / weight))

        def bfs(graph, start_node, end_node):
            if start_node not in graph or end_node not in graph:
                return -1.0
            if start_node == end_node:
                return 1.0

            queue = deque([(start_node, 1.0)])
            visited = {start_node} 

            while queue:
                node, product_so_far = queue.popleft()

                if node == end_node:
                    return product_so_far

                if node in graph: 
                    for neighbor, weight in graph[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, product_so_far * weight))

            return -1.0

        res = []
        for query in queries:
            start_node, end_node = query[0], query[1]
            
            if start_node not in graph or end_node not in graph:
                res.append(-1.0)
            elif start_node == end_node:
                res.append(1.0)
            else:
                res.append(bfs(graph, start_node, end_node))

        return res

    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        
        rows = len(heights)
        cols = len(heights[0])

        pset = set()
        aset = set()

        def dfs(r,c, visited, prev_height):

            #Base case
            if not (0<= r<rows and 0 <= c < cols) or not heights[r][c] >= prev_height or (r,c) in visited:
                return 
            
            visited.add((r,c))
    
            dfs(r -1, c, visited, heights[r][c])
            dfs(r +1, c, visited, heights[r][c])
            dfs(r, c -1, visited,  heights[r][c])
            dfs(r, c +1, visited, heights[r][c])
        
        # starting dfs on pacific borders
        for r in range(rows):
            dfs(r,0,pset, heights[r][0])
        for c in range(cols):
            dfs(0,c,pset, heights[0][c])

        #starting dfs on atlantic borders
        for r in range(rows):
            dfs(r,cols-1,aset, heights[r][cols-1])
        for c in range(cols):
            dfs(rows-1,c,aset, heights[rows-1][c])

        
        res = []
        for r in range(rows):
            for c in range(cols):
                if (r,c) in aset and (r, c) in pset:
                    res.append([r,c])
        return res

    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        rows = len(board)
        cols = len(board[0])

        def dfs(r,c):

            if not ( 0 <= r < rows and 0 <= c < cols) or board[r][c] != "O":
                return

            board[r][c] = "T"

            dfs(r-1,c)
            dfs(r+1,c)
            dfs(r,c-1)
            dfs(r,c+1)
        
        #starting on top and bottom
        for c in range(cols):
            if board[0][c] == "O":
                dfs(0,c)
            if board[rows-1][c] == "O":
                dfs(rows-1,c)
        
        # checking the vertical borders
        for r in range(rows):
            if board[r][0] == "O":
                dfs(r,0)
            if board[r][cols-1] == "O":
                dfs(r,cols-1)

        for r in range(rows):
            for c in range(cols):
                if board[r][c] == "O":
                    board[r][c] = "X"
                elif board[r][c] == "T":
                    board[r][c] = "O"
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        visited = set()
        rec_stack = set()
        stack = []

        graph = defaultdict(list)
        for course, prereq in prerequisites:
            graph[course].append(prereq)
        
        def dfs(course):
            if course in rec_stack:
                return False
            if course in visited:
                return True
            
            rec_stack.add(course)
            for prereq in graph[course]:
                if prereq not in visited:
                    if not dfs(prereq):
                        return False

            visited.add(course)
            stack.append(course)
            rec_stack.remove(course)
            return True
            
        for i in range(numCourses):
            if i not in visited:
                if not dfs(i):
                    return []

        return stack   
    
    def validTree(self, n: int, edges: List[List[int]]) -> bool:

        graph = defaultdict(list)

        for u , v in edges:
            graph[u].append(v)
            graph[v].append(u)

        visited = set()

        if len(edges) != n -1:
            return False

        def cycle(node, parent):

            visited.add(node)

            for child in graph[node]:
                if child not in visited:
                    if cycle(child, node):
                        return True
                elif child != parent:
                    return True

            return False
        
        has_cycle = cycle(0,-1)

        is_connected = len(visited) == n

        return not has_cycle and is_connected
    
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        
        graph = defaultdict(list)

        for u,v in edges:
            graph[u].append(v)
            graph[v].append(u)

        visited = set()

        def dfs(node):

            visited.add(node)

            for nbr in graph[node]:
                if nbr not in visited:
                    dfs(nbr)

        count  = 0
        for i in range(n):
            if i not in visited:
                count += 1
                dfs(i)
        return count